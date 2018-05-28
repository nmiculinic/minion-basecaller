from minion_data import dataset_pb2
import tensorflow as tf
import itertools
import logging
from typing import *
import voluptuous
from itertools import count
import os
import random
import gzip
import numpy as np
from multiprocessing import Queue, Manager, Process
import queue
from threading import Thread
import sys


class InputFeederCfg(NamedTuple):
    batch_size: int
    seq_length: int
    ratio: int
    surrogate_base_pair: bool
    num_bases: int
    min_signal_size: int = 10000

    @classmethod
    def schema(cls, data):
        return cls(
            **voluptuous.Schema({
                voluptuous.Optional('batch_size', 10): int,
                'seq_length': int,
                'surrogate_base_pair': bool,
                voluptuous.Optional("min_signal_size"): int,
                voluptuous.Optional("num_bases"): int,
            })(data)
        )


class DataQueue():
    def __init__(
        self,
        cfg: InputFeederCfg,
        fnames,
        capacity=-1,
        min_after_deque=10,
        shuffle=True,
        trace=False,
    ):
        """
        :param cap: queue capacity
        :param batch_size: output batch size
        """
        self.cfg = cfg
        self.fnames = fnames
        self.logger = logging.getLogger(__name__)
        self._values_ph = tf.placeholder(
            dtype=tf.int32, shape=[None], name="labels"
        )
        self._values_len_ph = tf.placeholder(
            dtype=tf.int64, shape=[], name="labels_len"
        )
        self._signal_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, 1], name="signal"
        )
        self._signal_len_ph = tf.placeholder(
            dtype=tf.int64, shape=[], name="signal_len"
        )

        self.closing = []
        self.queue = tf.PaddingFIFOQueue(
            capacity=capacity,
            dtypes=[tf.int32, tf.int64, tf.float32, tf.int64],
            shapes=[
                [None],
                [],
                [None, 1],
                [],
            ]
        )
        # self.closing.append(self.queue.close()) Closed with queue runners...no idea how&why it works
        self.summaries = []
        self.summaries.append(
            tf.summary.scalar(
                "paddingFIFOQueue_input", self.queue.size(), family="queue"
            )
        )

        if shuffle:
            self.shuffle_queue = tf.RandomShuffleQueue(
                capacity=capacity,
                dtypes=[tf.int32, tf.int64, tf.float32, tf.int64],
                min_after_dequeue=min_after_deque,
            )
            self.enq = self.shuffle_queue.enqueue([
                self._values_ph, self._values_len_ph, self._signal_ph,
                self._signal_len_ph
            ])
            num_threads = 4
            qr = tf.train.QueueRunner(
                self.queue,
                [self.queue.enqueue(self.shuffle_queue.dequeue())] * num_threads
            )
            tf.train.add_queue_runner(qr)
            self.closing.append(
                self.shuffle_queue.close(cancel_pending_enqueues=True)
            )
            self.summaries.append(
                tf.summary.scalar(
                    "randomShuffleQueue_input",
                    self.queue.size(),
                    family="queue"
                )
            )
        else:
            self.enq = self.queue.enqueue([
                self._values_ph, self._values_len_ph, self._signal_ph,
                self._signal_len_ph
            ])

        values_op, values_len_op, signal_op, signal_len_op = self.queue.dequeue_many(
            cfg.batch_size
        )
        if trace:
            values_op = tf.Print(
                values_op, [values_op, tf.shape(values_op)],
                message="values op"
            )
            values_len_op = tf.Print(
                values_len_op,
                [values_len_op, tf.shape(values_len_op)],
                message="values len op"
            )

        sp = []
        for label_idx, label_len in zip(
            tf.split(values_op, cfg.batch_size),
            tf.split(values_len_op, cfg.batch_size)
        ):
            label_len = tf.squeeze(label_len, axis=0)
            ind = tf.transpose(
                tf.stack([
                    tf.zeros(shape=label_len, dtype=tf.int64),
                    tf.range(label_len, dtype=tf.int64),
                ])
            )

            sp.append(
                tf.SparseTensor(
                    indices=ind,
                    values=tf.squeeze(label_idx, axis=0)[:label_len],
                    dense_shape=tf.stack([1, tf.maximum(label_len, 1)], 0)
                )
            )

        # labels: An `int32` `SparseTensor`.
        # `labels.indices[i, :] == [b, t]` means `labels.values[i]` stores
        # the id for (batch b, time t).
        #    `labels.values[i]` must take on values in `[0, num_labels)`.
        # See `core/ops/ctc_ops.cc` for more details.
        # That's ok implemented

        self.batch_labels = tf.sparse_concat(
            axis=0, sp_inputs=sp, expand_nonconcat_dim=True
        )
        self.batch_labels_len = values_len_op
        self.batch_dense_labels = tf.sparse_to_dense(
            sparse_indices=self.batch_labels.indices,
            sparse_values=self.batch_labels.values,
            output_shape=self.batch_labels.dense_shape,
            default_value=9,
        )
        self.batch_signal = signal_op
        if trace:
            self.batch_signal = tf.Print(
                self.batch_signal, [
                    self.batch_dense_labels,
                    tf.shape(self.batch_dense_labels),
                    tf.shape(signal_op)
                ], "dense labels"
            )

        self.batch_signal_len = signal_len_op

    def push_to_queue(
        self, sess: tf.Session, signal: np.ndarray, label: np.ndarray
    ):
        if self.cfg.surrogate_base_pair:
            for i in range(1, len(label)):
                if label[i - 1] == label[i]:
                    label[i] += self.cfg.num_bases

        sess.run(
            self.enq,
            feed_dict={
                self._values_ph: label,
                self._values_len_ph: len(label),
                self._signal_ph: signal.reshape((-1, 1)),
                self._signal_len_ph: len(signal),
            }
        )

    def start_input_processes(
        self, sess: tf.Session, coord: tf.train.Coordinator, cnt=1
    ):
        class Wrapper():
            def __init__(self):
                pass

            def __enter__(iself):
                m = Manager()
                q: Queue = m.Queue()
                iself.poison_queue: Queue = m.Queue()

                iself.processes: List[Process] = []
                for _ in range(cnt):
                    p = Process(
                        target=produce_datapoints,
                        args=(self.cfg, self.fnames, q, iself.poison_queue)
                    )
                    p.start()
                    iself.processes.append(p)

                def worker_fn():
                    exs = 0
                    for i in count(start=1):
                        try:
                            iself.poison_queue.get_nowait()
                            return
                        except queue.Empty:
                            pass
                        if coord.should_stop():
                            return
                        try:
                            it = q.get(timeout=0.5)
                            if isinstance(it, Exception):
                                self.logger.debug(
                                    f"Exception happened during processing data {type(it).__name__}:\n{it}"
                                )
                                exs += 1
                                continue
                            signal, labels = it
                            try:
                                self.push_to_queue(sess, signal, labels)
                            except tf.errors.CancelledError:
                                if coord.should_stop():
                                    return
                                else:
                                    raise
                            if i % 2000 == 0:
                                self.logger.info(
                                    f"sucessfully submitted {i - exs}/{i} samples; -- {(i-exs)/i:.2f}"
                                )
                        except queue.Empty:
                            pass

                iself.th = Thread(target=worker_fn, daemon=True)
                iself.th.start()
                logging.getLogger(__name__).info("Started all feeders")

            def __exit__(iself, exc_type, exc_val, exc_tb):
                logging.getLogger(__name__
                                 ).info("Starting to close all feeders")
                for x in self.closing:
                    try:
                        sess.run(x)
                    except Exception as ex:
                        logging.getLogger(__name__).warning(
                            f"Cannot close queue {type(ex).__name__}: {ex}"
                        )
                        pass
                logging.getLogger(__name__).info("Closed all queues")
                for _ in range(cnt + 1):
                    iself.poison_queue.put(None)
                for p in iself.processes:
                    p.join(timeout=5)
                iself.th.join(timeout=5)
                logging.getLogger(__name__).info("Closed all feeders")

        return Wrapper()


def produce_datapoints(
    cfg: InputFeederCfg,
    fnames: List[str],
    q: Queue,
    poison: Queue,
    repeat=True
):
    """

    Pushes single instances to the queue of the form:
        signal[None,], labels [None,]
    That is 1D numpy array
    :param cfg:
    :param fnames:
    :param q:
    :return:
    """
    for cnt in itertools.count(1):
        random.seed(os.urandom(20))
        random.shuffle(fnames)
        for x in fnames:
            with gzip.open(x, "r") as f:
                dp = dataset_pb2.DataPoint()
                dp.ParseFromString(f.read())
                signal = np.array(dp.signal, dtype=np.float32)
                if len(signal) < cfg.min_signal_size:
                    q.put(
                        ValueError(
                            f"Signal too short {len(dp.signal)} < {cfg.min_signal_size}"
                        )
                    )
                    continue

                buff = np.zeros(cfg.seq_length, dtype=np.int32)

                label_idx = 0
                for start in range(0, len(signal), cfg.seq_length):
                    while label_idx < len(
                        dp.labels
                    ) and dp.labels[label_idx].upper < start:
                        label_idx += 1
                    buff_idx = 0
                    while label_idx < len(
                        dp.labels
                    ) and dp.labels[label_idx].lower < start + cfg.seq_length:
                        buff[buff_idx] = dp.labels[label_idx].pair
                        buff_idx += 1
                        label_idx += 1
                    try:
                        poison.get_nowait()
                        return
                    except queue.Empty:
                        pass
                    if buff_idx == 0:
                        q.put(ValueError("Empty labels"))
                    else:
                        q.put([
                            signal[start:start + cfg.seq_length],
                            np.copy(buff[:buff_idx]),
                        ])
        if not repeat:
            break
