from mincall import dataset_pb2
import tensorflow as tf
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


class InputFeederCfg(NamedTuple):
    batch_size: int
    seq_length: int

    @classmethod
    def schema(cls, data):
        return cls(
            **voluptuous.Schema({
                voluptuous.Optional('batch_size', 10): int,
                'seq_length': int,
            })(data))


class DataQueue():
    def __init__(self,
                 cfg: InputFeederCfg,
                 fnames,
                 capacity=-1,
                 min_after_deque=10,
                 shuffle=True,
                 trace=False,):
        """
        :param cap: queue capacity
        :param batch_size: output batch size
        """
        self.cfg = cfg
        self.fnames = fnames
        self.logger = logging.getLogger(__name__)
        self.values_ph = tf.placeholder(
            dtype=tf.int32, shape=[None], name="labels")
        self.values_len_ph = tf.placeholder(
            dtype=tf.int64, shape=[], name="labels_len")
        self.signal_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, 1], name="signal")
        self.signal_len_ph = tf.placeholder(
            dtype=tf.int64, shape=[], name="signal_len")

        self.queue = tf.PaddingFIFOQueue(
            capacity=capacity,
            dtypes=[tf.int32, tf.int64, tf.float32, tf.int64],
            shapes=[
                [None],
                [],
                [None, 1],
                [],
            ])

        if shuffle:
            self.shuffle_queue = tf.RandomShuffleQueue(
                capacity=capacity,
                dtypes=[tf.int32, tf.int64, tf.float32, tf.int64],
                min_after_dequeue=10,
            )
            self.enq = self.shuffle_queue.enqueue([
                self.values_ph, self.values_len_ph, self.signal_ph,
                self.signal_len_ph
            ])
            num_threads = 4
            qr = tf.train.QueueRunner(
                self.queue, [self.queue.enqueue(self.shuffle_queue.dequeue())
                             ] * num_threads)
            tf.train.add_queue_runner(qr)
        else:
            self.enq = self.queue.enqueue([
                self.values_ph, self.values_len_ph, self.signal_ph,
                self.signal_len_ph
            ])

        values_op, values_len_op, signal_op, signal_len_op = self.queue.dequeue_many(
            cfg.batch_size)
        if trace:
            values_op = tf.Print(
                values_op, [values_op, tf.shape(values_op)],
                message="values op")
            values_len_op = tf.Print(
                values_len_op,
                [values_len_op, tf.shape(values_len_op)],
                message="values len op")

        sp = []
        for label_idx, label_len in zip(
                tf.split(values_op, cfg.batch_size),
                tf.split(values_len_op, cfg.batch_size)):
            label_len = tf.squeeze(label_len, axis=0)
            ind = tf.transpose(
                tf.stack([
                    tf.zeros(shape=label_len, dtype=tf.int64),
                    tf.range(label_len, dtype=tf.int64),
                ]))

            sp.append(
                tf.SparseTensor(
                    indices=ind,
                    values=tf.squeeze(label_idx, axis=0)[:label_len],
                    dense_shape=tf.stack([1, tf.maximum(label_len, 1)], 0)))

        # labels: An `int32` `SparseTensor`.
        # `labels.indices[i, :] == [b, t]` means `labels.values[i]` stores
        # the id for (batch b, time t).
        #    `labels.values[i]` must take on values in `[0, num_labels)`.
        # See `core/ops/ctc_ops.cc` for more details.
        # That's ok implemented

        self.batch_labels = tf.sparse_concat(
            axis=0, sp_inputs=sp, expand_nonconcat_dim=True)
        self.batch_labels_len = values_len_op
        self.batch_dense_labels = tf.sparse_to_dense(
            sparse_indices=self.batch_labels.indices,
            sparse_values=self.batch_labels.values,
            output_shape=self.batch_labels.dense_shape,
            default_value=9,
        )
        self.batch_signal = signal_op
        if trace:
            self.batch_signal = tf.Print(self.batch_signal, [
                self.batch_dense_labels,
                tf.shape(self.batch_dense_labels),
                tf.shape(signal_op)
            ], "dense labels")

        self.batch_signal_len = signal_len_op

    def push_to_queue(self, sess: tf.Session, signal: np.ndarray,
                      label: np.ndarray):
        sess.run(
            self.enq,
            feed_dict={
                self.values_ph: label,
                self.values_len_ph: len(label),
                self.signal_ph: signal.reshape((-1, 1)),
                self.signal_len_ph: len(signal),
            })

    def start_input_processes(self, sess: tf.Session, cnt=1):
        m = Manager()
        q: Queue = m.Queue()
        poison_queue: Queue = m.Queue()

        processes: List[Process] = []
        for _ in range(cnt):
            p = Process(
                target=produce_datapoints,
                args=(self.cfg, self.fnames, q, poison_queue))
            p.start()
            processes.append(p)

        def worker_fn():
            exs = 0
            for i in count(start=1):
                try:
                    poison_queue.get_nowait()
                    return
                except queue.Empty:
                    pass

                it = q.get(timeout=0.5)
                if isinstance(it, Exception):
                    self.logger.debug(
                        f"Exception happened during processing data {type(it).__name__}:\n{it}"
                    )
                    exs += 1
                    continue
                signal, labels = it
                self.push_to_queue(sess, signal, labels)
                if i % 2000 == 0:
                    self.logger.info(
                        f"sucessfully submitted {i - exs}/{i} samples; -- {(i-exs)/i:.2f}"
                    )

        th = Thread(target=worker_fn, daemon=True)
        th.start()

        def close():
            for _ in range(cnt + 1):
                poison_queue.put(None)
            for p in processes:
                p.join()
            th.join()

        return close


def produce_datapoints(cfg: InputFeederCfg, fnames: List[str], q: Queue,
                       poison: Queue):
    """

    Pushes single instances to the queue of the form:
        signal[None,], labels [None,]
    That is 1D numpy array
    :param cfg:
    :param fnames:
    :param q:
    :return:
    """
    while True:
        random.seed(os.urandom(20))
        random.shuffle(fnames)
        for x in fnames:
            with gzip.open(x, "r") as f:
                dp = dataset_pb2.DataPoint()
                dp.ParseFromString(f.read())

                signal = np.array(dp.signal, dtype=np.float32)
                buff = np.zeros(cfg.seq_length, dtype=np.int32)

                basecall_idx = 0
                basecall_squiggle_idx = 0
                for start in range(0, len(signal), cfg.seq_length):
                    buff_idx = 0
                    while basecall_idx < len(
                            dp.basecalled
                    ) and dp.lower_bound[basecall_idx] < start + cfg.seq_length:
                        while basecall_squiggle_idx < len(
                                dp.basecalled_squiggle
                        ) and dp.basecalled_squiggle[basecall_squiggle_idx] == dataset_pb2.BLANK:  # Find first non-blank basecall_squiggle
                            basecall_squiggle_idx += 1

                        if basecall_squiggle_idx >= len(
                                dp.basecalled_squiggle):
                            break
                        else:
                            buff[buff_idx] = dp.basecalled_squiggle[
                                basecall_squiggle_idx]
                            buff_idx += 1
                            basecall_squiggle_idx += 1
                            basecall_idx += 1
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
