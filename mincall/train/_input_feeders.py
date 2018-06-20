from minion_data import dataset_pb2
import tensorflow as tf
import itertools
import logging
from typing import *
import voluptuous
import os
import random
import gzip
import numpy as np
from threading import Thread
import scrappy


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
        capacity=10000,
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
                    assert label[i
                                ] < self.cfg.num_bases, "invalid base pair data"
                    label[i] += self.cfg.num_bases

        sess.run(
            self.enq,
            feed_dict={
                self._values_ph: label,
                self._values_len_ph: len(label),
                self._signal_ph: signal.reshape((-1, 1)),
                self._signal_len_ph: len(signal),
            },
            options=tf.RunOptions(
                timeout_in_ms=5 * 60 *
                1000,  # if nothing gets in the queue for 5min something is probably wrong
            )
        )

    def start_input_processes(
        self, sess: tf.Session, coord: tf.train.Coordinator
    ):
        class Wrapper():
            def __init__(self):
                pass

            def __enter__(iself):
                def worker_fn():
                    try:
                        exs = 0
                        for i, it in enumerate(
                            produce_datapoints(
                                cfg=self.cfg,
                                fnames=self.fnames,
                            )
                        ):
                            if isinstance(it, Exception):
                                self.logger.debug(
                                    f"Exception happened during processing data {type(it).__name__}:\n{it}"
                                )
                                exs += 1
                                continue
                            signal, labels = it
                            try:
                                self.logger.debug("about to push to tf queue")
                                self.push_to_queue(sess, signal, labels)
                            except tf.errors.CancelledError:
                                if coord.should_stop():
                                    self.logger.warning(
                                        "enqueue op canceled, and coord is stopping"
                                    )
                                    return
                                else:
                                    self.logger.error(
                                        "Cancelled error occurred yet coord is not stopping!",
                                        exc_info=True
                                    )
                                    raise
                            except tf.errors.DeadlineExceededError:
                                self.logger.warning(
                                    "Queue pushing timeout exceeded"
                                )
                            if i > 0 and i % 2000 == 0:
                                self.logger.info(
                                    f"sucessfully submitted {i - exs}/{i} samples; -- {(i-exs)/i:.2f}"
                                )
                    except Exception as e:
                        self.logger.critical(
                            f"{type(e).__name__}: {e}", exc_info=True
                        )
                        coord.request_stop(e)
                        raise

                iself.th = Thread(target=worker_fn, daemon=False)
                iself.th.start()
                logging.getLogger(__name__).info("Started all feeders")

            def __exit__(iself, exc_type, exc_val, exc_tb):
                if exc_val:
                    logging.getLogger(__name__).error(
                        f"Error happened and closing {exc_val}"
                    )
                    coord.request_stop(exc_val)
                logging.getLogger(__name__
                                 ).info("Starting to close all feeders")
                for x in self.closing:
                    try:
                        sess.run(x)
                    except Exception as ex:
                        logging.getLogger(__name__).error(
                            f"Cannot close queue {type(ex).__name__}: {ex}"
                        )
                logging.getLogger(__name__).info("Closed all queues")
                iself.th.join(timeout=5)
                if iself.th.is_alive():
                    logging.getLogger(__name__
                                     ).error("Input thread is still alive")
                else:
                    logging.getLogger(__name__).info("Closed all feeders")

        return Wrapper()


def produce_datapoints(cfg: InputFeederCfg, fnames: List[str], repeat=True):
    """

    Pushes single instances to the queue of the form:
        signal[None,], labels [None,]
    That is 1D numpy array
    :param cfg:
    :param fnames:
    :param q:
    :return:
    """
    random.seed(os.urandom(20))
    for cnt in itertools.count(1):
        random.shuffle(fnames)
        for x in fnames:
            with gzip.open(x, "r") as f:
                dp = dataset_pb2.DataPoint()
                dp.ParseFromString(f.read())
                signal = np.array(dp.signal, dtype=np.float32)
                signal = scrappy.RawTable(signal).scale().data(as_numpy=True)
                assert len(signal) == len(dp.signal), "Trimming occured"
                if len(signal) < cfg.min_signal_size:
                    yield ValueError(
                        f"Signal too short {len(dp.signal)} < {cfg.min_signal_size}"
                    )
                    continue

                label_idx = 0
                for start in range(0, len(signal), cfg.seq_length):
                    buff = []
                    while label_idx < len(
                        dp.labels
                    ) and dp.labels[label_idx].upper < start:
                        label_idx += 1
                    while label_idx < len(
                        dp.labels
                    ) and dp.labels[label_idx].lower < start + cfg.seq_length:
                        buff.append(dp.labels[label_idx].pair)

                        # Sanity check
                        assert start <= dp.labels[label_idx].lower
                        assert start <= dp.labels[label_idx].upper
                        assert dp.labels[label_idx
                                        ].lower <= start + cfg.seq_length

                        label_idx += 1

                    signal_segment = signal[start:start + cfg.seq_length]
                    if len(buff) == 0:
                        yield ValueError("Empty labels")
                    elif len(signal_segment) / cfg.ratio < len(buff):
                        yield ValueError(
                            f"max possible labels {signal_segment/cfg.ratio}, have {len(buff)} labels"
                        )
                    else:
                        logging.debug(f"produce_datapoints: yielding datapoint")
                        yield [
                            signal_segment,
                            np.array(buff, dtype=np.int32),
                        ]
        if not repeat:
            logging.info("Repeat is false, quiting")
            break
