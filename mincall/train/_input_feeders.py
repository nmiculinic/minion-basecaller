from mincall import dataset_pb2
import tensorflow as tf
from typing import *
import voluptuous
import os
import random
import gzip
import numpy as np
from multiprocessing import Queue


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
    def __init__(self, batch_size, capacity=-1):
        """
        :param cap: queue capacity
        :param batch_size: output batch size
        """
        self.values_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="labels")
        self.values_len_ph = tf.placeholder(dtype=tf.int64, shape=[], name="labels_len")
        self.signal_ph = tf.placeholder(dtype=tf.float64, shape=[None], name="signal")
        self.signal_len_ph = tf.placeholder(dtype=tf.int64, shape=[], name="signal_len")

        self.queue = tf.PaddingFIFOQueue(
            capacity=capacity,
            dtypes=[tf.int32, tf.int64, tf.float64, tf.int64],
            shapes=[
                [None],
                [],
                [None],
                [],
            ])
        self.enq = self.queue.enqueue([self.values_ph, self.values_len_ph, self.signal_ph, self.signal_len_ph])

        values_op, values_len_op, signal_op, signal_len_op = self.queue.dequeue_many(batch_size)
        sp = []
        for label_idx, label_len in zip(tf.split(values_op, batch_size), tf.split(values_len_op, batch_size)):
            label_len = tf.squeeze(label_len, axis=0)
            ind = tf.transpose(
                tf.stack(
                    [
                        tf.zeros(shape=label_len, dtype=tf.int64),
                        tf.range(label_len, dtype=tf.int64),
                    ]
                ))

            print(ind, ind.shape, ind.dtype)
            sp.append(
                tf.SparseTensor(
                    indices=ind,
                    values=tf.squeeze(label_idx, axis=0)[:label_len],
                    dense_shape=tf.stack([1,label_len], 0)
                )
            )

        self.batch_labels = tf.sparse_concat(axis=0, sp_inputs=sp, expand_nonconcat_dim=True)
        self.batch_labels_len = values_len_op
        self.batch_signal = signal_op
        self.batch_signal_len = signal_len_op

    def push_to_queue(self, sess: tf.Session, signal: np.ndarray, label: np.ndarray):
        sess.run(
            self.enq, feed_dict={
                self.values_ph:label,
                self.values_len_ph: len(label),
                self.signal_ph:signal,
                self.signal_len_ph:len(signal),
            }
        )


def produce_datapoints(cfg: InputFeederCfg, fnames: List[str], q: Queue):
    """

    Pushes single instances to the queue of the form:
        signal[None,], labels [None,]
    That is 1D numpy array
    :param cfg:
    :param fnames:
    :param q:
    :return:
    """
    # TODO: Check correctness
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

                    if basecall_squiggle_idx >= len(dp.basecalled_squiggle):
                        break
                    else:
                        buff[buff_idx] = dp.basecalled_squiggle[
                            basecall_squiggle_idx]
                        buff_idx += 1
                        basecall_squiggle_idx += 1
                        basecall_idx += 1
                q.put([
                    signal[start:start + cfg.seq_length],
                    np.copy(buff[:buff_idx]),
                ])
