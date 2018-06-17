import unittest
from typing import *
import logging
import tempfile
import numpy as np
import itertools
import threading
import pickle
from minion_data import dataset_pb2
import gzip
import queue
from mincall.train import _input_feeders
import os
import tensorflow as tf

mincall_root_folder = os.path.dirname(os.path.dirname(__file__))
update_golden = False

ex_fname = os.path.join(
    mincall_root_folder,
    "example",
    "example.datapoint",
)


class TestInputFeeders(unittest.TestCase):
    def test_simple(self):
        golden_fn = os.path.join(
            os.path.dirname(__file__), "test_simple.golden"
        )

        cfg: _input_feeders.InputFeederCfg = _input_feeders.InputFeederCfg(
            batch_size=None,
            seq_length=50,
            ratio=1,
            num_bases=4,
            surrogate_base_pair=False,
        )
        got = []
        for x in _input_feeders.produce_datapoints(
                cfg,
                fnames=[ex_fname],
                repeat=False,
            ):
            if isinstance(x, ValueError):
                got.append("ValueError")
            else:
                got.append(x)
        if update_golden:
            golden = got
            with open(golden_fn, 'wb') as f:
                pickle.dump(golden, f)
        else:
            with open(golden_fn, 'rb') as f:
                golden = pickle.load(f)

        self.assertEqual(len(got), len(golden))
        for a, b in zip(got, golden):
            if isinstance(a, list):
                assert len(a) == len(b)
                for aa, bb in zip(a, b):
                    np.testing.assert_allclose(aa, bb)
            else:
                self.assertEqual(a, b)

    def test_processing(self):
        g = tf.Graph()
        with g.as_default():
            dq = _input_feeders.DataQueue(
                cfg=_input_feeders.InputFeederCfg(
                    batch_size=2,
                    seq_length=None,
                    ratio=1,
                    num_bases=4,
                    surrogate_base_pair=False,
                ),
                fnames=None,
                min_after_deque=0,
                shuffle=False,
            )
        with tf.Session(graph=g) as sess:
            dq.push_to_queue(
                sess,
                np.array([1, 2, 3, 4, 5]),
                np.array([1, 2, 3]),
            )

            dq.push_to_queue(
                sess,
                np.array([10, 20, 30, 40, 50, 60, 70]),
                np.array([10, 20]),
            )
            batch_labels, batch_dense_labels, signal, signal_len = sess.run([
                dq.batch_labels, dq.batch_dense_labels, dq.batch_signal,
                dq.batch_signal_len
            ])
            np.testing.assert_allclose(
                batch_labels.indices,
                np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]])
            )
            np.testing.assert_allclose(
                batch_labels.values, np.array([1, 2, 3, 10, 20])
            )
            np.testing.assert_allclose(
                batch_labels.dense_shape, np.array([2, 3])
            )
            np.testing.assert_allclose(
                signal,
                np.array([[1, 2, 3, 4, 5, 0, 0], [10, 20, 30, 40, 50, 60,
                                                  70]]).reshape(2, 7, 1)
            )
            np.testing.assert_allclose(signal_len, np.array([5, 7]))

    def test_end2end(self):
        g = tf.Graph()
        with g.as_default():
            dq = _input_feeders.DataQueue(
                cfg=_input_feeders.InputFeederCfg(
                    batch_size=2,
                    seq_length=50,
                    ratio=1,
                    num_bases=4,
                    surrogate_base_pair=False,
                ),
                fnames=[ex_fname],
                min_after_deque=40,
                shuffle=True,
            )
        with tf.Session(graph=g) as sess:
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=sess, coord=coord)
            with dq.start_input_processes(sess, coord=coord):
                for _ in range(10):
                    batch_labels, signal, signal_len = sess.run([
                        dq.batch_labels, dq.batch_signal, dq.batch_signal_len
                    ])
                coord.request_stop()

    def _create_valid_dp(self, seq: List[Tuple[int, int]]):
        signal = []
        labels = []

        for bp, l in seq:
            signal.extend([float(bp)] * l)
            labels.append(
                dataset_pb2.DataPoint.BPConfidenceInterval(
                    lower=len(signal) - l,
                    upper=len(signal),
                    pair=bp,
                )
            )
        return dataset_pb2.DataPoint(
            signal=signal,
            labels=labels,
        )

    def test_ok_process(self):
        _, t = tempfile.mkstemp()
        try:
            with gzip.open(t, "w") as f:
                f.write(
                    self._create_valid_dp([
                        (dataset_pb2.A, 10),
                        (dataset_pb2.A, 10),
                        (dataset_pb2.A, 10),
                        (dataset_pb2.A, 10),
                        (dataset_pb2.C, 20),
                        (dataset_pb2.A, 10),
                        (dataset_pb2.C, 10),
                        (dataset_pb2.C, 10),
                        (dataset_pb2.C, 10),
                    ]).SerializeToString()
                )

            cfg: _input_feeders.InputFeederCfg = _input_feeders.InputFeederCfg(
                batch_size=None,
                seq_length=50,
                ratio=1,
                num_bases=4,
                surrogate_base_pair=False,
                min_signal_size=10,
            )
            got = []
            for x in  _input_feeders.produce_datapoints(
                cfg,
                fnames=[t],
                repeat=False,
            ):
                if isinstance(x, ValueError):
                    got.append("ValueError")
                else:
                    got.append(x)
            self.assertEqual(len(got), 2)
            np.testing.assert_equal(
                got[0][1],
                np.array(4 * [dataset_pb2.A] + [dataset_pb2.C], dtype=np.int32)
            )
            np.testing.assert_equal(
                got[1][1],
                np.array([dataset_pb2.A] + 3 * [dataset_pb2.C], dtype=np.int32)
            )
        finally:
            os.unlink(t)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
