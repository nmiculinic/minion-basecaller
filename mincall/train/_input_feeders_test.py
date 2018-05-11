import unittest
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
            os.path.dirname(__file__), "test_simple.golden")

        with gzip.open(ex_fname, "r") as f:
            dp = dataset_pb2.DataPoint()
            dp.ParseFromString(f.read())

        q = queue.Queue()
        p = queue.Queue()
        cfg: _input_feeders.InputFeederCfg = _input_feeders.InputFeederCfg(
            batch_size=None,
            seq_length=50,
        )
        _input_feeders.produce_datapoints(
            cfg,
            fnames=[ex_fname],
            q=q,
            poison=p,
            repeat=False,
        )
        got = []
        for i in itertools.count():
            try:
                x = q.get_nowait()
                if isinstance(x, ValueError):
                    got.append("ValueError")
            except queue.Empty:
                break
        if update_golden:
            golden = got
            with open(golden_fn, 'wb') as f:
                pickle.dump(golden, f)
        else:
            with open(golden_fn, 'rb') as f:
                golden = pickle.load(f)

        self.assertListEqual(got, golden)

    def test_processing(self):
        g = tf.Graph()
        with g.as_default():
            dq = _input_feeders.DataQueue(
                cfg=_input_feeders.InputFeederCfg(
                    batch_size=2,
                    seq_length=None,
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
            np.testing.assert_allclose(batch_labels.indices,
                                       np.array([[0, 0], [0, 1], [0, 2],
                                                 [1, 0], [1, 1]]))
            np.testing.assert_allclose(batch_labels.values,
                                       np.array([1, 2, 3, 10, 20]))
            np.testing.assert_allclose(batch_labels.dense_shape,
                                       np.array([2, 3]))
            np.testing.assert_allclose(signal,
                                       np.array([[1, 2, 3, 4, 5, 0, 0],
                                                 [10, 20, 30, 40, 50, 60,
                                                  70]]).reshape(2, 7, 1))
            np.testing.assert_allclose(signal_len, np.array([5, 7]))


if __name__ == "__main__":
    unittest.main()
