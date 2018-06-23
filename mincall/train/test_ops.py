import unittest
import logging
import os
import numpy as np
from mincall.train import ops
from typing import *
import tensorflow as tf
from minion_data import dataset_pb2
import pandas as pd

ops_data = os.path.join(os.path.dirname(__file__), "ops_data")
update_golden = False


class TestAlignmentStats(unittest.TestCase):
    def test_end2end_empty_query(self):
        x = np.load(os.path.join(ops_data, "end2end_empty_query.npz"))
        data = {k: x[k] for k in x.files}
        res = ops.alignment_stats(**data)
        *astats, identity = res
        df = pd.DataFrame({
            "identity": identity,
            **{
                dataset_pb2.Cigar.Name(op): stat
                for op, stat in zip(
                ops.aligment_stats_ordering, astats
            )
            },
        })
        if update_golden:
            np.savez(
                os.path.join(ops_data, "end2end_empty_query.golden.npz"), *res
            )
        else:
            golden = np.load(
                os.path.join(ops_data, "end2end_empty_query.golden.npz")
            )
            golder_arr = [golden[k] for k in golden.files]
            self.assertEqual(len(res), len(golder_arr))
            for actual, desired in zip(res, golder_arr):
                np.testing.assert_allclose(actual, desired)


class TestDynamicMask(unittest.TestCase):
    def test_simple(self):
        with tf.Graph().as_default(), tf.Session() as sess:
            signal = tf.placeholder(tf.float32)
            signal_rec = tf.placeholder(tf.float32)
            signal_len = tf.placeholder(tf.int64, shape=(None,))

            op = ops.autoencoder_loss(
                signal=signal,
                signal_reconstruction=signal_rec,
                signal_len=signal_len,
            )

            sol = sess.run(
                op,
                feed_dict={
                    signal: np.arange(6).reshape([2, 3, 1]),
                    signal_rec: np.zeros([2, 3, 1]),
                    signal_len: np.array([1, 2]),
                }
            )
            np.testing.assert_allclose(sol, (0 + 9 + 16) / 2 / 3)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
