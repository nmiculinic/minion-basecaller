import unittest
import logging
import os
import numpy as np
from mincall.train import ops
from typing import *

ops_data = os.path.join(os.path.dirname(__file__), "ops_data")
update_golden = False


class TestAlignmentStats(unittest.TestCase):
    def test_end2end_empty_query(self):
        x = np.load(os.path.join(ops_data, "end2end_empty_query.npz"))
        data = {k: x[k] for k in x.files}
        res = ops.alignment_stats(**data)
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
