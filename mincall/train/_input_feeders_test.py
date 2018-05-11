import unittest
import itertools
import threading
import pickle
from minion_data import dataset_pb2
import gzip
import queue
from mincall.train import _input_feeders
import os

mincall_root_folder = os.path.dirname(os.path.dirname(__file__))
update_golden = False


class TestInputFeeders(unittest.TestCase):
    def test_simple(self):
        ex_fname = os.path.join(
            mincall_root_folder,
            "example",
            "example.datapoint",
        )
        golden_fn = os.path.join(
            os.path.dirname(__file__),
            "test_simple.golden"
        )

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


if __name__ == "__main__":
    unittest.main()