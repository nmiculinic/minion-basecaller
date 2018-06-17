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
import numpy as np

mincall_root_folder = os.path.dirname(os.path.dirname(__file__))

ex_fname = os.path.join(
    mincall_root_folder,
    "example",
    "example.datapoint",
)


class TestAlignmentStats(unittest.TestCase):
    def test_end2end(self):
        x = np.load("/home/lpp/Desktop/pyop.npz")
        print(type(x))
        print(x)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
