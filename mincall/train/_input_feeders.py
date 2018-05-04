from mincall import dataset_pb2
from typing import *
import voluptuous
import os
import random
import gzip

class InputFeederCfg(NamedTuple):
    batch_size: int
    seq_length: int

    @classmethod
    def schema(cls, data):
        return cls(**voluptuous.Schema({
            voluptuous.Optional('batch_size', 10): int,
            'seq_length': int,
        })(data))

def produce_datapoints(cfg: InputFeederCfg, fnames: List[str]):
    random.seed(os.urandom(20))
    random.shuffle(fnames)
    for x in fnames:
        with gzip.open(x, "r") as f:
            dp = dataset_pb2.DataPoint()
            dp.ParseFromString(f.read())
            print(dp)
