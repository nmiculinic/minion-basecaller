from mincall import dataset_pb2
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


def produce_datapoints(cfg: InputFeederCfg, fnames: List[str], q: Queue):
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
