from typing import *
from mincall.common import named_tuple_helper
import voluptuous
import os

__all__ = ["BasecallCfg"]

default_seq_len = 10_000_000


class BasecallCfg(NamedTuple):
    input_dir: List[str]
    output_fasta: str
    model: str
    logdir: str = None
    gzip: bool = False
    recursive: bool = False
    threads: int = os.cpu_count() or 4
    batch_size: int = 1
    seq_length: int = default_seq_len
    jump: int = default_seq_len - 3000
    beam_width: int = 50

    @classmethod
    def schema(cls, data):
        return named_tuple_helper(
            cls, {
                'input_dir':
                    voluptuous.All(
                        voluptuous.validators.Length(min=1),
                        [
                            voluptuous.Any(
                                voluptuous.IsDir(), voluptuous.IsFile()
                            )
                        ],
                    ),
                'model':
                    voluptuous.validators.IsFile(),
            }, data
        )
