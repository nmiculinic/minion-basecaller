from typing import *
from mincall.common import named_tuple_helper
import voluptuous

__all__ = ["BasecallCfg"]


class BasecallCfg(NamedTuple):
    input_dir: List[str]
    output_fasta: str
    model: str
    logdir: str = None
    jump: int = 80000
    gzip: bool = False
    recursive: bool = False
    batch_size: int = 1
    seq_length: int = 100000
    beam_width: int = 50

    @classmethod
    def schema(cls, data):
        return named_tuple_helper(cls, {
            'input_dir':
                voluptuous.All(
                    voluptuous.validators.Length(min=1),
                    [
                        voluptuous.
                            Any(voluptuous.IsDir(), voluptuous.IsFile())
                    ],
                            ),
            'model':
                voluptuous.validators.IsFile(),
        }, data)
