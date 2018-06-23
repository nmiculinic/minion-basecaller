from typing import *
import voluptuous

__all__ = ["BasecallCfg"]


class BasecallCfg(NamedTuple):
    input_dir: List[str]
    recursive: bool
    output_fasta: str
    model: str
    batch_size: int
    seq_length: int
    beam_width: int
    logdir: str
    jump: int
    gzip: bool

    @classmethod
    def schema(cls, data):
        return cls(
            **voluptuous.Schema({
                'input_dir':
                    voluptuous.All(
                        voluptuous.validators.Length(min=1),
                        [
                            voluptuous.
                                Any(voluptuous.IsDir(), voluptuous.IsFile())
                        ],
                                ),
                voluptuous.Optional('recursive', default=False):
                    bool,
                'output_fasta':
                    str,
                'model':
                    voluptuous.validators.IsFile(),
                voluptuous.Optional('batch_size', default=1100):
                    int,
                voluptuous.Optional('seq_length', default=300):
                    int,
                voluptuous.Optional('beam_width', default=50):
                    int,
                voluptuous.Optional('jump', default=30):
                    int,
                voluptuous.Optional('logdir', default=None):
                    voluptuous.Any(str, None),
                voluptuous.Optional('gzip', default=False):
                    bool,
            },
                required=True)(data)
        )
