from typing import *
import voluptuous


class DataDir(NamedTuple):
    name: str
    dir: str

    @classmethod
    def schema(cls, data):
        return cls(
            **voluptuous.Schema({
                'name': str,
                'dir': voluptuous.validators.IsDir(),
            },
                                required=True)(data)
        )


class InputFeederCfg(NamedTuple):
    batch_size: int
    seq_length: int
    ratio: int
    surrogate_base_pair: bool
    num_bases: int
    min_signal_size: int = 10000
    max_label_size: int = 1_000_000_000  # Unrealistically large number

    @classmethod
    def schema(cls, data):
        return cls(
            **voluptuous.Schema({
                voluptuous.Optional('batch_size', 10): int,
                'seq_length': int,
                'surrogate_base_pair': bool,
                voluptuous.Optional("min_signal_size"): int,
                voluptuous.Optional("num_bases"): int,
            })(data)
        )
