from typing import *
from mincall.common import *
import numpy as np
import voluptuous


class Param(NamedTuple):
    min: float
    max: float
    type: str

    @classmethod
    def scheme(cls, data):
        if isinstance(data, (int, float, np.int, np.float, bool)):
            return data
        return named_tuple_helper(
            cls, {"type": voluptuous.validators.In([
                "int",
                "double",
            ])}, data
        )


class Observation(NamedTuple):
    metric: float
    metadata: Dict = {}
