from ._types import Param
import logging
from typing import *
import cytoolz as toolz
import numpy as np
from mincall.common import *


class Assignment(NamedTuple):
    params: Dict
    name: str
    context: Any = None


class Observation(NamedTuple):
    metric: float


class AbstractSolver:
    def __init__(self, params: Dict):
        """

        :param params: tree of dictionaries where keys are strings and values are either Param or dictionaries conforting to the same structure
        """
        self.params = params

    def report(self, assignment: Assignment, observation: Observation):
        raise NotImplemented

    def new_assignment(self) -> Assignment:
        raise NotImplemented


class RandomSolver(AbstractSolver):
    def _random_assigement(self, x):
        if isinstance(x, Param):
            if x.type == "int":
                return int(np.random.randint(x.min, x.max))
            if x.type == "float":
                return float(np.random.ranf(x.min, x.max))
            raise ValueError(f"Unwknown type {x.type}")
        if isinstance(x, dict):
            return toolz.valmap(self._random_assigement, x)

    def new_assignment(self):
        return Assignment(
            params=self._random_assigement(self.params),
            name=name_generator(),
        )

    def report(self, assignment: Assignment, observation: Observation):
        logging.info(f"Assigement {assignment} has observation {observation}")
