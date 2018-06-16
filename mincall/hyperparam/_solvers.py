from ._types import Param, Observation
from pprint import pformat
import logging
from typing import *
import cytoolz as toolz
import numpy as np
from mincall.common import *



class Assignment(NamedTuple):
    params: Dict
    name: str
    context: Any = None


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
    def __init__(self, params):
        self.logger = logging.getLogger(".".join(__name__.split(".")[:-1] + ["RandomSolver"]))
        super().__init__(params)
        self.cnt = -1

    def _random_assigement(self, x):
        if isinstance(x, Param):
            if x.type == "int":
                return int(np.random.randint(x.min, x.max))
            if x.type == "float":
                return float(np.random.ranf(x.min, x.max))
            raise ValueError(f"Unknown type {x.type}")
        if isinstance(x, dict):
            return toolz.valmap(self._random_assigement, x)

    def new_assignment(self):
        self.cnt += 1
        return Assignment(
            params=self._random_assigement(self.params),
            name=f"{self.cnt:02}-{name_generator()}",
        )

    def report(self, assignment: Assignment, observation: Observation):
        self.logger.info(f"Assigement:\n{pformat(dict(assignment._asdict()))}\n"
                         f"has observation:\n{pformat(dict(observation._asdict()))}"
                         )
