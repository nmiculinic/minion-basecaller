from ._types import Param, Observation
import socket
from sigopt import Connection
import os
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


def flatten_params(params: Union[Dict, Param], cls=Param) -> Dict[str, Param]:
    """Flattens params into "." separated values

    e.g.
    {"a":{"b": Param(...)}} -> {"a.b": Param}

    :param params:
    :return:
    """
    if isinstance(params, cls):
        return params
    sol = {}
    for k, v in toolz.valmap(flatten_params, params).items():
        if isinstance(v, cls):
            sol[k] = v
        else:
            sol = toolz.merge(sol, toolz.keymap(lambda x: f"{k}.{x}", v))
    return sol


def unflatten_params(params: Dict[str, Any]):
    sol = {}
    for k, v in params.items():
        sol = toolz.assoc_in(sol, k.split("."), v)
    return sol


class RandomSolver(AbstractSolver):
    def __init__(self, params):
        self.logger = logging.getLogger(
            ".".join(__name__.split(".")[:-1] + ["RandomSolver"])
        )
        super().__init__(params)
        self.cnt = -1

    def _random_assigement(self, x):
        if isinstance(x, Param):
            if x.type == "int":
                return int(np.random.randint(x.min, x.max))
            if x.type == "double":
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
        self.logger.info(
            f"Assignment:\n{pformat(dict(assignment._asdict()))}\n"
            f"has observation:\n{pformat(dict(observation._asdict()))}"
        )


class SigOpt(AbstractSolver):
    def __init__(self, params):
        super().__init__(params)
        self.experiment_id = os.getenv("SIGOPT_EXPERIMENT", None)
        api_token = os.getenv("SIGOPT_API_TOKEN")
        if api_token is None:
            raise ValueError(
                "envvar SIGOPT_API_TOKEN must be set with SigOptSolver"
            )
        self.conn = Connection(client_token=api_token)
        self.logger = logging.getLogger(
            ".".join(__name__.split(".")[:-1] + ["SigOptSolver"])
        )
        if self.experiment_id is None:
            params = flatten_params(params)
            experiment = self.conn.experiments().create(
                name=f'Mincall opt {socket.gethostname()}',
                parameters=[{
                    "name": name,
                    "type": p.type,
                    "bounds": {
                        "min": p.min,
                        "max": p.max
                    }
                } for name, p in params.items()],
            )
            self.experiment_id = experiment.id
            self.logger.info(
                "Created experiment: https://sigopt.com/experiment/" +
                self.experiment_id
            )
        else:
            self.logger.info(
                "Using experiment: https://sigopt.com/experiment/" +
                self.experiment_id
            )

        self.conn = Connection(client_token=api_token)

    def new_assignment(self):
        suggestion = self.conn.experiments(self.experiment_id
                                          ).suggestions().create()
        return Assignment(
            params=unflatten_params(suggestion.assignments),
            name=f"{suggestion.id}-{name_generator()}",
            context=suggestion.id,
        )

    def report(self, assignment: Assignment, observation: Observation):
        self.logger.info(
            f"Assignment:\n{pformat(dict(assignment._asdict()))}\n"
            f"has observation:\n{pformat(dict(observation._asdict()))}"
        )
        self.conn.experiments(self.experiment_id).observations().create(
            suggestion=assignment.context,
            value=observation.metric,
            metadata=observation.metadata,
        )


available_solvers = {
    "sigopt": SigOpt,
    "random": RandomSolver,
}
