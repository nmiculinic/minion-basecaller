import unittest
from pprint import pformat
import logging
from typing import *
from ._solvers import *

params = [Param(min=0, max=i, type="int") for i in range(10)]


class TestFlatten(unittest.TestCase):
    def test_flatten(self):
        self.assertEqual(params[0], flatten_params(params[0]))

    def test_flatten_depth1(self):
        self.assertEqual({"a": params[0]}, flatten_params({"a": params[0]}))

    def test_flatten_depth2(self):
        self.assertEqual({
            "a.b": params[0]
        }, flatten_params(toolz.assoc_in({}, ["a", "b"], params[0])))

    def test_flatten_depth3(self):
        self.assertEqual({
            "a.b.c": params[0]
        }, flatten_params(toolz.assoc_in({}, ["a", "b", "c"], params[0])))

    def test_unflatten(self):
        self.assertEqual({
            "a": {
                "b": 2,
            },
            "c": 3,
        }, unflatten_params({
            "a.b": 2,
            "c": 3
        }))


@unittest.skipIf(
    os.getenv("SIGOPT_API_TOKEN") is None, "Sigopt api token not setup!"
)
class TestSigOpt(unittest.TestCase):
    def test_doesnt_crash(self):
        opt = SigOpt({"a": {"b": Param(min=0, max=5, type="double")}})
        try:
            for _ in range(3):
                assignment = opt.new_assignment()
                logging.info(f"Assignment: {pformat(assignment)}")
                opt.report(assignment, Observation(metric=1.0))
        finally:
            opt.conn.experiments(id=opt.experiment_id).delete()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
