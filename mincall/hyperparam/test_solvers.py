import unittest
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
