import argparse
import voluptuous
import yaml
from typing import *
from pprint import pformat, pprint

class DataDir(NamedTuple):
    name: str
    dir: str

    @classmethod
    def schema(cls):
        return cls(**voluptuous.Schema({
            'name': str,
            'dir': voluptuous.validators.IsDir(),
        }))

class TrainConfig(NamedTuple):
    data: List[DataDir]

    @classmethod
    def schema(cls):
        return cls(**voluptuous.Schema({
            'data': [DataDir.schema]
        }))

def run(cfg: TrainConfig):
    pass

def run_args(args):
    with open(args.config) as f:
        config = yaml.parse()

    run(TrainConfig(
        config="",
    ))

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--config", "-c", help="config file", required=True)
    pass