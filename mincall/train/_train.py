import argparse
import voluptuous
import yaml
from typing import NamedTuple
from pprint import pformat, pprint

class TrainConfig(NamedTuple):
    config: str
    pass

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