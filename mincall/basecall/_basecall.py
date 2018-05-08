import argparse
import os
import re
from collections import defaultdict
from scipy import stats
import numpy as np
import voluptuous
import sys
import yaml
from typing import *
import logging
from voluptuous.humanize import humanize_error
from glob import glob
from pprint import pformat
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline
from tensorboard.plugins.beholder import Beholder
from mincall import dataset_pb2
from keras import backend as K
from keras import models
import edlib

import toolz
from tqdm import tqdm

logger = logging.getLogger(__name__)


def decode(x):
    return "".join(map(dataset_pb2.BasePair.Name, x))

class BasecallCfg(NamedTuple):
    input_dir: List[str]
    output_fasta: str
    batch_size: int
    seq_length: int
    beam_width: int
    jump: int

    @classmethod
    def schema(cls, data):
        return cls(**voluptuous.Schema(
            {
                'input_dir': [voluptuous.Any(voluptuous.IsDir(), voluptuous.IsFile())],
                voluptuous.Optional('batch_size', default=1100):
                    int,
                voluptuous.Optional('seq_length', default=300):
                    int,
                voluptuous.Optional('train_steps', default=1000):
                    int,
            },
            required=True)(data))

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--config", "-c", help="config file", required=True)
    parser.add_argument(
        "--trace",
        dest='train.trace',
        help="trace",
        action="store_true",
        default=None)
    parser.add_argument("--batch_size", dest='train.batch_size', type=int)
    parser.add_argument("--seq_length", dest='train.seq_length', type=int)
    parser.set_defaults(func=run_args)
    parser.set_defaults(name="mincall_train")

def run_args(args):
    with open(args.config) as f:
        config = yaml.load(f)
    for k, v in vars(args).items():
        if v is not None and "." in k:
            config = toolz.assoc_in(config, k.split("."), v)
            print(k, v)
    if args.logdir is not None:
        config['train']['logdir'] = args.logdir
    try:
        cfg = voluptuous.Schema(
            {
                'version': str,
            },
            extra=voluptuous.REMOVE_EXTRA,
            required=True)(config)
        logger.info(f"Parsed config\n{pformat(cfg)}")
        run(cfg['train'])
    except voluptuous.error.Error as e:
        logger.error(humanize_error(config, e))
        sys.exit(1)


def run()
