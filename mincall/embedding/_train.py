import tensorflow as tf
import scrappy
import numpy as np
import sys
import h5py
import argparse
import logging
from typing import *
import toolz
import yaml
import voluptuous
from voluptuous.humanize import humanize_error
from pprint import pformat

logger = logging.getLogger(__name__)


class EmbeddingCfg(NamedTuple):
    files: str

    @classmethod
    def schema(cls, data):
        return cls(**voluptuous.Schema({'files': str}, required=True)(data))


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--config", "-c", help="config file")
    parser.add_argument("--files", "-f", dest="embed.files", help="target files")
    parser.set_defaults(func=run_args)
    parser.set_defaults(name="mincall_embedding")

def run_args(args):
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.load(f)
    else:
        config = {
            'version': "v0.1"
        }

    for k, v in vars(args).items():
        if v is not None and "." in k:
            config = toolz.assoc_in(config, k.split("."), v)
    if args.logdir is not None:
        config['embed']['logdir'] = args.logdir
    try:
        cfg = voluptuous.Schema(
            {
                'embed': EmbeddingCfg.schema,
                'version': str,
            },
            extra=voluptuous.REMOVE_EXTRA,
            required=True)(config)
        logger.info(f"Parsed config\n{pformat(cfg)}")
        run(cfg['embed'])
    except voluptuous.error.Error as e:
        logger.error(humanize_error(config, e))
        sys.exit(1)


def run(cfg: EmbeddingCfg):
    with h5py.File(cfg.files, 'r') as f:
        raw_dat = list(f['/Raw/Reads/'].values())[0]
        raw_dat = np.array(raw_dat['Signal'].value)
        print(dir(scrappy))
        print(help(scrappy.trim_raw))
