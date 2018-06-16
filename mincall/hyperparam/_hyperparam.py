from typing import *
import os
from pprint import pformat
import logging
import cytoolz as toolz
import numpy as np
import yaml
import argparse
import voluptuous
from mincall.train import _train
from mincall.common import *
from mincall.train._train import DataDir, TrainConfig
from voluptuous.humanize import humanize_error
import sys


class Param(NamedTuple):
    min: float
    max: float
    type: str

    @classmethod
    def scheme(cls, data):
        if isinstance(data, (int, float, np.int, np.float, bool)):
            return data
        return named_tuple_helper(cls, {"type": voluptuous.validators.In([
            "int", "double",
        ])}, data)


class HyperParamCfg(NamedTuple):
    model_name: str
    train_data: List[DataDir]
    test_data: List[DataDir]
    seq_length: Param
    batch_size: Param
    surrogate_base_pair: Param

    train_steps: Param
    init_learning_rate: Param
    lr_decay_steps: Param
    lr_decay_rate: Param

    model_hparams: Dict[str, Param]

    work_dir: str = "."
    grad_clipping: float = 10.0
    validate_every: int = 50
    run_trace_every: int = 5000
    save_every: int = 10000

    tensorboard_debug: str = None
    debug: bool = False
    trace: bool = False

    @classmethod
    def schema(cls, data):
        return named_tuple_helper(cls, {
            'train_data': [DataDir.schema],
            'test_data': [DataDir.schema],
            voluptuous.Optional('tensorboard_debug', default=None):
                voluptuous.Any(str, None),
            'model_hparams': {str: Param.scheme},
        }, data)


def run_args(args: argparse.Namespace):
    with open(args.config) as f:
        config = yaml.load(f)
    for k, v in vars(args).items():
        if v is not None and "." in k:
            config = toolz.assoc_in(config, k.split("."), v)
            print(k, v)
    logger = logging.getLogger(__name__)
    try:
        cfg = voluptuous.Schema({
            'hyperparam': HyperParamCfg.schema,
            'version': str,
        },
            extra=voluptuous.REMOVE_EXTRA,
            required=True)(config)
    except voluptuous.error.Error as e:
        logger.error(humanize_error(config, e))
        sys.exit(1)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)5s]:%(name)20s: %(message)s"
    )
    cfg: HyperParamCfg = cfg['hyperparam']
    print("##", cfg, type(cfg))
    os.makedirs(cfg.work_dir, exist_ok=True)
    fn = os.path.join(
        cfg.work_dir, f"{getattr(args, 'name', 'mincall_hyper')}.log"
    )
    h = (logging.FileHandler(fn))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logging.getLogger().addHandler(h)
    logging.info(f"Added handler to {fn}")
    logger.info(f"Parsed config\n{pformat(cfg)}")
    run(cfg)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--config", "-c", help="config file", required=True)
    parser.add_argument("--work_dir", "-w", dest="hyperparam.work_dir", help="working directory")
    parser.set_defaults(func=run_args)
    parser.set_defaults(name="mincall_hyperparam_search")

def run(cfg: HyperParamCfg):
    print(pformat(cfg._asdict()))
    print("---")
    print(yaml.dump(dict(cfg._asdict())))
    pass
