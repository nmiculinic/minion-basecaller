from typing import *
import os
import logging
import cytoolz as toolz
import yaml
import argparse
import voluptuous
from . import _train
from mincall.common import *
from ._train import DataDir, TrainConfig


class HyperParamCfg(NamedTuple):
    train_data: List[DataDir]
    test_data: List[DataDir]
    batch_size: int
    seq_length: int
    train_steps: int
    model_name: str
    init_learning_rate: float
    lr_decay_steps: int
    lr_decay_rate: float
    surrogate_base_pair: bool

    grad_clipping: float = 10.0
    run_trace_every: int = 0
    validate_every: int = 50
    save_every: int = 2000

    @classmethod
    def schema(cls, data):
        return named_tuple_helper(cls, {}, data)


def run_args(args: argparse.Namespace):
    with open(args.config) as f:
        config = yaml.load(f)
    for k, v in vars(args).items():
        if v is not None and "." in k:
            config = toolz.assoc_in(config, k.split("."), v)
            print(k, v)
    if args.logdir is not None:
        config['train']['logdir'] = args.logdir
    try:
        cfg = voluptuous.Schema({
            'train': TrainConfig.schema,
            'version': str,
        },
            extra=voluptuous.REMOVE_EXTRA,
            required=True)(config)
        if args.logdir is None:
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)5s]:%(name)20s: %(message)s"
            )
            train_cfg: TrainConfig = cfg['train']
            os.makedirs(train_cfg.logdir, exist_ok=True)
            fn = os.path.join(
                train_cfg.logdir, f"{getattr(args, 'name', 'mincall')}.log"
            )
            h = (logging.FileHandler(fn))
            h.setLevel(logging.DEBUG)
            h.setFormatter(formatter)
            logging.getLogger().addHandler(h)
            logging.info(f"Added handler to {fn}")
        logger.info(f"Parsed config\n{pformat(cfg)}")
        run(cfg['train'])
    except voluptuous.error.Error as e:
        logger.error(humanize_error(config, e))
        sys.exit(1)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--config", "-c", help="config file", required=True)
    parser.set_defaults(func=run_args)
    parser.set_defaults(name="mincall_hyperparam_search")
