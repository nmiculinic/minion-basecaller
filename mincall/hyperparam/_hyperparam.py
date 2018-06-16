from typing import *
from itertools import count
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
from ._solvers import AbstractSolver, RandomSolver
from ._types import Param
import sys


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

    @classmethod
    def schema(cls, data):
        return named_tuple_helper(
            cls, {
                'train_data': [DataDir.schema],
                'test_data': [DataDir.schema],
                'model_hparams': {
                    str: Param.scheme
                },
            }, data
        )


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
    parser.add_argument(
        "--work_dir",
        "-w",
        dest="hyperparam.work_dir",
        help="working directory"
    )
    parser.set_defaults(func=run_args)
    parser.set_defaults(name="mincall_hyperparam_search")


def make_dict(x, subs: Dict) -> Tuple[Dict, Dict]:
    if x is None:
        return {}, {}
    if isinstance(x, (int, str, float, bool)):  # scalar
        return x, {}
    if isinstance(x, Param):
        return x, x
    if isinstance(x, dict):
        sol = {}
        params = {}
        for k, v in x.items():
            if k in subs:
                d, p = subs[k], {}
            else:
                d, p = make_dict(v, subs.get(k, {}))
            sol[k] = d
            if len(p):
                params[k] = p
        return sol, params
    if isinstance(x, list):
        sol = []
        for d, p in map(lambda k: make_dict(k, subs), x):
            if len(p) > 0:
                raise ValueError(
                    f"Cannot have params in list!{x}\nparams: {p}\ndata:{d}"
                )
            sol.append(d)
        return sol, {}
    if hasattr(x, '_asdict'):
        return make_dict(dict(x._asdict()), subs)
    raise ValueError(f"Unknown type {type(x).__name__}: {x}")


def subs_dict(x, subs: Dict) -> Dict:
    sol, _ = make_dict(x, subs)
    return sol


def run(cfg: HyperParamCfg):
    train_cfg, params = make_dict(cfg, {})
    train_cfg = toolz.keyfilter(
        lambda x: x in TrainConfig.__annotations__.keys(), train_cfg
    )
    solver = RandomSolver(params)

    for i in count():
        assigement = solver.new_assignment()
        concrete_params = assigement.params
        folder = os.path.normpath(
            os.path.abspath(
                os.path.join(cfg.work_dir, "%02d-%s" % (i, assigement.name))
            )
        )
        cfg_path = os.path.join(folder, "config.yml")
        os.makedirs(folder, exist_ok=False)

        concrete_cfg = subs_dict(train_cfg, concrete_params)
        print(pformat(concrete_cfg))
        concrete_cfg['logdir'] = folder
        concrete_cfg = subs_dict(TrainConfig.schema(concrete_cfg), {})
        print(pformat(concrete_cfg))
        with open(cfg_path, "w") as f:
            yaml.safe_dump({
                'train': concrete_cfg,
                'version': "v0.1",
            },
                           stream=f,
                           default_flow_style=False)
        _train.run_args(argparse.Namespace(
            config=cfg_path,
            logdir=None,
        ))
        if i > 20:
            break
