import argparse
import re
from collections import defaultdict
from scipy import stats
import sys
import itertools
import numpy as np
import voluptuous
import sys
import yaml
from typing import *
import logging
from voluptuous.humanize import humanize_error
from glob import glob
from pprint import pformat, pprint
from ._input_feeders import InputFeederCfg, produce_datapoints, DataQueue
from multiprocessing import Manager, Process, Queue
import tensorflow as tf
import queue
from mincall import dataset_pb2
from keras import backend as K
from keras.layers import Dense, Conv1D
from .models import Model
import edlib

import toolz
from tqdm import tqdm

logger = logging.getLogger(__name__)


def decode(x):
    return "".join(map(dataset_pb2.BasePair.Name, x))


def squggle(query: str, target: str) -> Tuple[str, str, Dict]:
    if len(query) > 0:
        alignment = edlib.align(query, target, task='path')
    else:
        alignment = {
            'editDistance': len(target),
            'cigar': f"{len(target)}D",
            'alphabetLength': 4,
            'locations': [(0, len(target))],
        }

    cigar = alignment['cigar']
    q_idx = 0
    t_idx = 0

    qq, tt = "", ""
    for x in re.findall(r"\d+[=XIDSHM]", cigar):
        cnt = int(x[:-1])
        op = x[-1]
        if op in ["=", "X"]:
            qq += query[q_idx:q_idx + cnt]
            q_idx += cnt

            tt += target[t_idx:t_idx + cnt]
            t_idx += cnt
        elif op == "D":
            qq += "-" * cnt

            tt += target[t_idx:t_idx + cnt]
            t_idx += cnt
        elif op == "I":
            qq += query[q_idx:q_idx + cnt]
            q_idx += cnt

            tt += "-" * cnt
        else:
            ValueError(f"Unknown op {op}")
    assert len(target) == t_idx, "Not all target base pairs used"
    assert len(query) == q_idx, "Not all target base pairs used"
    return qq, tt, alignment


class DataDir(NamedTuple):
    name: str
    dir: str

    @classmethod
    def schema(cls, data):
        return cls(**voluptuous.Schema(
            {
                'name': str,
                'dir': voluptuous.validators.IsDir(),
            },
            required=True)(data))


class TrainConfig(NamedTuple):
    train_data: List[DataDir]
    test_data: List[DataDir]
    batch_size: int
    seq_length: int
    trace: bool = False

    @classmethod
    def schema(cls, data):
        return cls(**voluptuous.Schema(
            {
                'train_data': [DataDir.schema],
                'test_data': [DataDir.schema],
                'batch_size': int,
                'seq_length': int,
                'trace': bool,
            },
            required=True)(data))


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--config", "-c", help="config file", required=True)
    parser.add_argument("--trace", dest='train.trace', help="trace", action="store_true")
    parser.add_argument("--batch_size", dest='train.batch_size', type=int)
    parser.add_argument("--seq_length", dest='train.seq_length', type=int)
    parser.set_defaults(func=run_args)


def run_args(args):
    with open(args.config) as f:
        config = yaml.load(f)
    for k, v in vars(args).items():
        if v is not None and "." in k:
            config = toolz.assoc_in(config, k.split("."), v)
    try:
        cfg = voluptuous.Schema(
            {
                'train': TrainConfig.schema,
                'version': str,
            },
            extra=voluptuous.REMOVE_EXTRA,
            required=True)(config)
        logger.info(f"Parsed config\n{pformat(cfg)}")
        run(cfg['train'])
    except voluptuous.error.Error as e:
        logger.error(humanize_error(config, e))
        sys.exit(1)


def run(cfg: TrainConfig):
    datapoints = []
    for x in cfg.train_data:
        dps = list(glob(f"{x.dir}/*.datapoint"))
        datapoints.extend(dps)
        logger.info(
            f"Added {len(dps)} datapoint from {x.name} to train set; dir: {x.dir}"
        )

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    dq = DataQueue(
        InputFeederCfg(batch_size=cfg.batch_size, seq_length=cfg.seq_length),
        trace=cfg.trace)
    model = Model(
        cfg,
        dq.batch_labels,
        dq.batch_signal,
        dq.batch_signal_len,
        trace=cfg.trace)

    with tf.train.MonitoredSession(
            session_creator=tf.train.ChiefSessionCreator(
                config=config)) as sess:
        K.set_session(sess)
        close = dq.start_input_processes(sess, datapoints)

        with tqdm() as pbar:
            for i in itertools.count():
                _, lbs, lbs_len, logits, predict, lb, loss, losses = sess.run([
                    model.train_step, dq.batch_dense_labels,
                    dq.batch_labels_len, model.logits, model.predict,
                    dq.batch_labels, model.total_loss, model.losses
                ])
                pbar.set_postfix(loss=loss, refresh=False)
                pbar.update()
                if i % 100 == 0:
                    logger.info(
                        f"Logits[{logits.shape}]:\n describe:{pformat(stats.describe(logits, axis=None))}"
                    )

                    yt = defaultdict(list)
                    yp = defaultdict(list)
                    for ind, val in zip(lb.indices, lb.values):
                        yt[ind[0]].append(val)

                    for ind, val in zip(predict.indices, predict.values):
                        yp[ind[0]].append(val)

                    for x in range(cfg.batch_size):
                        q, t, alignment = squggle(
                            decode(yp[x]),
                            decode(yt[x]),
                        )
                        logger.info(
                            f"{x}: \n"
                            # f"Target    : {yt[x]}\n"
                            # f"Basecalled: {yp[x]}\n"
                            f"Basecalled: {q}\n"
                            f"Target    : {t}\n"
                            f"Loss      : {losses[x]}\n"
                            f"Edit dist : {alignment['editDistance'] * 'x'}\n")
        close()


