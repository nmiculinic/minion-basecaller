import argparse
from collections import defaultdict
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

import toolz
from tqdm import tqdm

logger = logging.getLogger(__name__)


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
    data: List[DataDir]
    batch_size: int
    seq_length: int

    @classmethod
    def schema(cls, data):
        return cls(**voluptuous.Schema(
            {
                'data': [DataDir.schema],
                'batch_size': int,
                'seq_length': int,
            },
            required=True)(data))


def run(cfg: TrainConfig):
    datapoints = []
    for x in cfg.data:
        dps = list(glob(f"{x.dir}/*.datapoint"))
        datapoints.extend(dps)
        logger.info(
            f"Added {len(dps)} datapoint from {x.name} to train set; dir: {x.dir}"
        )


    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    dq = DataQueue(InputFeederCfg(batch_size=cfg.batch_size, seq_length=cfg.seq_length))

    model = Model(cfg, dq.batch_labels, dq.batch_signal, dq.batch_signal_len)
    with tf.train.MonitoredSession(
            session_creator=tf.train.ChiefSessionCreator(
                config=config)) as sess:
        K.set_session(sess)
        close = dq.start_input_processes(sess, datapoints)

        for i in tqdm(itertools.count()):
            _, lbs, lbs_len, logits, predict, lb_ind, lb_val = sess.run([model.train_step,dq.batch_dense_labels, dq.batch_labels_len, model.logits, model.predict, dq.batch_labels.indices, dq.batch_labels.values])
            logger.info(f"Labels:\n{lbs}\nlabels len\n{lbs_len}\nLogits[{logits.shape}]:\n{logits}\nPredict:\n{predict}")

            yt = defaultdict(list)
            yp = defaultdict(list)
            for ind, val in zip(lb_ind, lb_val):
                yt[ind[0]].append(val)
            for ind, val in zip(predict.indices, predict.values):
                yp[ind[0]].append(val)

            for x in range(cfg.batch_size):
                logger.info(f"{x}: \nTarget    : {yt[x]}\nBasecalled:\n {yp[x]}")
        close()


def run_args(args):
    with open(args.config) as f:
        config = yaml.load(f)
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


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--config", "-c", help="config file", required=True)
    parser.set_defaults(func=run_args)
