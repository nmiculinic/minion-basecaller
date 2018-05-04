import argparse
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
    dq = DataQueue(10)
    learning_phase = K.learning_phase()

    net = dq.batch_signal
    net = Conv1D(5, 3, input_shape=(None, None, 1))(net)

    logits = net # Tensor of shape [batch_size, max_time, class_num]
    logger.info(f"Logits shape: {logits.shape}")

    ratio = 1
    loss = tf.reduce_mean(tf.nn.ctc_loss(
        dq.batch_labels,
        logits,
        tf.cast(tf.floor_div(dq.batch_signal_len + ratio - 1, ratio), tf.int32),  # Round up
        ctc_merge_repeated=True,
        time_major=False,
    ))

    tf.add_to_collection('losses',loss)
    tf.summary.scalar('loss', loss)
    total_loss = tf.add_n(tf.get_collection('losses'),name = 'total_loss')

    train_step = tf.train.AdadeltaOptimizer().minimize(total_loss)

    with tf.train.MonitoredSession(
            session_creator=tf.train.ChiefSessionCreator(
                config=config)) as sess:
        K.set_session(sess)
        close = dq.start_input_processes(sess, datapoints)

        for i in tqdm(itertools.count()):
            sess.run(train_step)
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
