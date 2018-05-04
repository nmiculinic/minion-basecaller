import argparse
import numpy as np
import voluptuous
import sys
import yaml
from typing import *
import logging
from voluptuous.humanize import humanize_error
from glob import glob
from pprint import pformat, pprint
from ._input_feeders import InputFeederCfg, produce_datapoints
from multiprocessing import Manager, Process, Queue
import tensorflow as tf
import queue
from mincall import dataset_pb2

import toolz
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataDir(NamedTuple):
    name: str
    dir: str

    @classmethod
    def schema(cls, data):
        return cls(**voluptuous.Schema({
            'name': str,
            'dir': voluptuous.validators.IsDir(),
        }, required=True)(data))


class TrainConfig(NamedTuple):
    data: List[DataDir]
    batch_size: int
    seq_length: int

    @classmethod
    def schema(cls, data):
        return cls(**voluptuous.Schema({
            'data': [DataDir.schema],
            'batch_size': int,
            'seq_length': int,
        }, required=True)(data))


def run(cfg: TrainConfig):
    datapoints = []
    for x in cfg.data:
        dps = list(glob(f"{x.dir}/*.datapoint"))
        datapoints.extend(dps)
        logger.info(
            f"Added {len(dps)} datapoint from {x.name} to train set; dir: {x.dir}"
        )

    input_feeder_cfg: InputFeederCfg = InputFeederCfg(
        batch_size=10,
        seq_length=10,
    )

    m = Manager()
    q: Queue = m.Queue()

    p = Process(target=produce_datapoints, args=(input_feeder_cfg, datapoints, q))
    p.start()
    p.join()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    indices = tf.placeholder(dtype=tf.int64)
    values = tf.placeholder(dtype=tf.int32)
    dense_shape = tf.placeholder(dtype=tf.int64)
    st = tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=dense_shape,
    )
    c = tf.sparse_concat(0, [st,st], expand_nonconcat_dim=True)
    with tf.train.MonitoredSession(session_creator=tf.train.ChiefSessionCreator(config=config)) as sess:
        while True:
            try:
                signal, labels = q.get(timeout=0.5)
                v = tf.SparseTensorValue(
                    indices=np.array(np.vstack([np.zeros(len(labels)), np.arange(len(labels))]).transpose(), dtype=np.int32),
                    values=labels,
                    dense_shape=[1, len(labels)],
                )
                x = sess.run(c, feed_dict={
                    indices: v.indices,
                    values:  v.values,
                    dense_shape: v.dense_shape,
                })

                print(v)
                print(x)
                if len(labels) > 0:
                    break
            except queue.Empty:
                break
            except Exception as e:
                print(type(e).__name__, type(e))


def run_args(args):
    with open(args.config) as f:
        config = yaml.load(f)
    try:
        cfg = voluptuous.Schema(
            {
                'train': TrainConfig.schema,
                'version': str,
            },
            extra=voluptuous.REMOVE_EXTRA, required=True)(config)
        logger.info(f"Parsed config\n{pformat(cfg)}")
        run(cfg['train'])
    except voluptuous.error.Error as e:
        logger.error(humanize_error(config, e))
        sys.exit(1)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--config", "-c", help="config file", required=True)
    parser.set_defaults(func=run_args)
