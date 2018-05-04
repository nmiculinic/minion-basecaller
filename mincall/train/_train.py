import argparse
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

    # input_feeder_cfg: InputFeederCfg = InputFeederCfg(
    #     batch_size=10,
    #     seq_length=10,
    # )
    #
    # m = Manager()
    # q: Queue = m.Queue()
    #
    # p = Process(
    #     target=produce_datapoints, args=(input_feeder_cfg, datapoints, q))
    # p.start()
    # p.join()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    dq = DataQueue(10)
    with tf.train.MonitoredSession(
            session_creator=tf.train.ChiefSessionCreator(
                config=config)) as sess:

    # with tf.Session(config=config) as sess:
        for i in tqdm(itertools.count()):
            try:
                # signal, labels = q.get(timeout=0.5)
                signal = np.arange(4)
                labels = np.arange(2)
                print(f"about to enqueue \n{signal}\n{labels}")
                dq.push_to_queue(sess, signal, labels)
                print(f"enqueue successful\n{signal}\n{labels}")

                if i % 20 == 0 and i >= 20:
                    print(sess.run([
                        dq.batch_labels,
                        dq.batch_labels_len,
                        dq.batch_signal,
                        dq.batch_signal_len,
                    ]))

            except queue.Empty:
                break
            except Exception as e:
                print(type(e).__name__, type(e))
                raise


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
