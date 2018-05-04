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

    input_feeder_cfg: InputFeederCfg = InputFeederCfg(
        batch_size=10,
        seq_length=10,
    )

    m = Manager()
    q: Queue = m.Queue()

    p = Process(
        target=produce_datapoints, args=(input_feeder_cfg, datapoints, q))
    p.start()
    p.join()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    values = tf.placeholder(dtype=tf.int32, shape=[None], name="labels")
    cntValues = tf.placeholder(dtype=tf.int64, shape=[], name="count_labels")

    queue = tf.PaddingFIFOQueue(
        10,
        dtypes=[tf.int32, tf.int64],
        shapes=[
            [None],
            [],
        ])

    enq = queue.enqueue([values, cntValues])
    NN = 3
    ind_op, cnt_op = queue.dequeue_many(NN)
    gg = tf.split(ind_op, NN)
    cnt = tf.split(cnt_op, NN)
    sp = []
    for g, c in zip(gg, cnt):
        c = tf.squeeze(c, axis=0)
        print(c.shape)
        print(c.dtype)

        ind = tf.transpose(
                    tf.stack(
                        [
                            tf.zeros(shape=c, dtype=tf.int64),
                            tf.range(c, dtype=tf.int64),
                        ]
                ))

        print(ind, ind.shape, ind.dtype)
        sp.append(
            tf.SparseTensor(
                indices=ind,
                values=tf.squeeze(g, axis=0)[:c],
                dense_shape=tf.stack([1,c], 0)
            )
        )
    ggg = tf.sparse_concat(axis=0, sp_inputs=sp, expand_nonconcat_dim=True)

    # with tf.train.MonitoredSession(
    #         session_creator=tf.train.ChiefSessionCreator(
    #             config=config)) as sess:
    with tf.Session() as sess:

        # ggg = tf.constant(5, shape=(), dtype=tf.int64)
        # print(sess.run([
        #     tf.transpose(
        #         tf.stack(
        #         [
        #             tf.zeros(shape=ggg, dtype=tf.int64),
        #             tf.range(ggg, dtype=tf.int64),
        #         ]
        #     ))
        # ]))

        sess.run(enq, feed_dict={
            values:[1,2,3],
            cntValues: 3,
        })
        sess.run(enq, feed_dict={
            values:[1,2],
            cntValues: 2,
        })
        sess.run(enq, feed_dict={
            values:[1,2, 3, 4, 5],
            cntValues: 5,
        })

        al, xxx = sess.run([ggg, sp])
        print(al)
        for x in xxx:
            print(x)

        while True:
            try:
                signal, labels = q.get(timeout=0.5)
                break
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
