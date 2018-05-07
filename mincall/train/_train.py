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
from keras import models, layers
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
    train_steps: int
    trace: bool

    @classmethod
    def schema(cls, data):
        return cls(**voluptuous.Schema(
            {
                'train_data': [DataDir.schema],
                'test_data': [DataDir.schema],
                'batch_size': int,
                'seq_length': int,
                voluptuous.Optional('train_steps', default=1000): int,
                voluptuous.Optional('trace', default=False): bool,
            },
            required=True)(data))


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--config", "-c", help="config file", required=True)
    parser.add_argument("--trace", dest='train.trace', help="trace", action="store_true", default=None)
    parser.add_argument("--batch_size", dest='train.batch_size', type=int)
    parser.add_argument("--seq_length", dest='train.seq_length', type=int)
    parser.add_argument("--train_steps", dest='train.train_steps', type=int)
    parser.set_defaults(func=run_args)


class Model():
    def __init__(self, cfg: InputFeederCfg, model: models.Model, data_dir: List[DataDir], trace=False):
        self.dataset = []
        for x in data_dir:
            dps = list(glob(f"{x.dir}/*.datapoint"))
            self.dataset.extend(dps)
            logger.info(
                f"Added {len(dps)} datapoint from {x.name} to train set; dir: {x.dir}"
            )

        self.logger = logging.getLogger(__name__)
        learning_phase = K.learning_phase()
        self.dq = DataQueue(
            cfg, self.dataset,
            trace=trace)

        input_signal: tf.Tensor = self.dq.batch_signal
        labels: tf.SparseTensor = self.dq.batch_labels
        signal_len: tf.Tensor = self.dq.batch_signal_len

        self.logits = tf.transpose(
            model(input_signal), [1, 0, 2])  # [max_time, batch_size, class_num]
        self.logger.info(f"Logits shape: {self.logits.shape}")

        ratio = 1
        seq_len = tf.cast(
            tf.floor_div(signal_len + ratio - 1, ratio), tf.int32)  # Round up

        if trace:
            self.logits = tf.Print(
                self.logits, [
                    self.logits,
                    tf.shape(self.logits),
                    tf.shape(input_signal), labels.indices, labels.values,
                    labels.dense_shape
                ],
                message="varios debug out")
            seq_len = tf.Print(
                seq_len, [tf.shape(seq_len), seq_len], message="seq len")

        self.losses = tf.nn.ctc_loss(
            labels=labels,
            inputs=self.logits,
            sequence_length=seq_len,
            preprocess_collapse_repeated=False,
            ctc_merge_repeated=True,
            time_major=True,
        )
        self.ctc_loss = tf.reduce_mean(self.losses)

        tf.add_to_collection('losses', self.ctc_loss)
        tf.summary.scalar('loss', self.ctc_loss)

        self.total_loss = tf.add_n(
            tf.get_collection('losses'), name='total_loss')
        self.train_step = tf.train.AdamOptimizer().minimize(self.total_loss)

        self.predict = tf.nn.ctc_beam_search_decoder(
            inputs=self.logits,
            sequence_length=seq_len,
            merge_repeated=False,
            top_paths=1,
            beam_width=50)[0][0]

    def input_wrapper(self, sess:tf.Session):
        return self.dq.start_input_processes(sess)

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

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    input = layers.Input(shape=(None, 1))
    net = input
    for _ in range(5):
        net = layers.BatchNormalization()(net)
        net = layers.Conv1D(10, 3, padding="same")(net)
        net = layers.Activation('relu')(net)

    net = layers.Conv1D(
        5,
        3,
        padding="same")(net)
    model = Model(
        InputFeederCfg(batch_size=cfg.batch_size, seq_length=cfg.seq_length),
        models.Model(inputs=[input], outputs=[net]),
        cfg.train_data,
        trace=cfg.trace,
    )

    with tf.train.MonitoredSession(
            session_creator=tf.train.ChiefSessionCreator(
                config=config)) as sess:
        K.set_session(sess)
        with tqdm(total=cfg.train_steps) as pbar, model.input_wrapper(sess):
            for i in range(cfg.train_steps):
                _, lbs, lbs_len, logits, predict, lb, loss, losses = sess.run([
                    model.train_step, model.dq.batch_dense_labels,
                    model.dq.batch_labels_len, model.logits, model.predict,
                    model.dq.batch_labels, model.total_loss, model.losses
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


