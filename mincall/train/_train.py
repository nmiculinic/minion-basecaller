import argparse
import os
import re
from collections import defaultdict
from scipy import stats
import numpy as np
import voluptuous
import sys
import yaml
from typing import *
import logging
from voluptuous.humanize import humanize_error
from glob import glob
from pprint import pformat
from ._input_feeders import InputFeederCfg, DataQueue
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline
from tensorboard.plugins.beholder import Beholder
from minion_data import dataset_pb2
from keras import backend as K
from keras import models
from .models import dummy_model
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
    logdir: str
    validate_every: int
    save_every: int
    debug: bool
    tensorboard_debug: str
    run_trace_every: int

    @classmethod
    def schema(cls, data):
        return cls(**voluptuous.Schema(
            {
                'train_data': [DataDir.schema],
                'test_data': [DataDir.schema],
                'batch_size':
                int,
                'seq_length':
                int,
                voluptuous.Optional('train_steps', default=1000):
                int,
                voluptuous.Optional('trace', default=False):
                bool,
                'logdir':
                str,
                voluptuous.Optional('save_every', default=2000):
                int,
                voluptuous.Optional('run_trace_every', default=5000):
                int,
                voluptuous.Optional('validate_every', default=50):
                int,
                voluptuous.Optional('debug', default=False):
                bool,
                voluptuous.Optional('tensorboard_debug', default=None):
                voluptuous.Any(str, None),
            },
            required=True)(data))


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--config", "-c", help="config file", required=True)
    parser.add_argument(
        "--trace",
        dest='train.trace',
        help="trace",
        action="store_true",
        default=None)
    parser.add_argument("--batch_size", dest='train.batch_size', type=int)
    parser.add_argument("--seq_length", dest='train.seq_length', type=int)
    parser.add_argument("--train_steps", dest='train.train_steps', type=int)
    parser.add_argument(
        "--run_trace_every",
        dest='train.run_trace_every',
        type=int,
        help="Full trace session.run() every x steps. Use 0 do disable")
    parser.add_argument(
        "--debug",
        dest='train.debug',
        default=None,
        action="store_true",
        help="activate debug mode")
    parser.add_argument(
        "--tensorboard_debug",
        dest='train.tensorboard_debug',
        help=
        "if debug mode is activate and this is set, use tensorboard debugger")

    parser.add_argument("--logdir", dest='logdir', type=str)
    parser.set_defaults(func=run_args)
    parser.set_defaults(name="mincall_train")


class Model():
    def __init__(self,
                 cfg: InputFeederCfg,
                 model: models.Model,
                 data_dir: List[DataDir],
                 trace=False,
                 create_train_ops=False):
        self.dataset = []
        for x in data_dir:
            dps = list(glob(f"{x.dir}/*.datapoint"))
            self.dataset.extend(dps)
            logger.info(
                f"Added {len(dps)} datapoint from {x.name} to train set; dir: {x.dir}"
            )

        self.logger = logging.getLogger(__name__)
        self.learning_phase = K.learning_phase()
        self.dq = DataQueue(
            cfg, self.dataset, capacity=10 * cfg.batch_size, trace=trace)
        input_signal: tf.Tensor = self.dq.batch_signal
        labels: tf.SparseTensor = self.dq.batch_labels
        signal_len: tf.Tensor = self.dq.batch_signal_len

        self.logits = tf.transpose(
            model(input_signal),
            [1, 0, 2])  # [max_time, batch_size, class_num]
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
                message="various debug out")
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
        self.regularization_loss = tf.add_n(model.losses)
        self.total_loss = self.ctc_loss + self.regularization_loss
        if create_train_ops:
            self.train_step = tf.train.AdamOptimizer().minimize(
                self.total_loss)

        self.summaries = [
            tf.summary.scalar(f'total_loss', self.total_loss, family="losses"),
            tf.summary.scalar(f'ctc_loss', self.ctc_loss, family="losses"),
            tf.summary.scalar(
                f'regularization_loss',
                self.regularization_loss,
                family="losses"),
            *self.dq.summaries,
        ]

        self.predict = tf.nn.ctc_beam_search_decoder(
            inputs=self.logits,
            sequence_length=seq_len,
            merge_repeated=False,
            top_paths=1,
            beam_width=50)[0][0]

    def input_wrapper(self, sess: tf.Session, coord: tf.train.Coordinator):
        return self.dq.start_input_processes(sess, coord)


def run_args(args):
    with open(args.config) as f:
        config = yaml.load(f)
    for k, v in vars(args).items():
        if v is not None and "." in k:
            config = toolz.assoc_in(config, k.split("."), v)
            print(k, v)
    if args.logdir is not None:
        config['train']['logdir'] = args.logdir
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

    os.makedirs(cfg.logdir, exist_ok=True)
    model, _ = dummy_model()

    with tf.name_scope("train"):
        train_model = Model(
            InputFeederCfg(
                batch_size=cfg.batch_size, seq_length=cfg.seq_length),
            model,
            cfg.train_data,
            trace=cfg.trace,
            create_train_ops=True,
        )

    with tf.name_scope("test"):
        test_model = Model(
            InputFeederCfg(
                batch_size=cfg.batch_size, seq_length=cfg.seq_length),
            model,
            cfg.test_data,
            trace=cfg.trace,
        )

    global_step = tf.train.get_or_create_global_step()
    step = tf.assign_add(global_step, 1)
    saver = tf.train.Saver(max_to_keep=10)
    init_op = tf.global_variables_initializer()

    # Basic only train summaries
    train_model.summary = tf.summary.merge(train_model.summaries)

    # Extended validation summaries
    var_summaries = []
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        var: tf.Variable = var
        mean = tf.reduce_mean(var)
        name = var.name.split(":")[0]
        var_summaries.extend([
            tf.summary.scalar(name + '/mean', mean),
            tf.summary.scalar(name + '/stddev',
                              tf.sqrt(tf.reduce_mean(tf.square(var - mean)))),
            tf.summary.scalar(name + '/max', tf.reduce_max(var)),
            tf.summary.scalar(name + '/min', tf.reduce_min(var)),
            tf.summary.histogram(name + '/histogram', var),
        ])

    mean_logits, var_logits = tf.nn.moments(test_model.logits, axes=[])
    test_model.summary = tf.summary.merge(
        test_model.summaries + var_summaries + [
            tf.summary.histogram("logits/histogram", test_model.logits),
            tf.summary.histogram("logits/min", tf.reduce_min(test_model.logits)),
            tf.summary.histogram("logits/max", tf.reduce_max(test_model.logits)),
            tf.summary.histogram("logits/mean", mean_logits),
            tf.summary.histogram("logits/stddev", tf.sqrt(var_logits)),
        ])
    beholder = Beholder(cfg.logdir)

    with tf.Session(config=config) as sess:
        if cfg.debug:
            if cfg.tensorboard_debug is None:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            else:
                sess = tf_debug.TensorBoardDebugWrapperSession(
                    sess, cfg.tensorboard_debug)
        summary_writer = tf.summary.FileWriter(
            os.path.join(cfg.logdir), sess.graph)
        K.set_session(sess)
        last_check = tf.train.latest_checkpoint(cfg.logdir)
        if last_check is None:
            logger.info(f"Running new checkpoint")
            sess.run(init_op)
        else:
            logger.info(f"Restoring checkpoint {last_check}")
            saver.restore(sess=sess, save_path=last_check)

        coord = tf.train.Coordinator()
        try:
            tf.train.start_queue_runners(sess=sess, coord=coord)
            gs = sess.run(global_step)
            with tqdm(
                    total=cfg.train_steps,
                    initial=gs) as pbar, train_model.input_wrapper(
                        sess, coord), test_model.input_wrapper(sess, coord):
                for i in range(gs + 1, cfg.train_steps + 1):
                    #  Train hook
                    opts = {}
                    if cfg.run_trace_every > 0 and i % cfg.run_trace_every == 0:
                        opts['options'] = tf.RunOptions(
                            trace_level=tf.RunOptions.FULL_TRACE)
                        opts['run_metadata'] = tf.RunMetadata()

                    _, _, loss, summary = sess.run([
                        step,
                        train_model.train_step,
                        train_model.ctc_loss,
                        train_model.summary,
                    ], **opts)
                    summary_writer.add_summary(summary, i)

                    if cfg.run_trace_every > 0 and i % cfg.run_trace_every == 0:
                        opts['options'] = tf.RunOptions(
                            trace_level=tf.RunOptions.FULL_TRACE)
                        fetched_timeline = timeline.Timeline(
                            opts['run_metadata'].step_stats)
                        chrome_trace = fetched_timeline.generate_chrome_trace_format(
                            show_memory=True)
                        with open(
                                os.path.join(cfg.logdir,
                                             f'timeline_{i:05}.json'),
                                'w') as f:
                            f.write(chrome_trace)
                        summary_writer.add_run_metadata(
                            opts['run_metadata'],
                            f"step_{i:05}",
                            global_step=i)
                        logger.info(
                            f"Saved trace metadata both to timeline_{i:05}.json and step_{i:05} in tensorboard"
                        )
                    pbar.update()

                    #  Validate hook
                    if i % cfg.validate_every == 0:
                        beholder.update(session=sess)
                        logits, predict, lb, val_loss, losses, test_summary = sess.run(
                            [
                                test_model.logits,
                                test_model.predict,
                                test_model.dq.batch_labels,
                                test_model.ctc_loss,
                                test_model.losses,
                                test_model.summary,
                            ],
                            feed_dict={
                                test_model.learning_phase: 0,
                            })

                        logger.info(
                            f"Logits[{logits.shape}]:\n describe:{pformat(stats.describe(logits, axis=None))}"
                        )
                        summary_writer.add_summary(test_summary, i)
                        pbar.set_postfix(
                            loss=loss, val_loss=val_loss, refresh=False)

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
                                f"Basecalled: {q}\n"
                                f"Target    : {t}\n"
                                f"Loss      : {losses[x]}\n"
                                f"Edit dist : {alignment['editDistance'] * 'x'}\n"
                            )

                    #  Save hook
                    if i % cfg.save_every == 0:
                        saver.save(
                            sess=sess,
                            save_path=os.path.join(cfg.logdir, 'model.ckpt'),
                            global_step=global_step)
                        logger.info(f"Saved new model checkpoint")
                coord.request_stop()
                p = os.path.join(cfg.logdir, f"full-model.save")
                model.save(p, overwrite=True, include_optimizer=False)
                logger.info(f"Finished training saved model to {p}")
            logger.info(f"Input queues exited ok")
        finally:
            coord.request_stop()
            coord.join(stop_grace_period_secs=5)
