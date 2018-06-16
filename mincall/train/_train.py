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
from keras import backend as K
from keras import models
from .models import all_models
from mincall.common import *
from mincall.train import ops
from minion_data import dataset_pb2

import toolz
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataDir(NamedTuple):
    name: str
    dir: str

    @classmethod
    def schema(cls, data):
        return cls(
            **voluptuous.Schema({
                'name': str,
                'dir': voluptuous.validators.IsDir(),
            },
                                required=True)(data)
        )


class TrainConfig(NamedTuple):
    model_name: str
    train_data: List[DataDir]
    test_data: List[DataDir]
    logdir: str
    seq_length: int
    batch_size: int
    surrogate_base_pair: bool

    train_steps: int
    init_learning_rate: float
    lr_decay_steps: int
    lr_decay_rate: float

    model_hparams: dict = {}
    grad_clipping: float = 10.0
    validate_every: int = 50
    run_trace_every: int = 5000
    save_every: int = 10000

    tensorboard_debug: str = ""  # Empty string is use CLI debug
    debug: bool = False
    trace: bool = False

    @classmethod
    def schema(cls, data):
        return named_tuple_helper(
            cls, {
                'train_data': [DataDir.schema],
                'test_data': [DataDir.schema],
            }, data
        )


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--config", "-c", help="config file", required=True)
    parser.add_argument(
        "--trace",
        dest='train.trace',
        help="trace",
        action="store_true",
        default=None
    )
    parser.add_argument("--batch_size", dest='train.batch_size', type=int)
    parser.add_argument("--seq_length", dest='train.seq_length', type=int)
    parser.add_argument("--train_steps", dest='train.train_steps', type=int)
    parser.add_argument(
        "--run_trace_every",
        dest='train.run_trace_every',
        type=int,
        help="Full trace session.run() every x steps. Use 0 do disable"
    )
    parser.add_argument(
        "--debug",
        dest='train.debug',
        default=None,
        action="store_true",
        help="activate debug mode"
    )
    parser.add_argument(
        "--tensorboard_debug",
        dest='train.tensorboard_debug',
        help="if debug mode is activate and this is set, use tensorboard debugger"
    )

    parser.add_argument("--model", dest='train.model_name', type=str)
    parser.add_argument("--hparams", dest='train.model_hparams', type=str)
    parser.add_argument("--logdir", dest='logdir', type=str)
    parser.add_argument(
        "--grad_clipping",
        dest='train.grad_clipping',
        type=float,
        help="max grad clipping norm"
    )
    parser.add_argument(
        "--surrogate-base-pair",
        dest='train.surrogate_base_pair',
        default=None,
        action="store_true",
        help=
        "Activate surrogate base pairs, that is repeated base pair shall be replaces with surrogate during training phase."
        "for example, let A=0. We have AAAA, which ordinarily will be 0, 0, 0, 0. With surrogate base pairs this will be 0, 4, 0, 4"
    )
    parser.set_defaults(func=run_args)
    parser.set_defaults(name="mincall_train")


class Model():
    def __init__(
        self,
        cfg: InputFeederCfg,
        model: models.Model,
        data_dir: List[DataDir],
        trace=False,
    ):
        self.dataset = []
        for x in data_dir:
            dps = list(glob(f"{x.dir}/*.datapoint"))
            self.dataset.extend(dps)
            logger.info(
                f"Added {len(dps)} datapoint from {x.name} to train set; dir: {x.dir}"
            )

        self._logger = logging.getLogger(__name__)
        self.learning_phase = K.learning_phase()
        with K.name_scope("data_in"):
            self.dq = DataQueue(
                cfg,
                self.dataset,
                capacity=10 * cfg.batch_size,
                trace=trace,
                min_after_deque=2 * cfg.batch_size
            )
        input_signal: tf.Tensor = self.dq.batch_signal
        input_signal = tf.Print(
            input_signal, [tf.shape(input_signal)],
            first_n=1,
            summarize=10,
            message="input signal shape, [batch_size,max_time, 1]"
        )

        labels: tf.SparseTensor = self.dq.batch_labels
        signal_len: tf.Tensor = self.dq.batch_signal_len

        self.labels = labels

        self.logits = tf.transpose(model(input_signal), [1, 0, 2]
                                  )  # [max_time, batch_size, class_num]
        self._logger.info(f"Logits shape: {self.logits.shape}")
        self.logits = tf.Print(
            self.logits, [tf.shape(self.logits)],
            first_n=1,
            summarize=10,
            message="logits shape [max_time, batch_size, class_num]"
        )

        seq_len = tf.cast(
            tf.floor_div(signal_len + cfg.ratio - 1, cfg.ratio), tf.int32
        )  # Round up
        seq_len = tf.Print(
            seq_len, [tf.shape(seq_len), seq_len],
            first_n=5,
            summarize=15,
            message="seq_len [expected around max_time]"
        )

        self.losses = tf.nn.ctc_loss(
            labels=labels,
            inputs=self.logits,
            sequence_length=seq_len,
            preprocess_collapse_repeated=False,
            ctc_merge_repeated=True,
            time_major=True,
        )

        self.predict = tf.nn.ctc_beam_search_decoder(
            inputs=self.logits,
            sequence_length=seq_len,
            merge_repeated=cfg.
            surrogate_base_pair,  # Gotta merge if we have surrogate_base_pairs
            top_paths=1,
            beam_width=100,
        )[0][0]

        finite_mask = tf.logical_not(
            tf.logical_or(
                tf.is_nan(self.losses),
                tf.is_inf(self.losses),
            )
        )

        # self.ctc_loss = tf.reduce_mean(self.losses)
        self.ctc_loss = tf.reduce_mean(
            tf.boolean_mask(
                self.losses,
                finite_mask,
            )
        )
        if model.losses:
            self.regularization_loss = tf.add_n(model.losses)
        else:
            self.regularization_loss = tf.constant(0.0)
        # Nice little hack to get inf/NaNs out of the way. In the beginning of the training
        # logits shall move to some unrealistically large numbers and it shall be hard
        # finding path through the network
        self.regularization_loss += tf.train.exponential_decay(
            learning_rate=tf.nn.l2_loss(self.logits),
            global_step=tf.train.get_or_create_global_step(),
            decay_rate=0.5,
            decay_steps=200,
        )

        self.total_loss = self.ctc_loss + self.regularization_loss

        percent_finite = tf.reduce_mean(tf.cast(finite_mask, tf.int32))
        percent_finite = tf.Print(
            percent_finite, [percent_finite], first_n=10, message="%finite"
        )
        self.summaries = [
            tf.summary.scalar(f'total_loss', self.total_loss),
            tf.summary.scalar(f'ctc_loss', self.ctc_loss),
            tf.summary.scalar(
                f'regularization_loss',
                self.regularization_loss,
                family="losses"
            ),
            tf.summary.scalar("finite_percent", percent_finite),
            *self.dq.summaries,
        ]

    def input_wrapper(self, sess: tf.Session, coord: tf.train.Coordinator):
        return self.dq.start_input_processes(sess, coord)


def extended_summaries(m: Model):
    sums = []

    *alignment_stats, identity = tf.py_func(
        ops.alignment_stats,
        [
            m.labels.indices, m.labels.values, m.predict.indices,
            m.predict.values, m.labels.dense_shape[0]
        ],
        (len(ops.aligment_stats_ordering) + 1) * [tf.float32],
        stateful=False,
    )

    for stat_type, stat in zip(ops.aligment_stats_ordering, alignment_stats):
        stat.set_shape((None,))
        sums.append(
            tensor_default_summaries(
                dataset_pb2.Cigar.Name(stat_type) + "_rate",
                stat,
            )
        )

    identity.set_shape((None,))
    sums.append(tensor_default_summaries(
        "IDENTITY",
        identity,
    ))
    sums.extend(tensor_default_summaries("logits", m.logits))

    sums.append(
        tf.summary.image(
            "logits",
            tf.expand_dims(
                tf.nn.softmax(tf.transpose(m.logits, [1, 2, 0])),
                -1,
            )
        )
    )
    return sums


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
        cfg = voluptuous.Schema({
            'train': TrainConfig.schema,
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
    try:
        return run(cfg['train'])
    finally:
        logging.getLogger().removeHandler(h)


def run(cfg: TrainConfig):
    tf.reset_default_graph()
    os.makedirs(cfg.logdir, exist_ok=True)
    num_bases = TOTAL_BASE_PAIRS
    if cfg.surrogate_base_pair:
        num_bases += TOTAL_BASE_PAIRS
    try:
        model, ratio = all_models[cfg.model_name](
            n_classes=num_bases + 1, hparams=cfg.model_hparams
        )
        logger.info(f"Compression ratio: {ratio}")
    except voluptuous.error.Error as e:
        logger.error(
            f"Invalid hyper params, check your config {humanize_error(cfg.model_hparams, e)}"
        )
        raise

    input_feeder_cfg = InputFeederCfg(
        batch_size=cfg.batch_size,
        seq_length=cfg.seq_length,
        ratio=ratio,
        surrogate_base_pair=cfg.surrogate_base_pair,
        num_bases=TOTAL_BASE_PAIRS,
    )

    global_step = tf.train.get_or_create_global_step()
    step_inc = tf.assign_add(global_step, 1)
    learning_rate = tf.train.exponential_decay(
        learning_rate=cfg.init_learning_rate,
        global_step=global_step,
        decay_steps=cfg.lr_decay_steps,
        decay_rate=cfg.lr_decay_rate,
    )

    with tf.name_scope("train"):
        train_model = Model(
            input_feeder_cfg,
            model=model,
            data_dir=cfg.train_data,
            trace=cfg.trace,
        )
        # Basic only train summaries
        train_model.summaries.append(
            tf.summary.scalar("learning_rate", learning_rate),
        )

        train_model.summary = tf.summary.merge(train_model.summaries)
        train_model.ext_summary = tf.summary.merge(
            train_model.summaries + extended_summaries(train_model)
        )

    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(train_model.total_loss)
    train_op = optimizer.apply_gradients(
        [(tf.clip_by_norm(grad, cfg.grad_clipping), var)
         for grad, var in grads_and_vars],
        global_step=global_step
    )

    with tf.name_scope("test"):
        test_model = Model(
            input_feeder_cfg,
            model=model,
            data_dir=cfg.test_data,
            trace=cfg.trace,
        )
        var_summaries = []
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            name = var.name.split(":")[0]
            var_summaries.extend(tensor_default_summaries(name, var))

        for grad, var in grads_and_vars:
            if grad is not None:
                name = var.name.split(":")[0]
                var_summaries.extend(
                    tensor_default_summaries(name + "/grad", grad)
                )

        var_summaries.extend(extended_summaries(test_model))
        test_model.summary = tf.summary.merge(
            test_model.summaries + var_summaries
        )

    # Session stuff
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=10)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    beholder = Beholder(cfg.logdir)
    with tf.Session(config=config) as sess:
        if cfg.debug:
            if cfg.tensorboard_debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            else:
                sess = tf_debug.TensorBoardDebugWrapperSession(
                    sess, cfg.tensorboard_debug
                )
        summary_writer = tf.summary.FileWriter(
            os.path.join(cfg.logdir), sess.graph
        )
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
            val_loss = None
            with tqdm(
                total=cfg.train_steps, initial=gs
            ) as pbar, train_model.input_wrapper(
                sess, coord
            ), test_model.input_wrapper(sess, coord):
                for step in range(gs + 1, cfg.train_steps + 1):
                    #  Train hook
                    opts = {}
                    if cfg.run_trace_every > 0 and step % cfg.run_trace_every == 0:
                        opts['options'] = tf.RunOptions(
                            trace_level=tf.RunOptions.FULL_TRACE
                        )
                        opts['run_metadata'] = tf.RunMetadata()

                    summary_op = train_model.summary
                    if step % cfg.validate_every == 0:
                        summary_op = train_model.ext_summary

                    _, _, loss, summary = sess.run([
                        step_inc,
                        train_op,
                        train_model.ctc_loss,
                        summary_op,
                    ], **opts)
                    summary_writer.add_summary(summary, step)

                    if cfg.run_trace_every > 0 and step % cfg.run_trace_every == 0:
                        log_trace(cfg, step, opts, summary_writer)
                    pbar.update()

                    #  Validate hook
                    if step % cfg.validate_every == 0:
                        beholder.update(session=sess)
                        val_loss = log_validation(
                            cfg, sess, step, summary_writer, test_model
                        )

                    pbar.set_postfix(
                        loss=loss, val_loss=val_loss, refresh=False
                    )
                    #  Save hook
                    if step % cfg.save_every == 0:
                        saver.save(
                            sess=sess,
                            save_path=os.path.join(cfg.logdir, 'model.ckpt'),
                            global_step=global_step
                        )
                        logger.info(f"Saved new model checkpoint")
                mean_val_loss = np.mean([
                    log_validation(cfg, sess, None, None, test_model)
                    for _ in range(5)
                ])
                coord.request_stop()
                p = os.path.join(cfg.logdir, f"full-model.save")
                model.save(p, overwrite=True, include_optimizer=False)
                logger.info(f"Finished training saved model to {p}")
            logger.info(f"Input queues exited ok")
            return mean_val_loss
        finally:
            coord.request_stop()
            coord.join(stop_grace_period_secs=5)


def log_validation(
    cfg: TrainConfig, sess: tf.Session, step: int,
    summary_writer: tf.summary.FileWriter, test_model: Model
):
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
        }
    )
    logger.info(
        f"Logits[{logits.shape}]: describe:{pformat(stats.describe(logits, axis=None))}"
    )
    if summary_writer is not None:
        summary_writer.add_summary(test_summary, step)
    return val_loss


def log_trace(
    cfg: TrainConfig, step: int, opts, summary_writer: tf.summary.FileWriter
):
    opts['options'] = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    fetched_timeline = timeline.Timeline(opts['run_metadata'].step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format(
        show_memory=True
    )
    with open(os.path.join(cfg.logdir, f'timeline_{step:05}.json'), 'w') as f:
        f.write(chrome_trace)
    summary_writer.add_run_metadata(
        opts['run_metadata'], f"step_{step:05}", global_step=step
    )
    logger.info(
        f"Saved trace metadata both to timeline_{step:05}.json and step_{step:05} in tensorboard"
    )
