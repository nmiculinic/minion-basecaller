import argparse
import pandas as pd
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
from pprint import pformat
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline
from keras import backend as K
from .models import all_models, AbstractModel, BindedModel
from mincall.common import *
from mincall.train import ops
from minion_data import dataset_pb2
from ._types import *

import toolz
from tqdm import tqdm


logger = logging.getLogger(__name__)


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
    save_every: int = 2000

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
    parser.add_argument(
        "--name",
        help="This model name. It's only used in logs so far",
        default=name_generator()
    )
    parser.set_defaults(func=run_args)
    parser.set_defaults(name="mincall_train")


def run_args(args) -> pd.DataFrame:
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

    logger.info(f"Parsed config\n{pformat(cfg)}")
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)5s]:%(name)20s: %(message)s"
    )
    train_cfg: TrainConfig = cfg['train']
    os.makedirs(train_cfg.logdir, exist_ok=True)
    fn = os.path.join(
        train_cfg.logdir, f"{getattr(args, 'name', 'mincall')}.log"
    )
    h = (logging.FileHandler(fn))
    h.setLevel(logging.INFO)
    h.setFormatter(formatter)
    name_filter = ExtraFieldsFilter({"run_name": args.name})
    root_logger = logging.getLogger()

    root_logger.addHandler(h)
    root_logger.addFilter(name_filter)
    logging.info(f"Added handler to {fn}")
    try:
        with tf.Graph().as_default():
            return run(cfg['train'])
    finally:
        root_logger.removeHandler(h)
        root_logger.removeFilter(name_filter)


def run(cfg: TrainConfig) -> pd.DataFrame:
    try:
        # https://github.com/baidu-research/warp-ctc/tree/master/tensorflow_binding
        import warpctc_tensorflow
        logger.info("Using warpctc_tensorflow GPU kernel")

        # https://github.com/baidu-research/warp-ctc#known-issues---limitations
        max_label_size = 630
    except ImportError:
        max_label_size = 1_000_000_000  #  Unrealistically large number
        logger.info("Cannot use warpctc_tensorflow GPU kernel")
    os.makedirs(cfg.logdir, exist_ok=True)
    num_bases = TOTAL_BASE_PAIRS
    if cfg.surrogate_base_pair:
        num_bases += TOTAL_BASE_PAIRS
    try:
        model = all_models[cfg.model_name](
            n_classes=num_bases + 1, hparams=cfg.model_hparams
        )
        model: AbstractModel = model
        logger.info(f"Compression ratio: {model.ratio}")
    except voluptuous.error.Error as e:
        logger.error(
            f"Invalid hyper params, check your config {humanize_error(cfg.model_hparams, e)}"
        )
        raise

    input_feeder_cfg = InputFeederCfg(
        batch_size=cfg.batch_size,
        seq_length=cfg.seq_length,
        ratio=model.ratio,
        surrogate_base_pair=cfg.surrogate_base_pair,
        num_bases=TOTAL_BASE_PAIRS,
        max_label_size=max_label_size,
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
        train_model = model.bind(
            cfg=input_feeder_cfg,
            data_dir=cfg.train_data,
        )
        # Basic only train summaries
        train_model.summaries.append(
            tf.summary.scalar("learning_rate", learning_rate),
        )

        train_model.summary = tf.summary.merge(train_model.summaries)
        train_model.ext_summary = tf.summary.merge(train_model.ext_summaries)

    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(train_model.total_loss)
        train_op = optimizer.apply_gradients(
            [(tf.clip_by_norm(grad, cfg.grad_clipping), var)
             for grad, var in grads_and_vars],
            global_step=global_step
        )

    with tf.name_scope("test"):
        test_model = model.bind(
            cfg=input_feeder_cfg,
            data_dir=cfg.test_data,
        )
        var_summaries = grad_and_vars_summary(grads_and_vars)
        test_model.summary = tf.summary.merge(
            test_model.ext_summaries + var_summaries
        )

    # Session stuff
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=10)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess, sess.as_default():
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
                    do_trace = cfg.run_trace_every > 0 and step % cfg.run_trace_every == 0
                    #  Train hook
                    logger.debug(f"Starting step {step}")
                    opts = {
                        'options':
                            tf.RunOptions(
                                timeout_in_ms=100 *
                                1000,  # Single op should complete in 100s
                            )
                    }
                    if do_trace:
                        logger.debug("Adding trace options")
                        opts['options'] = tf.RunOptions(
                            trace_level=tf.RunOptions.FULL_TRACE,
                            timeout_in_ms=200 * 1000,
                        )
                        opts['run_metadata'] = tf.RunMetadata()

                    summary_op = train_model.summary
                    if step % cfg.validate_every == 0:
                        summary_op = train_model.ext_summary

                    logger.debug("Sess_run started")
                    _, _, loss, summary = sess.run([
                        step_inc,
                        train_op,
                        train_model.ctc_loss,
                        summary_op,
                    ], **opts)

                    logger.debug("Sess_run finished")
                    summary_writer.add_summary(summary, step)
                    logger.debug("summary writer finished")

                    if do_trace:
                        log_trace(cfg, step, opts, summary_writer)
                    pbar.update()

                    #  Validate hook
                    if step % cfg.validate_every == 0:
                        logger.debug(f"running validation for step {step}")
                        val_loss = log_validation(
                            sess, step, summary_writer, test_model
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
                model.save(sess, cfg.logdir, cfg.train_steps)
                final_val = final_validation(sess, test_model)
                coord.request_stop()
            logger.info(f"Input queues exited ok")
            return final_val
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt")
        except Exception as e:
            logger.critical(
                f"Training interupter! {type(e).__name__}: {e}", exc_info=True
            )
        finally:
            coord.request_stop()
            try:
                coord.join()
            except:
                logger.critical(f"Join unsuccessful, some threads are still alive!")


def grad_and_vars_summary(grads_and_vars):
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
    return var_summaries


def final_validation(
    sess: tf.Session, test_model: BindedModel, min_cnt=100
) -> pd.DataFrame:
    logger = logging.getLogger("mincall.train.ops")
    lvl = logger.getEffectiveLevel()
    logger.setLevel(logging.WARNING)
    sol = None
    while True:
        ctc_loss, *alignment_stats, identity = sess.run(
            [
                test_model.ctc_loss_unaggregated,
                *test_model.alignment_stats,
                test_model.identity,
            ],
            feed_dict={
                test_model.learning_phase: 0,
            },
            options=tf.RunOptions(
                timeout_in_ms=200 * 1000,  # Single op should complete in 200s
            ),
        )

        tmp = pd.DataFrame({
            "ctc_loss": ctc_loss,
            "identity": identity,
            **{
                dataset_pb2.Cigar.Name(op): stat
                for op, stat in zip(
                    ops.aligment_stats_ordering, alignment_stats
                )
            },
        })
        if sol is None:
            sol = tmp
        else:
            sol = sol.append(tmp, ignore_index=True)
        if len(sol) > min_cnt:
            logger.setLevel(lvl)
            return sol


def log_validation(
    sess: tf.Session, step: int, summary_writer: tf.summary.FileWriter,
    test_model: BindedModel
):
    logits, predict, lb, val_loss, losses, test_summary = sess.run(
        [
            test_model.logits,
            test_model.predict,
            test_model.dq.batch_labels,
            test_model.ctc_loss,
            test_model.ctc_loss_unaggregated,
            test_model.summary,
        ],
        feed_dict={
            test_model.learning_phase: 0,
        },
        options=tf.RunOptions(
            timeout_in_ms=20 * 1000,  # Single op should complete in 20s
        ),
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

    logger.debug("Starting trace logging")
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
