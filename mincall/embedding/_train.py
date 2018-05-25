import tensorflow as tf
import random
from glob import glob
import scrappy
from tqdm import trange, tqdm
import numpy as np
import sys
import h5py
import argparse
import logging
from typing import *
import toolz
import yaml
import voluptuous
from keras import models, layers, regularizers, constraints, backend as K
from voluptuous.humanize import humanize_error
from pprint import pformat
from tensorflow.python import debug as tf_debug
import os
from tensorflow.python.client import timeline
from tensorboard.plugins.beholder import Beholder
from mincall.common import *

logger = logging.getLogger(__name__)


class EmbeddingCfg(NamedTuple):
    """
    "word" in this context is vector representing subsample on receptive_field consecutive raw signal samples
    """
    files: str
    window: int  # How big is the surrounding elements window
    receptive_field: int  # Receptive field -- e.g. how many raw signal samples consists single "word"
    stride: int  # How many raw signal steps to take between adjacent "words"
    embedding_size: int
    train_steps: int
    batch_size: int
    logdir: str
    run_trace_every: int = 10000
    grad_clipping: float = 5.0
    mode: str = "SkipGram"
    init_learning_rate: float = 1e-3
    lr_decay_steps: int = 10000
    lr_decay_rate: float = 0.5
    debug: bool = False
    tensorboard_debug: str = None
    save_every: int = 5000

    @classmethod
    def schema(cls, data):
        return named_tuple_helper(
            cls, {
                voluptuous.Optional('mode'): voluptuous.In({"SkipGram"}),
                'files': voluptuous.IsDir(),
            }, data)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--config", "-c", help="config file")
    parser.add_argument(
        "--files", "-f", dest="embed.files", help="target files")
    parser.add_argument(
        "--window", "-w", dest="embed.window", help="window size")
    parser.add_argument(
        "--stride", "-s", dest="embed.stride", help="Stride size")
    parser.add_argument(
        "--embedding-size",
        dest="embed.embedding_size",
        help="Size of resulting embedding")
    parser.set_defaults(func=run_args)
    parser.set_defaults(name="mincall_embedding")


def run_args(args):
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.load(f)
    else:
        config = {'version': "v0.1"}

    for k, v in vars(args).items():
        if v is not None and "." in k:
            config = toolz.assoc_in(config, k.split("."), v)
    if args.logdir is not None:
        config['embed']['logdir'] = args.logdir
    try:
        cfg = voluptuous.Schema(
            {
                'embed': EmbeddingCfg.schema,
                'version': str,
            },
            extra=voluptuous.REMOVE_EXTRA,
            required=True)(config)
        logger.info(f"Parsed config\n{pformat(cfg)}")
        run(cfg['embed'])
    except voluptuous.error.Error as e:
        logger.error(humanize_error(config, e))
        sys.exit(1)


def run(cfg: EmbeddingCfg):
    model, loss = create_model(cfg)
    train_model(cfg, model, loss)


################
# Input pipeline
################


def get_chunks(fname: str, cfg: EmbeddingCfg) -> List[np.ndarray]:
    with h5py.File(fname, 'r') as f:
        raw_dat = list(f['/Raw/Reads/'].values())[0]
        raw_dat = np.array(raw_dat['Signal'].value)
        raw_dat_processed = scrappy.RawTable(raw_dat).trim().scale().data(
            as_numpy=True)

        chunks = []
        for i in range(
                0,
                raw_dat_processed.shape[0] - cfg.receptive_field,
                cfg.stride,
        ):
            chunks.append(raw_dat_processed[i:i + cfg.receptive_field])
        return chunks


def random_chunk_dataset(cfg: EmbeddingCfg) -> tf.data.Dataset:
    def f():
        file_list = list(glob(cfg.files + "/*.fast5"))
        while True:
            random.shuffle(file_list)
            for x in file_list:
                chunks = get_chunks(x, cfg)
                random.shuffle(chunks)
                for y in chunks:
                    yield list(y)

    return tf.data.Dataset.from_generator(
        f,
        output_types=tf.float32,
        output_shapes=[cfg.receptive_field],
    ).shuffle(1024)


def real_chunks_gen(cfg: EmbeddingCfg):
    def f():
        file_list = list(glob(cfg.files + "/*.fast5"))
        while True:
            random.shuffle(file_list)
            for x in file_list:
                chunks = get_chunks(x, cfg)
                for i in range(len(chunks)):
                    for j in range(
                            max(0, i - cfg.window),
                            min(i + cfg.window + 1, len(chunks))):
                        if i != j:
                            yield chunks[j], chunks[i]

    return tf.data.Dataset.from_generator(
        f,
        output_types=(tf.float32, tf.float32),
        output_shapes=([cfg.receptive_field],
                       [cfg.receptive_field])).shuffle(1024)


###########################
# Model creation & training
###########################


def create_model(cfg: EmbeddingCfg) -> Tuple[models.Model, tf.Tensor]:
    context, target = real_chunks_gen(cfg).batch(
        cfg.batch_size).make_one_shot_iterator().get_next()
    noise = random_chunk_dataset(cfg).batch(
        cfg.batch_size).make_one_shot_iterator()

    model = models.Sequential([
        layers.InputLayer(input_shape=[cfg.receptive_field]),
        layers.Dense(cfg.embedding_size),
        layers.Dense(cfg.receptive_field),
    ])

    up = tf.reduce_mean(tf.nn.l2_loss(model(context) - target))
    down = tf.add_n([
        tf.reduce_mean(tf.nn.l2_loss(model(context) - noise.get_next()))
        for _ in range(5)
    ])
    loss = up - down
    return model, loss


def train_model(cfg: EmbeddingCfg, model: models.Model, loss: tf.Tensor):
    global_step = tf.train.get_or_create_global_step()
    step = tf.assign_add(global_step, 1)

    learning_rate = tf.train.exponential_decay(
        learning_rate=cfg.init_learning_rate,
        global_step=global_step,
        decay_steps=cfg.lr_decay_steps,
        decay_rate=cfg.lr_decay_rate,
    )
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)

    train_op = optimizer.apply_gradients(
        [(tf.clip_by_norm(grad, cfg.grad_clipping), var)
         for grad, var in grads_and_vars],
        global_step=global_step)

    saver = tf.train.Saver(max_to_keep=10)
    init_op = tf.global_variables_initializer()

    # Basic only train summaries
    summaries = [
        tf.summary.scalar("learning_rate", learning_rate),
    ]

    # Extended validation summaries
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        name = var.name.split(":")[0]
        summaries.extend(tensor_default_summaries(name, var))

    for grad, var in grads_and_vars:
        if grad is not None:
            name = var.name.split(":")[0]
            summaries.extend(tensor_default_summaries(name + "/grad", grad))

    merged_summary = tf.summary.merge(summaries)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
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

        gs = sess.run(global_step)
        pbar = trange(gs, cfg.train_steps)
        for i in pbar:
            #  Train hook
            opts = {}
            if cfg.run_trace_every > 0 and i % cfg.run_trace_every == 0:
                opts['options'] = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                opts['run_metadata'] = tf.RunMetadata()

            _, _, curr_loss, summary = sess.run([
                step,
                train_op,
                loss,
                merged_summary,
            ], **opts)
            summary_writer.add_summary(summary, i)
            pbar.set_postfix(loss=curr_loss)

            if cfg.run_trace_every > 0 and i % cfg.run_trace_every == 0:
                opts['options'] = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                fetched_timeline = timeline.Timeline(
                    opts['run_metadata'].step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format(
                    show_memory=True)
                with open(
                        os.path.join(cfg.logdir, f'timeline_{i:05}.json'),
                        'w') as f:
                    f.write(chrome_trace)
                summary_writer.add_run_metadata(
                    opts['run_metadata'], f"step_{i:05}", global_step=i)
                logger.info(
                    f"Saved trace metadata both to timeline_{i:05}.json and step_{i:05} in tensorboard"
                )

            beholder.update(session=sess)

            #  Save hook
            if i % cfg.save_every == 0:
                saver.save(
                    sess=sess,
                    save_path=os.path.join(cfg.logdir, 'model.ckpt'),
                    global_step=global_step)
                logger.info(f"Saved new model checkpoint")
        p = os.path.join(cfg.logdir, f"full-model.save")
        model.save(p, overwrite=True, include_optimizer=False)
        logger.info(f"Finished training saved model to {p}")
