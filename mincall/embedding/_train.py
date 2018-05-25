import tensorflow as tf
import random
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
    grad_clipping: float = 5.0
    mode: str = "SkipGram"

    @classmethod
    def schema(cls, data):
        return cls(**voluptuous.Schema(
            {
                'files': str,
                'window': int,
                'stride': int,
                'receptive_field': int,
                'embedding_size': int,
                voluptuous.Optional('mode'): voluptuous.In({"SkipGram"})
            },
            required=True)(data))


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


def get_chunks(fname: str, cfg: EmbeddingCfg) -> List[np.ndarray]:
    with h5py.File(cfg.files, 'r') as f:
        raw_dat = list(f['/Raw/Reads/'].values())[0]
        raw_dat = np.array(raw_dat['Signal'].value)
        raw_dat_processed = scrappy.RawTable(raw_dat).trim().scale().data(
            as_numpy=True)

        chunks = []
        logger.info(f"Shape {raw_dat_processed.shape[0]}")
        for i in trange(
                0,
                raw_dat_processed.shape[0] - cfg.receptive_field,
                cfg.stride,
                desc="reading data",
        ):
            chunks.append(raw_dat_processed[i:i + cfg.receptive_field])
        return chunks


def random_chunk_dataset(cfg: EmbeddingCfg) -> tf.data.Dataset:
    def f():
        for x in random.shuffle(cfg.files):
            for y in random.shuffle(get_chunks(x, cfg)):
                yield y

    return tf.data.Dataset.from_generator(
        f(),
        output_types=(tf.float32, ),
        output_shapes=([cfg.receptive_field], ),
    ).repeat()


def real_chunks_gen(cfg: EmbeddingCfg):
    def f():
        for x in random.shuffle(cfg.files):
            chunks = get_chunks(x, cfg)
            for i in range(len(chunks)):
                for j in range(
                        max(0, i - cfg.window), min(i + cfg.window + 1, len(chunks))):
                    if i != j:
                        yield chunks[j], chunks[i]

    return tf.data.Dataset.from_generator(
        f(),
        output_types=(tf.float32, tf.float32),
        output_shapes=([cfg.receptive_field], [cfg.receptive_field])
    ).repeat()


def run(cfg: EmbeddingCfg):

    global_step = tf.train.get_or_create_global_step()
    step = tf.assign_add(global_step, 1)

    learning_rate = tf.train.exponential_decay(1e-4, global_step, 100000, 0.5)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)

    train_op = optimizer.apply_gradients(
        [(tf.clip_by_norm(grad, cfg.grad_clipping), var)
         for grad, var in grads_and_vars],
        global_step=global_step)

    saver = tf.train.Saver(max_to_keep=10)
    init_op = tf.global_variables_initializer()

    # Basic only train summaries
    summaries = tf.summary.merge([
        tf.summary.scalar("learning_rate", learning_rate),
    ])

    # Extended validation summaries
    var_summaries = []
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        name = var.name.split(":")[0]
        var_summaries.extend(tensor_default_summaries(name, var))

    for grad, var in grads_and_vars:
        if grad is not None:
            name = var.name.split(":")[0]
            var_summaries.extend(
                tensor_default_summaries(name + "/grad", grad))

    dx = tf.data.Dataset.from_generator(
        generator(cfg),
        output_types=(tf.float32, tf.float32),
        output_shapes=([cfg.receptive_field], [cfg.receptive_field])
    ).repeat().shuffle(50).batch(10)
    dx = tf.data.Dataset.zip((dx, dx))
    # create a one-shot iterator
    iterator = dx.make_one_shot_iterator()
    # extract an element
    a, b = iterator.get_next()
    with tf.Session() as sess:
        for i in range(11):
            val = sess.run([a, b])
            print(val)
