import argparse
import tensorflow as tf
import string
import logging
from tqdm import trange
from threading import Thread
import numpy as np
import os
import voluptuous
import sys
import yaml
from typing import *
import logging
from voluptuous.humanize import humanize_error
from glob import glob
from pprint import pformat, pprint
from minion_data import dataset_pb2
from keras import models
import h5py

import toolz
from tqdm import tqdm

logger = logging.getLogger("mincall.basecall")


def decode(x):
    return "".join(map(dataset_pb2.BasePair.Name, x))


class Read(NamedTuple):
    name: str
    signal: np.ndarray


class BasecallCfg(NamedTuple):
    input_dir: List[str]
    recursive: bool
    output_fasta: str
    model: str
    batch_size: int
    seq_length: int
    beam_width: int
    logdir: str
    jump: int

    @classmethod
    def schema(cls, data):
        return cls(**voluptuous.Schema(
            {
                'input_dir':
                voluptuous.All(
                    voluptuous.validators.Length(min=1),
                    [voluptuous.Any(voluptuous.IsDir(), voluptuous.IsFile())],
                ),
                voluptuous.Optional('recursive', default=False):
                bool,
                'output_fasta':
                str,
                'model':
                voluptuous.validators.IsFile(),
                voluptuous.Optional('batch_size', default=1100):
                int,
                voluptuous.Optional('seq_length', default=300):
                int,
                voluptuous.Optional('beam_width', default=50):
                int,
                voluptuous.Optional('jump', default=30):
                int,
                voluptuous.Optional('logdir', default=None):
                voluptuous.Any(str, None),
            },
            required=True)(data))


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--config", "-c", help="config file", required=False)
    parser.add_argument("--in", "-i", nargs="*", dest='basecall.input_dir')
    parser.add_argument("--out", "-o", dest='basecall.output_fasta')
    parser.add_argument(
        "--model", "-m", dest='basecall.model', help="model savepoint")
    parser.add_argument(
        "--batch_size", "-b", dest='basecall.batch_size', type=int)
    parser.add_argument(
        "--seq_length",
        "-l",
        dest='basecall.seq_length',
        type=int,
        help="segment length")
    parser.add_argument("--jump", "-j", dest='basecall.jump', type=int)
    parser.add_argument(
        '--beam',
        dest='basecall.beam_width',
        type=int,
        default=50,
        help=
        "Beam width used in beam search decoder, default is 50, set to 0 to use a greedy decoder. Large beam width give better decoding result but require longer decoding time."
    )
    parser.set_defaults(func=run_args)
    parser.set_defaults(name="mincall_basecall")


def run_args(args):
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.load(f)
    else:
        config = {
            'basecall': {},
            'version': "v0.1",
        }

    for k, v in vars(args).items():
        if v is not None and "." in k:
            config = toolz.assoc_in(config, k.split("."), v)

    if args.logdir is not None:
        config['basecall']['logdir'] = args.logdir
    try:
        cfg = voluptuous.Schema(
            {
                'basecall': BasecallCfg.schema,
                'version': str,
            },
            extra=voluptuous.REMOVE_EXTRA,
            required=True)(config)
        logger.info(f"Parsed config\n{pformat(cfg)}")
        run(cfg['basecall'])
    except voluptuous.error.Error as e:
        logger.error(humanize_error(config, e))
        sys.exit(1)


def decoding_queue(cfg: BasecallCfg, logits_queue, num_threads=6):
    q_logits, q_name, q_index, seq_length = logits_queue.dequeue()
    if cfg.beam_width == 0:
        decode_decoded, decode_log_prob = tf.nn.ctc_greedy_decoder(
            tf.transpose(q_logits, perm=[1, 0, 2]),
            seq_length,
            merge_repeated=True)
    else:
        decode_decoded, decode_log_prob = tf.nn.ctc_beam_search_decoder(
            tf.transpose(q_logits, perm=[1, 0, 2]),
            seq_length,
            merge_repeated=False,
            beam_width=cfg.beam_width)
        # There will be a second merge operation after the decoding process
        # if the merge_repeated for decode search decoder set to True.
        # Check this issue https://github.com/tensorflow/tensorflow/issues/9550
    decodeedQueue = tf.FIFOQueue(
        capacity=2 * num_threads,
        dtypes=[tf.int64 for _ in decode_decoded] * 3 +
        [tf.float32, tf.float32, tf.string, tf.int32],
    )
    ops = []
    for x in decode_decoded:
        ops.append(x.indices)
        ops.append(x.values)
        ops.append(x.dense_shape)
    decode_enqueue = decodeedQueue.enqueue(
        tuple(ops + [decode_log_prob, q_name, q_index]))

    decode_dequeue = decodeedQueue.dequeue()
    decode_fname, decode_idx = decode_dequeue[-2:]

    decode_dequeue = decode_dequeue[:-3]
    decode_predict = [[], decode_dequeue[-1]]
    for i in range(0, len(decode_dequeue) - 1, 3):
        decode_predict[0].append(
            tf.SparseTensor(
                indices=decode_dequeue[i],
                values=decode_dequeue[i + 1],
                dense_shape=decode_dequeue[i + 2],
            ))

    decode_qr = tf.train.QueueRunner(decodeedQueue,
                                     [decode_enqueue] * num_threads)
    tf.train.add_queue_runner(decode_qr)
    return decode_predict, decode_fname, decode_idx, decodeedQueue.size()


class SignalFeeder():
    def __init__(self, max_seq_len, jump, capacity=1000):
        """SignalFeeder handled signal queue

        :param max_seq_len: max stripe length
        :param jump: how much to jump between stripes
        :param capacity: signal queue capacity
        """
        self.max_seq_len = max_seq_len
        self.jump = jump
        self.logger = logging.getLogger(__name__ + ".SignalFeeder")

        self.signal_fname_ph = tf.placeholder(tf.string, shape=())
        self.signal_start_ph = tf.placeholder(tf.int32, shape=())
        self.signal_ph = tf.placeholder(tf.float32, shape=(max_seq_len, 1))
        self.signal_length_ph = tf.placeholder(tf.int32, shape=())

        vs = [
            self.signal_fname_ph,
            self.signal_start_ph,
            self.signal_ph,
            self.signal_length_ph,
        ]
        self.signal_queue = tf.FIFOQueue(
            name="signal_queue",
            capacity=capacity,
            dtypes=[x.dtype for x in vs],
            shapes=[x.shape for x in vs])
        self.signal_queue_size = self.signal_queue.size()
        self.signal_queue_close = self.signal_queue.close(
            cancel_pending_enqueues=True)
        self.signal_enqueue = self.signal_queue.enqueue(vs)

    def feed_all(self, sess: tf.Session, coord: tf.train.Coordinator,
                 fnames: List[str]):
        for fn in tqdm(fnames, "loading files into memory"):
            with h5py.File(fn, 'r') as input_data:
                raw_attr = input_data['Raw/Reads/']
                read_name = list(raw_attr.keys())[0]
                raw_signal = np.array(raw_attr[read_name + "/Signal"].value)
                for i in trange(
                        0, len(raw_signal), self.jump,
                        desc="stripes inserted"):
                    if coord.should_stop():
                        self.logger.warning(f"Coord should stop killing")
                        sess.run(self.signal_queue_close)
                        return

                    signal = raw_signal[i:i + self.max_seq_len]
                    signal_len = len(signal)
                    try:
                        sess.run(
                            self.signal_enqueue,
                            feed_dict={
                                self.signal_fname_ph: fn,
                                self.signal_start_ph: i,
                                self.signal_ph: signal.reshape([-1, 1]),
                                self.signal_length_ph: signal_len,
                            })
                    except tf.errors.CancelledError:
                        if coord.should_stop():
                            sess.run(self.signal_queue_close)
                            return


class LogitProcessing:
    def __init__(self,
                 signal: SignalFeeder,
                 batch_size: int,
                 model: models.Model,
                 capacity: int = 1000,
                 num_threads=4):
        signal_fname, signal_start, signal, signal_length = signal.signal_queue.dequeue_up_to(
            batch_size)

        ratio = 1  # TODO: Compute this during runtime
        vs = [
            signal_fname,
            signal_start,
            model(signal),
            signal_length / ratio,
        ]
        self.logits_queue = tf.FIFOQueue(
            capacity=capacity,
            dtypes=[x.dtype for x in vs],
            shapes=[x.shape for x in vs])
        self.logit_queue_size = self.logits_queue.size()
        self.logit_queue_close = self.logits_queue.close()

        self.logit_enqueue = self.logits_queue.enqueue_many(
            self.logits_queue, vs)
        self.logit_qr = tf.train.QueueRunner(
            self.logits_queue, [self.logit_enqueue] * num_threads)


def run(cfg: BasecallCfg):
    fnames = []
    for x in cfg.input_dir:
        if os.path.isfile(x):
            if x.endswith(".fast5"):
                fnames.append(x)
            else:
                raise ValueError(f"Isn't a fast5 {x}")
        elif os.path.isdir(x):
            fnames.extend(glob(f"{x}/*.fast5", recursive=cfg.recursive))

    signal_feeder = SignalFeeder(
        max_seq_len=cfg.seq_length,
        jump=cfg.jump,
    )

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)

        threads = []
        close_fns = []
        try:
            model: models.Model = models.load_model(cfg.model)
            sum = []
            model.summary(print_fn=lambda x: sum.append(x))
            sum = "\n".join(sum)
            logger.info(f"Model summary:\n{sum}")

            t = Thread(
                target=signal_feeder.feed_all,
                kwargs={
                    'sess': sess,
                    'coord': coord,
                    'fnames': fnames,
                },
                daemon=False)
            threads.append(t)
            t.start()
            close_fns.append(
                lambda: sess.run(signal_feeder.signal_queue_close))

            for t in threads:
                t.join()
        finally:
            exc_type, exc_value, tb = sys.exc_info()
            if exc_value is None or isinstance(exc_value, KeyboardInterrupt):
                coord.request_stop()
            else:
                logging.critical("Critical error happened", exc_info=True)
                coord.request_stop(exc_value)

            for cl in close_fns:
                cl()
            for t in threads:
                t.join()
            coord.join()
