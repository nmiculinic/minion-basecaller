import argparse
from collections import defaultdict
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


class BasecallCfg(NamedTuple):
    input_dir: List[str]
    recursive: bool
    output_fasta: str
    keras_model: str
    graphdef_model: str
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
                voluptuous.Optional('keras_model'): voluptuous.validators.IsFile(),
                voluptuous.Optional('graphdef_model'): voluptuous.validators.IsFile(),
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
        "--keras-model", dest='basecall.keras_model', help="Keras model snapshot")
    parser.add_argument(
        "--graphdef-model", dest='basecall.grapfdef_model', help="graphdef model snapshot ((one where variables are converted into constants)) ")
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


class SignalFeeder:
    def __init__(self, max_seq_len, jump, capacity=1000):
        """SignalFeeder handled signal queue

        :param max_seq_len: max stripe length
        :param jump: how much to jump between stripes
        :param capacity: signal queue capacity
        """
        self.max_seq_len = max_seq_len
        self.jump = jump
        self.logger = logging.getLogger(__name__ + ".SignalFeeder")

        self.signal_fname_ph = tf.placeholder(tf.string, shape=(), name="signal_fname")
        self.signal_start_ph = tf.placeholder(tf.int32, shape=(), name="signal_start")
        self.signal_ph = tf.placeholder(tf.float32, shape=(max_seq_len, 1), name="signal")
        self.signal_length_ph = tf.placeholder(tf.int32, shape=(), name="signal_length")

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
        with tqdm(total=len(fnames), desc="loading files into memory") as pbar:
            for fn in fnames:
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
                        if signal_len < self.max_seq_len:
                           signal = np.pad(signal, (0, self.max_seq_len - signal_len), 'constant', constant_values=0)
                        try:
                            _, size = sess.run(
                                [
                                    self.signal_enqueue,
                                    self.signal_queue_size,
                                    ],
                                feed_dict={
                                    self.signal_fname_ph: fn,
                                    self.signal_start_ph: i,
                                    self.signal_ph: signal.reshape([-1, 1]),
                                    self.signal_length_ph: signal_len,
                                })
                            pbar.set_postfix(q_len=size)
                        except tf.errors.CancelledError:
                            if coord.should_stop():
                                sess.run(self.signal_queue_close)
                                return
                pbar.update()

class LogitProcessing:
    def __init__(self,
                 signal: SignalFeeder,
                 batch_size: int,
                 keras_model: models.Model=None,
                 graphdef_file=None,
                 capacity: int = 3000,
                 num_threads=4):
        self.logger = logging.getLogger(__name__ + ".LogitProcessing")
        assert batch_size <= capacity, f"Batch size {batch_size} is bigger than capacity {capacity}"
        signal_fname, signal_start, signal, signal_length = signal.signal_queue.dequeue_up_to(
            batch_size)

        if keras_model:
            logits = keras_model(signal)
            self.logger.info("Using keras model")
        elif graphdef_file:
            self.logger.info("Using graphdef file")
            with open(graphdef_file,'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            logits = tf.import_graph_def(
                graph_def,
                input_map={
                    "signal:0": signal,
                },
                return_elements=["logits:0"],
                name=None,
                op_dict=None,
                producer_op_list=None
            )
        else:
            raise ValueError("Neither keras no graphdef models defined")
        ratio, rem = divmod(int(signal.shape[1]), int(logits.shape[1]))
        assert rem == 0, "Non clear cut for signal, {signal.shape[1]}/{logits.shape[1]}"
        self.logger.info(f"Signal2Logit squeeze ratio {ratio}  = {signal.shape[1]}/{logits.shape[1]}")
        self.vs = [
            signal_fname,
            signal_start,
            logits,
            tf.cast(signal_length / ratio, dtype=tf.int32),
        ]
        self.logits_queue = tf.FIFOQueue(
            capacity=capacity,
            dtypes=[x.dtype for x in self.vs],
            shapes=[x.shape[1:] for x in self.vs],
        name="logits_queue")
        self.logit_queue_size = self.logits_queue.size()
        self.logit_queue_close = self.logits_queue.close()

        self.logit_enqueue = self.logits_queue.enqueue_many(self.vs, "enqueue_logits")
        self.logit_dequeue = self.logits_queue.dequeue_up_to(batch_size)
        self.logit_qr = tf.train.QueueRunner(
            self.logits_queue, [self.logit_enqueue] * num_threads)
        tf.train.add_queue_runner(self.logit_qr)

class Basecall:
    def __init__(self, max_seq_len:int, jump:int, logit_processing: LogitProcessing, output_file: str = None):
        self.max_seq_len = max_seq_len
        self.jump = jump
        self.output_file = output_file
        self.logger = logging.getLogger(__name__ + ".Basecall")
        self.logit_processing = logit_processing

        self.read_fname,\
        self.read_start,\
        self.logits,\
        self.signal_length = self.logit_processing.logit_dequeue
        self._construct_graph()

    def basecall_all(self, sess: tf.Session, coord: tf.train.Coordinator, fnames: List[str]):
        with tqdm(total=len(fnames), desc="basecalling all") as pbar, open(self.output_file, 'w') as fasta_out:
            cache = defaultdict(dict)
            for fn in fnames:
                with h5py.File(fn, 'r') as input_data:
                    raw_attr = input_data['Raw/Reads/']
                    read_name = list(raw_attr.keys())[0]
                    raw_signal_len = len(np.array(raw_attr[read_name + "/Signal"].value))

                    for i in trange(0, raw_signal_len, self.jump, desc="stripes inserted"):
                        while i not in cache[fn]:
                            if coord.should_stop():
                                self.logger.warning(f"Coord should stop killing")
                                return
                            try:
                                bread_fname, bread_start, blogits, signal_len, size = sess.run(
                                    [
                                        self.read_fname,
                                        self.read_start,
                                        self.logits,
                                        self.signal_length,
                                        self.logit_processing.logit_queue_size,
                                    ])
                                for j in range(bread_fname.shape[0]):
                                    read_start = bread_start[j]
                                    read_fname = bread_fname[j].decode("UTF-8")
                                    cache[read_fname][read_start] = blogits[j, :signal_len[j], :]
                                    self.logger.debug(f"Inserted {read_fname}:{read_start}, wants {fn}:{i}")
                                pbar.set_postfix(logit_q_size=size)
                            except tf.errors.CancelledError:
                                if coord.should_stop():
                                    return
                    assembly = cache.pop(fn)
                    predicted, log_prob = self.construct_stripes(sess, assembly, raw_signal_len)
                    fasta = "".join([dataset_pb2.BasePair.Name(x) for x in predicted[0].values])

                    print(f">{fn}", file=fasta_out)
                    for i in range(0, len(fasta), 80):
                        print(fasta[i:i+80], file=fasta_out)
                    self.logger.info(f"Decoded: {fn} to file {self.output_file}")
                pbar.update()

    def _construct_graph(self):
        self.logits_ph = tf.placeholder(tf.float32, shape=(None, 1, 5))
        self.seq_len = tf.placeholder(tf.int32, shape=(1, ))

        self.predict = tf.nn.ctc_beam_search_decoder(
            inputs=self.logits_ph,
            sequence_length=self.seq_len,
            merge_repeated=False,
            top_paths=1,
            beam_width=50)

    def construct_stripes(self, sess: tf.Session, assembly: Dict[int, np.ndarray], raw_signal_len):
        logits = np.zeros(shape=(raw_signal_len, 5), dtype=np.float32)
        for i in reversed(range(0, raw_signal_len, self.jump)):
            l = assembly[i]
            logits[i:i + l.shape[0], :] = l

        return sess.run(self.predict, feed_dict={
            self.logits_ph: logits.reshape(-1, 1, 5),
            self.seq_len: np.array([raw_signal_len])
        })


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

    with tf.Session() as sess:
        keras_model: models.Model = None
        if cfg.keras_model is not None:
            keras_model: models.Model = models.load_model(cfg.keras_model)
            sum = []
            keras_model.summary(print_fn=lambda x: sum.append(x))
            sum = "\n".join(sum)
            logger.info(f"Model summary:\n{sum}")

        signal_feeder = SignalFeeder(
            max_seq_len=cfg.seq_length,
            jump=cfg.jump,
        )

        logit_processing = LogitProcessing(
            signal=signal_feeder,
            batch_size=cfg.batch_size,
            keras_model=keras_model,
            graphdef_file=cfg.graphdef_model,
        )

        basecall = Basecall(
            max_seq_len=cfg.seq_length,
            jump=cfg.jump,
            logit_processing= logit_processing,
            output_file=cfg.output_fasta,
        )

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)

        threads = []
        close_fns = []
        try:

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

            t = Thread(
                target=basecall.basecall_all,
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
