import gzip
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
import tensorflow as tf
import numpy as np
import os
from typing import *
import logging
from glob import glob
from keras import models
import h5py
from mincall.train.models import custom_layers
from mincall.common import TOTAL_BASE_PAIRS, decode, timing_handler
from ._types import *
import scrappy
from threading import Thread

from tqdm import tqdm

logger = logging.getLogger("mincall.basecall")

#######################
# Real code starts
#######################


def read_fast5_signal(fname: str) -> np.ndarray:
    with h5py.File(fname, 'r') as input_data:
        raw_attr = input_data['Raw/Reads/']
        read_name = list(raw_attr.keys())[0]
        raw_signal = np.array(raw_attr[read_name + "/Signal"].value)
        raw_signal = scrappy.RawTable(raw_signal).trim().scale().data(
            as_numpy=True
        )
        logger.debug(f"Read {fname} size: {len(raw_signal)}")
        return raw_signal


class BeamSearch:
    def __init__(self, sess: tf.Session, surrogate_base_pair):
        self.sess = sess
        self.futures = {}
        self.tf_inq = tf.FIFOQueue(
            capacity=10,
            dtypes=[tf.string, tf.float32],
        )

        self.tf_outq = tf.FIFOQueue(
            capacity=10,
            dtypes=[tf.string, tf.int64],
        )

        self.inq_name = tf.placeholder(tf.string)
        self.inq_logits = tf.placeholder(tf.float32)
        self.inq_enqueue = self.tf_inq.enqueue([self.inq_name, self.inq_logits])
        self.inq_close = self.tf_inq.close()

        with tf.name_scope("logits_to_bases"):
            name, logits = self.tf_inq.dequeue()
            self.predict = tf.nn.ctc_beam_search_decoder(
                inputs=tf.transpose(logits, [1, 0, 2]),
                sequence_length=[tf.shape(logits)[1]],
                merge_repeated=surrogate_base_pair,
                top_paths=1,
                beam_width=50
            )
            enq_op = self.tf_outq.enqueue([
                name,
                self.predict[0][0].values,
            ])
        qr = tf.train.QueueRunner(self.tf_outq, [enq_op] * os.cpu_count())
        tf.train.add_queue_runner(qr)
        self.out_dequeue = self.tf_outq.dequeue()
        self.t = Thread(target=self._start, daemon=True)
        self.t.start()

    def _start(self):
        try:
            while True:
                name, ind = self.sess.run(
                    self.out_dequeue,
                )
                name = name.decode("ASCII")
                f = self.futures[name]
                f.set_result(ind)
                del self.futures[name]
        except tf.errors.OutOfRangeError:
            # Means the underlying queue is closed and we can safely exit
            return
        except Exception as ex:
            raise

    def beam_search(self, logits) -> Future:
        f = Future()
        name = uuid.uuid4().hex
        self.futures[name] = f
        self.sess.run(
            self.inq_enqueue,
            feed_dict={
                self.inq_name: name,
                self.inq_logits: logits[np.newaxis, :, :],
            },
        )
        return f

    def stop(self):
        self.sess.run(self.inq_close)
        self.t.join(timeout=10)
        if self.t.is_alive():
            raise ValueError("Thread still alive")


class BasecallMe:
    def __init__(
        self, cfg: BasecallCfg, sess: tf.Session, model: models.Model,
        beam_search_fn: Callable[[np.ndarray], Future]
    ):
        self.cfg = cfg
        self.sess = sess
        self.beam_seach_fn = beam_search_fn

        with tf.name_scope("signal_to_logits"):
            self.signal_batch = tf.placeholder(
                tf.float32, shape=(None, None, 1), name="signal"
            )
            self.logits = model(
                self.signal_batch
            )  # [batch size, max_time, channels]

            self.n_classes = self.logits.shape[-1]
            if self.n_classes == TOTAL_BASE_PAIRS + 1:
                self.surrogate_base_pair = False
            elif self.n_classes == 2 * TOTAL_BASE_PAIRS + 1:
                self.surrogate_base_pair = True
            else:
                raise ValueError(f"Not sure what to do with {self.n_classes}")

        with tf.name_scope("logits_to_bases"):
            self.seq_len_ph = tf.placeholder(tf.int32, shape=(1,))
            self.predict = tf.nn.ctc_beam_search_decoder(
                inputs=tf.transpose(self.logits, [1, 0, 2]),
                sequence_length=self.seq_len_ph,
                merge_repeated=self.surrogate_base_pair,
                top_paths=1,
                beam_width=50
            )

    def chunkify_signal(self, fname: str):
        cfg = self.cfg
        raw_signal = read_fast5_signal(fname)
        signal_chunks = []
        for i in range(0, len(raw_signal), cfg.jump):
            signal_chunks.append(raw_signal[i:i + cfg.seq_length])
        return signal_chunks, len(raw_signal)

    def chunk_logits(self, chunks: List[np.ndarray],
                     batch_size: int = 10) -> List[np.ndarray]:
        cfg = self.cfg
        logits = []
        for i in range(0, len(chunks), batch_size):
            batch = []
            lens = []
            for ch in chunks[i:i + batch_size]:
                lens.append(len(ch))
                if len(ch) < cfg.seq_length:
                    ch = np.pad(
                        ch, (0, cfg.seq_length - len(ch)),
                        'constant',
                        constant_values=0
                    )
                batch.append(ch)
            batch = np.vstack(batch)
            all_logits = self.batch_to_logits(batch)
            ratio = cfg.seq_length // len(all_logits[0])
            for single_logits, ll in zip(all_logits, lens):
                logits.append(single_logits[:ll // ratio])
        return logits, ratio

    def batch_to_logits(self, batch):
        """convert signal batch to logits

        :param batch: [<=batch_size, seq_len]
        :return: logits [<= batch_size, seq_len/ratio]
        """
        return self.sess.run(
            self.logits, feed_dict={self.signal_batch: batch[:, :, np.newaxis]}
        )

    def basecall_logits(
        self, raw_signal_len: int, logits_arr: List[np.ndarray], ratio
    ):
        logits = np.zeros(
            shape=(raw_signal_len // ratio, self.n_classes), dtype=np.float32
        )

        for i, l in zip(
            reversed(range(0, raw_signal_len // ratio, self.cfg.jump // ratio)),
            reversed(logits_arr),
        ):
            logits[i:i + l.shape[0], :] = l

        with timing_handler(logger, "beam search"):
            vals = self.beam_seach_fn(logits).result()
        return decode(vals)

    def basecall(self, fname: str):
        chunks, signal_len = self.chunkify_signal(fname)
        logger.debug(f"Split {fname} into {len(chunks)} overlapping chunks")
        logits, ratio = self.chunk_logits(chunks)
        logger.debug(f"Split {fname} ratio is {ratio}")
        sol = self.basecall_logits(signal_len, logits, ratio)
        logger.debug(f"Basecalled {fname} finalized")
        return sol

    def basecall_full(self, fname: str, ratio: int):
        raw_signal = read_fast5_signal(fname)
        vals = self.sess.run(
            self.predict[0][0].values,
            feed_dict={
                self.seq_len_ph: np.array([len(raw_signal) // ratio]),
                self.signal_batch: raw_signal[np.newaxis, :, np.newaxis]
            }
        )
        return decode(vals)


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

    with tf.Session() as sess, sess.as_default():
        model: models.Model = models.load_model(
            cfg.model, custom_objects=custom_layers
        )
        sum = []
        model.summary(print_fn=lambda x: sum.append(x))
        sum = "\n".join(sum)
        logger.info(f"Model summary:\n{sum}")

        bs = BeamSearch(
            sess=sess,
            surrogate_base_pair=True,
        )

        basecaller = BasecallMe(
            cfg=cfg,
            sess=sess,
            model=model,
            beam_search_fn=bs.beam_search,
        )

        fasta_out_ctor = open
        if cfg.gzip:
            fasta_out_ctor = gzip.open

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            with ThreadPoolExecutor(max_workers=os.cpu_count() or 4
                                   ) as executor, fasta_out_ctor(
                                       cfg.output_fasta, 'wb'
                                   ) as fasta_out:
                for fname, fasta in zip(
                    fnames,
                    tqdm(
                        executor.map(basecaller.basecall, fnames),
                        total=len(fnames),
                        desc="basecalling files"
                    )
                ):
                    fasta_out.write(f">{fname}\n".encode("ASCII"))
                    for i in range(0, len(fasta), 80):
                        fasta_out.write(f"{fasta[i: i+80]}\n".encode("ASCII"))

        except Exception as ex:
            coord.request_stop(ex)
        finally:
            coord.request_stop()
            bs.stop()
