import gzip
import time

from concurrent.futures import ThreadPoolExecutor, Future
from typing import *
import logging
from glob import glob
import h5py
from mincall.train.models import custom_layers
from mincall.common import decode, timing_handler
from ._types import *
import scrappy
from mincall.basecall.strategies import *

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


class BasecallMe:
    def __init__(
        self,
        cfg: BasecallCfg,
        signal_2_logit_fn: Callable[[np.ndarray], Future],
        beam_search_fn: Callable[[np.ndarray], Future],
        ratio: int,
        n_classes: int,
    ):
        self.cfg = cfg
        self.beam_search = beam_search_fn
        self.signal_2_logit_fn = signal_2_logit_fn
        self.ratio = ratio
        self.n_classes = n_classes

    def chunkify_signal(self, raw_signal: np.ndarray):
        cfg = self.cfg
        signal_chunks = []
        for i in range(0, len(raw_signal), cfg.jump):
            signal_chunks.append(raw_signal[i:i + cfg.seq_length])
        return signal_chunks

    def chunk_logits(self, chunks: List[np.ndarray]) -> List[np.ndarray]:
        logits = [self.signal_2_logit_fn(ch) for ch in chunks]
        return [l.result() for l in logits]

    def basecall_logits(
        self, raw_signal_len: int, logits_arr: List[np.ndarray]
    ):
        logits = np.zeros(
            shape=((raw_signal_len + self.ratio - 1) // self.ratio,
                   self.n_classes),
            dtype=np.float32
        )

        for i, l in zip(
            reversed(
                range(
                    0, raw_signal_len // self.ratio, self.cfg.jump // self.ratio
                )
            ),
            reversed(logits_arr),
        ):
            logits[i:i + l.shape[0], :] = l
        vals = self.beam_search(logits).result()
        return decode(vals)

    def basecall(self, fname: str):
        try:
            raw_signal = read_fast5_signal(fname)
            log_fname = f"{os.path.splitext(fname)[0][-20:]}[{len(raw_signal)}]"
            chunks = self.chunkify_signal(raw_signal)
            logger.debug(f"Split {fname} into {len(chunks)} overlapping chunks")
            with timing_handler(logger, f"{log_fname}_signal2logits"):
                logits = self.chunk_logits(chunks)
            with timing_handler(logger, f"{log_fname}_ctc_decoding"):
                sol = self.basecall_logits(len(raw_signal), logits)
            return sol
        except:
            logger.critical(f"Cannot basecall {fname}", exc_info=True)
            raise


def run(cfg: BasecallCfg):
    fnames = get_fast5_files(cfg)
    config = tf.ConfigProto()
    with tf.Session(config=config) as sess, sess.as_default():
        model: models.Model = models.load_model(
            cfg.model, custom_objects=custom_layers
        )
        sum = []
        model.summary(print_fn=lambda x: sum.append(x))
        sum = "\n".join(sum)
        logger.info(f"Model summary:\n{sum}")
        n_classes, ratio, surrogate_base_pair = model_props(model)

        s2l = Signal2LogitsSess(
            sess=sess,
            model=model,
        )
        bs = BeamSearchSess(
            sess=sess,
            surrogate_base_pair=surrogate_base_pair,
            beam_width=cfg.beam_width,
        )
        basecaller = BasecallMe(
            cfg=cfg,
            beam_search_fn=bs.beam_search,
            signal_2_logit_fn=s2l.signal2logit_fn,
            ratio=ratio,
            n_classes=n_classes,
        )

        fasta_out_ctor = open
        if cfg.gzip:
            fasta_out_ctor = gzip.open

        logger.info(f"Starting execution with {cfg.threads} workers")
        total_bp = 0
        with timing_handler(logger, "Basecalling all") as start_time, \
                ThreadPoolExecutor(max_workers=cfg.threads) as executor, \
                fasta_out_ctor(cfg.output_fasta, 'wb') as fasta_out, \
                tqdm(total=len(fnames), desc="basecalling files") as pbar:
            for fname, fasta in zip(
                fnames,
                executor.map(basecaller.basecall, fnames),
            ):

                total_bp += len(fasta)
                fasta_out.write(f">{fname}\n".encode("ASCII"))
                for i in range(0, len(fasta), 80):
                    fasta_out.write(f"{fasta[i: i+80]}\n".encode("ASCII"))
                pbar.update()
                pbar.set_postfix(
                    speed=f"{total_bp/(time.time() - start_time):.0f}",
                    refresh=False
                )
        logger.info(
            f"Basecalling speed: {total_bp/(time.time() - start_time)} bp/s"
        )


def get_fast5_files(cfg):
    fnames = []
    for x in cfg.input_dir:
        if os.path.isfile(x):
            if x.endswith(".fast5"):
                fnames.append(x)
            else:
                raise ValueError(f"Isn't a fast5 {x}")
        elif os.path.isdir(x):
            fnames.extend(glob(f"{x}/*.fast5", recursive=cfg.recursive))
    return fnames


def model_props(model):
    """

    :param model:
    :return:  n_classes, ratio, surrogate_base_pair
    """
    test_size = 2**5 * 3**5 * 5**2
    print(
        model.input, model.input_shape, model.output, model.output_shape,
        model.inputs, model.outputs
    )
    # Not sure why sometimes works with [[[ 1, test_size, 1 ]]] and sometimes with [[ 1, test_size, 1 ]]
    # Keras....
    try:
        out_shapes = model.compute_output_shape([
            [1, test_size, 1],
        ])
    except TypeError:
        logger.warning(
            f"Error happend, trying with another indirection level",
            exc_info=True
        )
        out_shapes = model.compute_output_shape([
            [[1, test_size, 1]],
        ])
    _, out_test_size, n_classes = out_shapes
    if n_classes == TOTAL_BASE_PAIRS + 1:
        surrogate_base_pair = False
    elif n_classes == 2 * TOTAL_BASE_PAIRS + 1:
        surrogate_base_pair = True
    else:
        raise ValueError(f"Not sure what to do with {n_classes}")
    logger.info(f"surogate base pair {surrogate_base_pair}")
    ratio, rem = divmod(test_size, out_test_size)
    assert rem == 0, "Reminder should be 0!"
    logger.info(f"Ratio is {ratio}")
    logger.info(f"n_classes is {n_classes}")
    return n_classes, ratio, surrogate_base_pair
