import os
import socket
from random import shuffle, randint
import numpy as np
import h5py
from glob import glob
from collections import defaultdict
import sys

from mincall import util
from mincall import errors
from mincall.errors import *

root_dir_map = {
    'karla': '/hgst8TB/fjurisic/ecoli',
    'protagonist': '/home/lpp/Downloads/minion',
    'inspiron5520': '/data',
    'r9': '/data/r9'
}
root_dir_default = root_dir_map.get(socket.gethostname(), '/data')


def find_ref(fast5_path):
    dirname = os.path.dirname(fast5_path)
    basename = os.path.basename(fast5_path)
    name, ext = os.path.splitext(basename)
    ref_path = os.path.join(dirname, name + '.ref')

    if not os.path.exists(ref_path):
        raise errors.RefFileNotFound(
            "{}-> search query:{}.ref in following directory {}".format(basename, name, dirname))

    return ref_path


class InputReader():
    def input_fn(self):
        raise NotImplemented

    def get_signal(self, fast5_path):
        raise NotImplemented


class AlignedRawAbstract(InputReader):
    def __init__(self, nedit_tol=0.35):
        super()
        self.nedit_tol = nedit_tol

    def preprocessSignal(self, signal):
        return (signal - 646.11133) / 75.673653

    def get_basecalled_data(self, h5, target_read, sampling_rate):
        raise NotImplemented

    def read_fast5(self, fast5_path):
        with h5py.File(fast5_path, 'r') as h5:
            reads = h5['Raw/Reads']
            target_read = list(reads.keys())[0]
            sampling_rate = h5['UniqueGlobalKey/channel_id'].attrs['sampling_rate']

            basecalled, fastq = self.get_basecalled_data(h5, target_read, sampling_rate)

            signal = h5['Raw/Reads/' + target_read]['Signal']
            start_time = basecalled[0]['start']
            start_pad = int(sampling_rate * basecalled[0]['start'])
            signal_len = int(sampling_rate * (basecalled[-1]['start'] + basecalled[-1]['length'] - basecalled[0]['start']))

            # np.testing.assert_allclose(len(signal), start_pad + signal_len, rtol=1e-2)  # Within 1% relative tolerance, TODO check for HMM and RNN discrepancy

            basecalled['start'] -= start_time
            signal = signal[start_pad:start_pad + signal_len]
            signal = self.preprocessSignal(signal)

            return {
                'signal': signal,
                'basecalled': basecalled,
                'sampling_rate': sampling_rate,
                'fastq': fastq,
                'start_pad': start_pad,
                'signal_len': signal_len
            }

    def read_fast5_raw_ref(self, fast5_path, ref_path, block_size_x, block_size_y, num_blocks, n_samples_per_ref=1,
                           verify_file=True):

        with open(ref_path, 'r') as ref_file:
            fast5 = self.read_fast5(fast5_path)

            signal = fast5['signal']
            basecalled = fast5['basecalled']
            sampling_rate = fast5['sampling_rate']
            fastq = fast5['fastq']

            # start, length, model_state, move
            bucketed_basecall = [basecalled[0]['model_state'].decode("ASCII")]
            for b in basecalled[1:]:
                if b['move'] != 0:
                    target_bucket = int(np.floor(b['start'] * sampling_rate / block_size_x))
                    while len(bucketed_basecall) <= target_bucket:
                        bucketed_basecall.append("")
                    bucketed_basecall[target_bucket] += \
                        b['model_state'][-b['move']:].decode("ASCII")

            # -2 for first and last that are always skipped
            num_blocks_max = len(bucketed_basecall) - 2
            if num_blocks_max < num_blocks:
                raise InsufficientDataBlocks("Has only {} blocks, while requesting {} + first and last"
                                             .format(len(bucketed_basecall), num_blocks))

            ref_ext = os.path.splitext(ref_path)[1]
            if ref_ext == ".ref":
                ref_seq = ref_file.readlines()[3].strip()
            elif ref_ext == ".fasta":
                ref_seq = util.read_fasta(ref_file)
            else:
                raise ValueError("extension not recognized %s" % ref_ext)

            ref_seq = ref_seq.upper()
            corrected_basecalled = util.correct_basecalled(bucketed_basecall, ref_seq, nedit_tol=self.nedit_tol)

            if verify_file:
                # Sanity check, correctly basecalled files
                np.testing.assert_string_equal("".join(bucketed_basecall), fastq[1])
                np.testing.assert_string_equal("".join(corrected_basecalled), ref_seq)

            n_different_blocks = num_blocks_max-num_blocks+1
            n_different_non_overlap_blocks = n_different_blocks // num_blocks
            n_samples = min(n_samples_per_ref, n_different_non_overlap_blocks)

            for i in range(n_samples):
                x = np.zeros([block_size_x * num_blocks, 1], dtype=np.float32)
                x_len = min(len(signal), block_size_x * num_blocks)

                # start from i-th block
                # does not guarantee non overlapping blocks but ok enough
                start_block = randint(1, n_different_blocks)
                x_offset = start_block*block_size_x
                x[:x_len, 0] = signal[x_offset:x_offset + x_len]
                y, y_len = util.prepare_y(corrected_basecalled[start_block:start_block + num_blocks], block_size_y)
                yield x, x_len, y, y_len

    def find_ref(self, fast5_path):
        dirname = os.path.dirname(fast5_path)
        basename = os.path.basename(fast5_path)
        name, ext = os.path.splitext(basename)
        ref_path = os.path.join(dirname, name + '.ref')
        if not os.path.exists(ref_path):
            raise errors.RefFileNotFound(
                "{}-> search query:{}.ref in following directory {}".format(basename, name, dirname))

        return ref_path

    def input_fn(self, model, subdir, root_dir=None):
        if root_dir is None:
            root_dir = root_dir_default

        root_dir = os.path.join(root_dir, subdir)
        if not os.path.exists(root_dir) or not os.path.isdir(root_dir):
            model.logger.error('Invalid input folder %s, expected root dir with .fast5 and .ref', root_dir)

        items = glob(os.path.join(root_dir, '*.fast5'))
        names = ["X", "X_len", "Y", "Y_len"]

        total = 0
        errors = defaultdict(int)

        while True:
            shuffle(items)
            for fast5_path in items:
                ref_path = find_ref(fast5_path)

                total += 1
                try:
                    sols = self.read_fast5_raw_ref(
                        fast5_path,
                        ref_path,
                        block_size_x=model.block_size_x,
                        block_size_y=model.block_size_y,
                        num_blocks=model.num_blocks,
                        n_samples_per_ref=model.n_samples_per_ref,
                        verify_file=False
                    )
                    for sol in sols:
                        yield {
                            name + "_enqueue_val": np.array([arr]) for name, arr in zip(names, sol)
                        }
                except KeyboardInterrupt:
                    raise
                except MinIONBasecallerException as ex:
                    errors[type(ex).__name__] += 1
                except Exception as ex:
                    errors[type(ex).__name__] += 1
                    model.logger.error('in filename %s \n' % fast5_path, exc_info=True)

                if total % 10000 == 0 or total in [10, 100, 200, 1000]:
                    model.logger.info(
                        "{%s} read %d datapoints: %s ", subdir,
                        total,
                        " ".join(["{}: {:.2f}%".format(key, 100 * errors[key] / total) for key in sorted(errors.keys())])
                    )

    in_dim = 1

    def get_signal(self, fast5_path):
        f5 = self.read_fast5(fast5_path)
        signal = f5['signal']
        start_pad = f5['start_pad']
        return signal.reshape(1, -1, 1), start_pad


class HMMAlignedRaw(AlignedRawAbstract):
    def get_basecalled_data(self, h5, target_read, sampling_rate):
        basecalled_events = h5['/Analyses/Basecall_1D_000/BaseCalled_template/Events']
        basecalled_events = np.array(basecalled_events.value[['start', 'length', 'model_state', 'move']])

        raw_start_time = h5['Raw/Reads/' + target_read].attrs['start_time']
        events = h5['/Analyses/EventDetection_000/Reads/' + target_read]
        event_start_time = events.attrs['start_time']

        basecalled_events['start'] -= events['Events'][0]['start'] / sampling_rate
        basecalled_events['start'] += (event_start_time - raw_start_time) / sampling_rate
        return (
            basecalled_events,
            h5['/Analyses/Basecall_1D_000/BaseCalled_template/Fastq'][()].decode().split('\n')
        )


class RNNAlignedRaw(AlignedRawAbstract):
    def get_basecalled_data(self, h5, target_read, sampling_rate):
        try:
            basecalled_events = h5['/Analyses/Basecall_RNN_1D_000/BaseCalled_template/Events']
            return (
                np.array(basecalled_events.value[['start', 'length', 'model_state', 'move']]),
                h5['/Analyses/Basecall_RNN_1D_000/BaseCalled_template/Fastq'][()].decode().split('\n')
            )
        except:
            raise MissingRNN1DBasecall("Not found RNN component")


class AlbacoreAlignedRaw(AlignedRawAbstract):
    def get_basecalled_data(self, h5, target_read, sampling_rate):
        assert "Albacore" in h5['Analyses/Basecall_1D_000'].attrs['name']
        basecalled_events = h5['/Analyses/Basecall_1D_000/BaseCalled_template/Events']
        adj = h5['Raw/Reads/%s' % target_read].attrs['start_time'] / sampling_rate
        basecalled_events = np.array(basecalled_events.value[['start', 'length', 'model_state', 'move']])
        basecalled_events['start'] -= adj
        return (
            basecalled_events,
            h5['/Analyses/Basecall_1D_000/BaseCalled_template/Fastq'][()].decode().split('\n')
        )


class MinCallAlignedRaw(InputReader):
    def input_fn(self, *args, **kwargs):
        return AlignedRawAbstract.input_fn(self, *args, **kwargs)

    def preprocessSignal(self, signal):
        return (signal - 646.11133) / 75.673653

    def get_signal(self, fast5_path):
        with h5py.File(fast5_path, 'r') as h5:
            reads = h5['Raw/Reads']
            target_read = list(reads.keys())[0]
            h5_logits = h5['Analyses/MinCall/Logits']
            shrink_factor = h5_logits.attrs.get("shrink_factor", 8)  # Default 8..TODO
            signal_len = h5_logits.shape[0] * shrink_factor
            signal = h5['Raw/Reads/' + target_read]['Signal']
            start_pad = h5_logits.attrs['start_pad']
            signal = signal[start_pad:start_pad + signal_len]

        return self.preprocessSignal(signal).reshape(1, -1, 1), start_pad

    def read_fast5_raw_ref(self, fast5_path, ref_path, block_size_x, block_size_y, num_blocks, n_samples_per_ref=1,
                           verify_file=True):

        with h5py.File(fast5_path, 'r') as h5:
            reads = h5['Raw/Reads']
            target_read = list(reads.keys())[0]
            try:
                h5_logits = h5['Analyses/MinCall/Logits']
            except:
                raise MissingMincallLogits()

            try:
                h5_refalignment = h5['Analyses/MinCall/RefAlignment']
            except:
                raise MissingMincallAlignedRef()

            shrink_factor = h5_logits.attrs.get("shrink_factor", 8)  # Default 8..TODO

            signal = h5['Raw/Reads/' + target_read]['Signal']
            start_pad = h5_logits.attrs['start_pad']
            signal_len = h5_logits.shape[0] * shrink_factor
            signal = signal[start_pad:start_pad + signal_len]
            signal = self.preprocessSignal(signal)

            # -2 for first and last that are always skipped
            num_blocks_max = signal_len // block_size_x - 2
            if num_blocks_max < num_blocks:
                raise InsufficientDataBlocks("Has only {} blocks, while requesting {} + first and last"
                                             .format(num_blocks_max, num_blocks))

            n_different_blocks = num_blocks_max - num_blocks + 1
            n_different_non_overlap_blocks = n_different_blocks // num_blocks
            n_samples = min(n_samples_per_ref, n_different_non_overlap_blocks)

            for _ in range(n_samples):
                x = np.zeros([block_size_x * num_blocks, 1], dtype=np.float32)
                x_len = min(len(signal), block_size_x * num_blocks)

                # start from i-th block
                # does not guarantee non overlapping blocks but ok enough
                start_block = randint(1, n_different_blocks)
                x_offset = start_block * block_size_x

                x[:x_len, 0] = signal[x_offset:x_offset + x_len]

                y = np.full([block_size_y * num_blocks], 9, dtype=np.uint8)
                y_len = np.zeros([num_blocks], dtype=np.int32)

                for i in range(num_blocks):
                    prev = -1
                    mult = block_size_x // shrink_factor
                    for b in range(mult * (start_block + i), mult * (start_block + i + 1)):
                        if h5_refalignment[b] != prev:
                            y[i * block_size_y + y_len[i]] = h5_refalignment[b]
                            y_len[i] += 1
                            prev = h5_refalignment[b]
                            if y_len[i] > block_size_y:
                                raise BlockSizeYTooSmall()

                yield x, x_len, y, y_len

    in_dim = 1


def sanitize_input_line(fname):
    fname = fname.strip().split()[0]
    return fname


def proc_wrapper(q, fun, *args):
    for feed in fun(*args):
        q.put(feed)

if __name__=='__main__':
    import pprint
    pp = pprint.PrettyPrinter()
    b = a.read_fast5('/home/lpp/Downloads/minion/nanopore2_20170301_FNFAF09967_MN17024_sequencing_run_170301_MG1655_PC_RAD002_62645_ch7_read372_strand.fast5')
    a = ()

    pp.pprint(b)
