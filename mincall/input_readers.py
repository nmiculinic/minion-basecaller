import os
import socket
from random import shuffle
import numpy as np
import h5py
from collections import defaultdict

from . import util
from . import errors
from .errors import MissingRNN1DBasecall, InsufficientDataBlocks, MinIONBasecallerException

root_dir_map = {
    'karla': '/hgst8TB/fjurisic/ecoli',
    'protagonist': '/home/lpp/Downloads/minion'
}
root_dir_default = root_dir_map.get(socket.gethostname(), '/data')


class InputReader():
    def input_fn(self):
        raise NotImplemented


class AlignedRaw(InputReader):

    def preprocessSignal(self, signal):
        return (signal - 646.11133) / 75.673653

    def getRNNBasecalled(self, h5, **kwargs):
        try:
            basecalled_events = h5['/Analyses/Basecall_RNN_1D_000/BaseCalled_template/Events']
            return (
                np.array(basecalled_events.value[['start', 'length', 'model_state', 'move']]),
                h5['/Analyses/Basecall_RNN_1D_000/BaseCalled_template/Fastq'][()].decode().split('\n')
            )
        except:
            raise MissingRNN1DBasecall("Not found RNN component")

    def getHMMBasecalled(self, h5, target_read, sampling_rate):
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

    def read_fast5(self, fast5_path):
        with h5py.File(fast5_path, 'r') as h5:
            reads = h5['Analyses/EventDetection_000/Reads']
            target_read = list(reads.keys())[0]
            sampling_rate = h5['UniqueGlobalKey/channel_id'].attrs['sampling_rate']

            basecalled, fastq = self.getHMMBasecalled(h5, target_read, sampling_rate)
            # basecalled = self.getRNNBasecalled(h5, target_read, sampling_rate)

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
                'fastq': fastq
            }

    def read_fast5_raw_ref(self, fast5_path, ref_path, block_size_x, block_size_y, num_blocks, verify_file=True):
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

            if len(bucketed_basecall) < num_blocks + 2:
                raise InsufficientDataBlocks("Has only {} blocks, while requesting {} + first and last".format(len(bucketed_basecall), num_blocks))

            ref_ext = os.path.splitext(ref_path)[1]
            if ref_ext == ".ref":
                ref_seq = ref_file.readlines()[3].strip()
            elif ref_ext == ".fasta":
                ref_seq = util.read_fasta(ref_file)
            else:
                raise ValueError("extension not recognized %s" % ref_ext)

            corrected_basecalled = util.correct_basecalled(bucketed_basecall, ref_seq, nedit_tol=0.35)

            if verify_file:
                # Sanity check, correctly basecalled files
                np.testing.assert_string_equal("".join(bucketed_basecall), fastq[1])
                np.testing.assert_string_equal("".join(corrected_basecalled), ref_seq)

            x = np.zeros([block_size_x * num_blocks, 1], dtype=np.float32)
            x_len = min(len(signal), block_size_x * num_blocks)

            # Skipping first block
            x[:x_len, 0] = signal[block_size_x:block_size_x + x_len]
            y, y_len = util.prepare_y(corrected_basecalled[1:1 + num_blocks], block_size_y)
            return x, x_len, y, y_len

    def find_ref(self, fast5, dirs):
        fname = os.path.splitext(fast5)[0].split(os.sep)[-1]
        for d in dirs:
            pick = os.path.join(d, fname + ".ref")
            if os.path.exists(pick):
                return pick

        raise errors.RefFileNotFound("{}-> search query:{}.ref in following directories {}".format(fast5, fname, dirs))

    def input_fn(self, model, file_list, root_dir=None):
        if root_dir is None:
            root_dir = root_dir_default
        with open(os.path.join(root_dir, file_list), 'r') as f:
            items = list(map(sanitize_input_line, f.readlines()))
        names = ["X", "X_len", "Y", "Y_len"]

        total = 0
        errors = defaultdict(int)

        while True:
            shuffle(items)
            for fname in items:
                fast5_path = os.path.join(root_dir, 'pass', fname + '.fast5')
                ref_path = self.find_ref(fast5_path, [os.path.join(root_dir, 'ref')])
                total += 1

                try:
                    sol = self.read_fast5_raw_ref(
                        fast5_path,
                        ref_path,
                        block_size_x=model.block_size_x,
                        block_size_y=model.block_size_y,
                        num_blocks=model.num_blocks,
                        verify_file=False
                    )
                    np.testing.assert_array_less(0, sol[3], err_msg='y_len must be > 0')

                    yield {
                        name + "_enqueue_val": np.array([arr]) for name, arr in zip(names, sol)
                    }
                except KeyboardInterrupt:
                    raise
                except MinIONBasecallerException as ex:
                    errors[type(ex).__name__] += 1
                except Exception as ex:
                    errors[type(ex).__name__] += 1
                    model.logger.error('in filename %s \n' % fname, exc_info=True)

                if total % 10000 == 0 or total in [10, 100, 200, 1000]:
                    model.logger.debug(
                        "read %d datapoints: %s ",
                        total,
                        " ".join(["{}: {:.2f}%".format(key, 100 * errors[key] / total) for key in sorted(errors.keys())])
                    )

    in_dim = 1

    def get_signal(self, fast5_path):
        signal = self.read_fast5(fast5_path)['signal']
        return signal.reshape(1, -1, 1)


def sanitize_input_line(fname):
    fname = fname.strip().split()[0]
    return fname


def proc_wrapper(q, fun, *args):
    for feed in fun(*args):
        q.put(feed)


if __name__ == '__main__':
    inp = AlignedRaw()
    x, x_len, y, y_len = inp.read_fast5_raw_ref("/home/lpp/Downloads/minion/pass/58342_ch183_read295_strand.fast5", "/home/lpp/Downloads/minion/ref/58342_ch183_read295_strand.ref", 8 * 3 * 600 // 2, 630, 3, verify_file=True)
