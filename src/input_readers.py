import os
import socket
from random import shuffle
from errors import MissingRNN1DBasecall, InsufficientDataBlocks, MinIONBasecallerException
import numpy as np
import util
import h5py

root_dir_map = {
    'karla': '/hgst8TB/fjurisic/ecoli',
    'protagonist': '/home/lpp/Downloads/minion'
}
root_dir_default = root_dir_map.get(socket.gethostname(), '/data')


class InputReader():
    def input_fn(self):
        raise NotImplemented

    def fn_args(self, model):
        raise NotImplemented


class AlignedRaw(InputReader):

    def preprocessSignal(self, signal):
        return (signal - 646.11133) / 75.673653

    def read_fast5(self, fast5_path):
        with h5py.File(fast5_path, 'r') as h5:
            reads = h5['Analyses/EventDetection_000/Reads']
            target_read = list(reads.keys())[0]
            sampling_rate = h5['UniqueGlobalKey/channel_id'].attrs['sampling_rate']

            try:
                basecalled_events = h5['/Analyses/Basecall_RNN_1D_000/BaseCalled_template/Events']
            except:
                MissingRNN1DBasecall("{} Not found RNN component".format(fast5_path))

            basecalled = np.array(basecalled_events.value[['start', 'length', 'model_state', 'move']])

            signal = h5['Raw/Reads/' + target_read]['Signal']
            start_time = basecalled[0]['start']
            start_pad = int(sampling_rate * basecalled[0]['start'])
            signal_len = int(sampling_rate * (basecalled[-1]['start'] + basecalled[-1]['length'] - basecalled[0]['start']))

            np.testing.assert_allclose(len(signal), start_pad + signal_len, rtol=1e-2)  # Within 1% relative tolerance

            basecalled['start'] -= start_time
            signal = signal[start_pad:start_pad + signal_len]
            signal = self.preprocessSignal(signal)
            fastq = h5['/Analyses/Basecall_RNN_1D_000/BaseCalled_template/Fastq'][()].decode().split('\n')

            return {
                'signal': signal,
                'basecalled': basecalled,
                'sampling_rate': sampling_rate,
                'fastq': fastq
            }

    def read_fast5_raw_ref(self, fast5_path, ref_path, block_size_x, block_size_y, num_blocks, verify_file=True):
        # num_blocks += 1
        ref_ext = os.path.splitext(ref_path)[1]
        with open(ref_path, 'r') as ref_file:
            fast5 = self.read_fast5(fast5_path)

            signal = fast5['signal']
            basecalled = fast5['basecalled']
            sampling_rate = fast5['sampling_rate']
            fastq = fast5['fastq']

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

            if ref_ext == ".ref":
                ref_seq = ref_file.readlines()[3].strip()
            elif ref_ext == ".fasta":
                ref_seq = util.read_fasta(ref_file)
            else:
                raise ValueError("extension not recognized %s" % ref_ext)

            # Sanity check, correctly basecalled files
            np.testing.assert_string_equal("".join(bucketed_basecall), fastq[1])
            corrected_basecalled = util.correct_basecalled(bucketed_basecall, ref_seq, nedit_tol=0.5)
            # Another sanity check
            np.testing.assert_string_equal("".join(corrected_basecalled), ref_seq)
            print(np.array(corrected_basecalled))

            x = np.zeros([block_size_x * num_blocks, 1], dtype=np.float32)
            x_len = min(len(signal), block_size_x * num_blocks)

            # Skipping first block
            x[:x_len, 0] = signal[block_size_x:block_size_x + x_len]
            y, y_len = util.prepare_y(corrected_basecalled[1:1 + num_blocks], block_size_y)

        return x, x_len, y, y_len

    def input_fn(self):
        def fn(logger, block_size_x, block_size_y, num_blocks, file_list, batch_size=10,
               root_dir=None):
            def load_f(fast5_path, ref_path):
                return self.read_fast5_raw_ref(fast5_path, ref_path,
                                               block_size_x=block_size_x,
                                               block_size_y=block_size_y,
                                               num_blocks=num_blocks,
                                               )

            return get_feed_yield_abs(logger, load_f, batch_size=batch_size, file_list=file_list, root_dir=root_dir)
        return fn

    def fn_args(self, model, file_list):
        return [model.logger, model.block_size_x, model.block_size_y, model.num_blocks, file_list, 10]

    in_dim = 1

    def get_signal(self, fast5_path):
        signal = self.read_fast5(fast5_path)['signal']
        return signal.reshape(1, -1, 1)


def sanitize_input_line(fname):
    fname = fname.strip().split()[0]
    return fname


def get_feed_yield_abs(logger, feed_fn, batch_size, file_list, root_dir=None, **kwargs):
    if root_dir is None:
        root_dir = root_dir_default
    with open(os.path.join(root_dir, file_list), 'r') as f:
        items = items = list(map(sanitize_input_line, f.readlines()))
    names = ["X", "X_len", "Y", "Y_len"]

    err_short = 0
    total = 0
    other_error = 0

    while True:
        shuffle(items)
        for i in range(0, len(items), batch_size):
            arrs = [[] for _ in range(len(names))]

            for fname in items[i:i + batch_size]:
                fast5_path = os.path.join(root_dir, 'pass', fname + '.fast5')
                ref_path = os.path.join(root_dir, 'ref', fname + '.ref')
                total += 1

                try:
                    sol = feed_fn(fast5_path, ref_path)
                    if np.any(sol[3] == 0):
                        err_short += 1
                        continue

                    np.testing.assert_array_less(0, sol[3], err_msg='y_len must be > 0')

                    for a, b in zip(arrs, sol):
                        a.append(b)
                except MinIONBasecallerException as ex:
                    if not isinstance(ex, util.AligmentError):
                        logger.error('in filename %s \n' % fname, exc_info=True)
                    other_error += 1
                    continue

                yield {
                    name + "_enqueue_val": np.array(arr) for name, arr in zip(names, arrs)
                }

                if total % 10000 == 0 or total in [10, 100, 200, 1000]:
                    logger.info("read %d datapoints. err_short_rate %.3f other %.3f ", total, err_short / total, other_error / total)


def proc_wrapper(q, fun, *args):
    for feed in fun(*args):
        q.put(feed)


if __name__ == '__main__':
    inp = AlignedRaw()
    x, x_len, y, y_len = inp.read_fast5_raw_ref("/home/lpp/Downloads/minion/pass/95274_ch178_read751_strand.fast5", "/home/lpp/Downloads/minion/ref/95274_ch178_read751_strand.ref", 200, 25, 5)

    print(y[:25*5].reshape(5,25))
    print(y_len[:5])
