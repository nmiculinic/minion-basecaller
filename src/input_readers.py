import os
import socket
from random import shuffle

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

    @staticmethod
    def preprocessSignal(signal):
        return (signal - 646.11133) / 75.673653

    def read_fast5_raw_ref(self, fast5_path, ref_path, block_size_x, block_size_y, num_blocks, warn_if_short=False):
        num_blocks += 1
        with h5py.File(fast5_path, 'r') as h5, open(ref_path, 'r') as ref_file:
            reads = h5['Analyses/EventDetection_000/Reads']
            target_read = list(reads.keys())[0]
            events = np.array(reads[target_read + '/Events'])
            start_time = events['start'][0]
            start_pad = int(start_time - h5['Raw/Reads/' + target_read].attrs['start_time'])

            basecalled_events = h5['/Analyses/Basecall_1D_000/BaseCalled_template/Events']
            basecalled = np.array(basecalled_events.value[['mean', 'stdv', 'model_state', 'move', 'start', 'length']])

            signal = h5['Raw/Reads/' + target_read]['Signal']
            signal_len = h5['Raw/Reads/' + target_read].attrs['duration'] - start_pad

            x = np.zeros([block_size_x * num_blocks, 1], dtype=np.float32)
            x_len = min(signal_len, block_size_x * num_blocks)
            x[:x_len, 0] = signal[start_pad:start_pad + x_len]

            np.testing.assert_allclose(len(signal), start_pad + np.sum(events['length']))

            events_len = np.zeros([num_blocks], dtype=np.int32)

            bcall_idx = 0
            prev, curr_sec = "N", 0
            for e in events:
                if (e['start'] - start_time) // block_size_x > curr_sec:
                    prev, curr_sec = "N", (e['start'] - start_time) // block_size_x
                if curr_sec >= num_blocks:
                    break

                if bcall_idx < basecalled.shape[0]:
                    b = basecalled[bcall_idx]

                    if b[0] == e[2] and b[1] == e[3]:  # mean == mean and stdv == stdv
                        added_bases = 0

                        if bcall_idx == 0:
                            added_bases = 5
                            assert len(list(b[2].decode("ASCII"))) == 5

                        bcall_idx += 1
                        assert 0 <= b[3] <= 2
                        added_bases += b[3]
                        events_len[curr_sec] += added_bases

            ref_seq = ref_file.readlines()[3].strip()
            called_seq = util.get_basecalled_sequence(basecalled_events)
            y, y_len = util.extract_blocks(ref_seq, called_seq, events_len, block_size_y, num_blocks)

            y_len = y_len[1:]
            y = y[block_size_y:]
            if any(y_len > block_size_y):
                raise util.AligmentError()

        x = self.preprocessSignal(x)
        return x[block_size_x:], max(0, x_len - block_size_x), y, y_len

    def input_fn(self):
        def fn(logger, block_size_x, block_size_y, num_blocks, file_list, batch_size=10, warn_if_short=False,
               root_dir=None):
            def load_f(fast5_path, ref_path):
                return self.read_fast5_raw_ref(fast5_path, ref_path,
                                               block_size_x=block_size_x,
                                               block_size_y=block_size_y,
                                               num_blocks=num_blocks,
                                               warn_if_short=warn_if_short)

            return get_feed_yield_abs(logger, load_f, batch_size=batch_size, file_list=file_list, root_dir=root_dir)
        return fn

    def fn_args(self, model, file_list):
        return [model.logger, model.block_size_x, model.block_size_y, model.num_blocks, file_list, 10]

    in_dim = 1

    def get_signal(self, fast5_path):
        signal = util.get_raw_signal(fast5_path)
        signal = self.preprocessSignal(signal)
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
                except Exception as ex:
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
