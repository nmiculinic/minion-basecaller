import os
import socket
from random import shuffle

import numpy as np
import util

root_dir_map = {
    'karla': '/hgst8TB/fjurisic/ecoli',
    'protagonist': '/home/lpp/Downloads/minion'
}
root_dir_default = root_dir_map.get(socket.gethostname(), '/data')


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

                if total % 1000 == 0 or total in [10, 100, 200]:
                    logger.info("read %d datapoints. err_short_rate %.3f other %.3f ", total, err_short / total, other_error / total)



def get_event_ref_feed_yield(logger, block_size, num_blocks, file_list, batch_size=10, warn_if_short=False, root_dir=None):
    def load_f(fast5_path, ref_path):
        return util.read_fast5_ref(fast5_path, ref_path, block_size, num_blocks, warn_if_short)

    return get_feed_yield_abs(logger, load_f, batch_size, file_list, root_dir=root_dir)


def get_raw_ref_feed_yield(logger, block_size_x, block_size_y, num_blocks, file_list, batch_size=10, warn_if_short=False,
                       root_dir=None):
    def load_f(fast5_path, ref_path):
        return util.read_fast5_raw_ref(fast5_path, ref_path,
                                   block_size_x=block_size_x,
                                   block_size_y=block_size_y,
                                   num_blocks=num_blocks,
                                   warn_if_short=warn_if_short)

    return get_feed_yield_abs(logger, load_f, batch_size=batch_size, file_list=file_list, root_dir=root_dir)


def get_event_feed_yield(block_size, num_blocks, file_list, batch_size=10, warn_if_short=False, root_dir=None):
    return get_feed_yield_abs(lambda filename: util.read_fast5(filename, block_size, num_blocks, warn_if_short), batch_size, file_list, root_dir=root_dir)


def get_raw_feed_yield(block_size_x, block_size_y, num_blocks, file_list, batch_size=10, warn_if_short=False, root_dir=None):
    return get_feed_yield_abs(lambda filename: util.read_fast5_raw(
        filename,
        block_size_x=block_size_x,
        block_size_y=block_size_y,
        num_blocks=num_blocks,
        warn_if_short=warn_if_short),
        batch_size=batch_size, file_list=file_list, root_dir=root_dir)


def proc_wrapper(q, fun, *args):
    for feed in fun(*args):
        q.put(feed)


if __name__ == "__main__":
    print("usao")
    for x in get_feed_yield2(10, 1):
        print(x)
    print("izasao")
