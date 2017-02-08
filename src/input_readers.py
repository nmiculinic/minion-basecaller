import numpy as np
import os
import util
from random import shuffle
import sys
import socket
import traceback as tb

root_dir_map = {
    'karla': '/hgst8TB/fjurisic/ecoli',
    'protagonist': '/home/lpp/Downloads/minion'
}


def sanitize_input_line(fname):
    fname = fname.strip().split()[0]
    if fname[-6:] != '.fast5':
        fname = fname + ".fast5"
    return fname


def get_feed_yield_abs(feed_fn, batch_size, file_list, root_dir=None, **kwargs):
    if root_dir is None:
        root_dir = root_dir_map.get(socket.gethostname(), '/data')
    with open(os.path.join(root_dir, file_list), 'r') as f:
        items = list(map(sanitize_input_line, f.readlines()))
    names = ["X", "X_len", "Y", "Y_len"]
    while True:
        shuffle(items)
        for i in range(0, len(items), batch_size):
            arrs = [[] for _ in range(len(names))]
            for fname in map(lambda x: os.path.join(root_dir, 'pass', x), items[i:i + batch_size]):
                    try:
                        sol = feed_fn(fname)
                        if sol is not None:
                            if np.any(sol[3] == 0):
                                # print(fname, "y_len 0, skipping", file=sys.stderr)
                                continue
                            for a, b in zip(arrs, sol):
                                a.append(b)
                    except Exception as ex:
                        print('\r=== ERROR ===\n', __file__, ex, file=sys.stderr)
                        tb.print_exc()
                        print('=== END ERROR ===')
                        continue

                    yield {
                        name + "_enqueue_val": np.array(arr) for name, arr in zip(names, arrs)
                    }


def get_feed_yield2(block_size, num_blocks, file_list, batch_size=10, warn_if_short=False, root_dir=None):
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
