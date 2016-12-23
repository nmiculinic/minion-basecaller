import numpy as np
import os
import util
from random import shuffle
import sys

def get_feed_yield(keys, batch_size):
    ds = np.load(os.path.expanduser('~/dataset.npz'))
    while True:
        feed = {
            name + "_enqueue_val": ds[name] for name in keys
        }
        feed['X_len_enqueue_val'] = np.array([500] * batch_size)
        yield feed


dataset_2_cache = {}


def get_feed_yield2(block_size, num_blocks, batch_size=10):
    root_dir = '/hgst8TB/fjurisic/ecoli/pass'
    items = list(os.listdir(root_dir))
    names = ["X", "X_len", "Y", "Y_len"]
    while True:
        shuffle(items)
        for i in range(0, len(items), batch_size):
            arrs = [[] for _ in range(len(names))]

            for fname in map(lambda x: os.path.join(root_dir, x), items[i:i + batch_size]):
                if fname in dataset_2_cache:
                    sol = dataset_2_cache[fname]
                else:
                    sol = util.read_fast5(fname, block_size, num_blocks)
                    dataset_2_cache[fname] = sol
                if sol is not None:
                    if np.any(sol[3] == 0):
                        # print(fname, "y_len 0, skipping", file=sys.stderr)
                        continue
                    if sol[0].shape != (block_size * num_blocks, 3):
                        print(fname, "unexpected shape, skipping")
                        continue
                    for a, b in zip(arrs, sol):
                        a.append(b)

            yield {
                name + "_enqueue_val": np.array(arr) for name, arr in zip(names, arrs)
            }


def proc_wrapper(q, fun, *args):
    for feed in fun(*args):
        q.put(feed)
