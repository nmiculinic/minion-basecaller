import numpy as np
import os
import util
from random import shuffle
import sys
import socket

dataset_2_cache = {}


def get_feed_yield2(block_size, num_blocks, batch_size=10, root_dir=None):
    if root_dir is None:
        if socket.gethostname() == "karla":
            root_dir = '/hgst8TB/fjurisic/ecoli/pass'
        elif socket.gethostname() == "protagonist":
            root_dir = '/home/lpp/Downloads/minion/pass'
        else:
            raise ValueError("Root dir cannot be infered")
    items = list(os.listdir(root_dir))
    names = ["X", "X_len", "Y", "Y_len"]
    while True:
        shuffle(items)
        for i in range(0, len(items), batch_size):
            arrs = [[] for _ in range(len(names))]

            for fname in map(lambda x: os.path.join(root_dir, x), items[i:i + batch_size]):
                try:
                    if fname in dataset_2_cache:
                        sol = dataset_2_cache[fname]
                    else:
                        sol = util.read_fast5(fname, block_size, num_blocks)
                        dataset_2_cache[fname] = sol
                    if sol is not None:
                        if np.any(sol[3] == 0):
                            print(fname, "y_len 0, skipping", file=sys.stderr)
                            continue
                        if sol[0].shape != (block_size * num_blocks, 3):
                            print(fname, "unexpected shape, skipping", file=sys.stderr)
                            continue
                        for a, b in zip(arrs, sol):
                            a.append(b)
                except Exception as ex:
                    print(ex, file=sys.stderr)
                    continue

            yield {
                name + "_enqueue_val": np.array(arr) for name, arr in zip(names, arrs)
            }


def proc_wrapper(q, fun, *args):
    for feed in fun(*args):
        q.put(feed)
