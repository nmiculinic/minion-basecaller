import numpy as np
import multiprocessing
from threading import Thread
import os


def get_feed_yield(keys, batch_size):
    ds = np.load(os.path.expanduser('~/dataset.npz'))
    while True:
        sum([i for i in range(int(1e7))])
        feed = {
            name + "_enqueue_val": ds[name] for name in keys
        }
        feed['X_len_enqueue_val'] = np.array([500] * batch_size)
        yield feed


def proc_wrapper(q, fun, *args):
    for feed in fun(*args):
        q.put(feed)
