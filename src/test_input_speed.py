import input_readers
import logging
from time import perf_counter
import multiprocessing
import argparse


def f():
    log_fmt = '\r[%(levelname)s] %(name)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)
    logger = logging.getLogger("inputTest")
    t0 = perf_counter()
    for t, _ in enumerate(input_readers.get_raw_ref_feed_yield(logger, 8 * 600 * 3 // 2, 630, 1, 'train.txt')):
        it = t + 1
        if it in [10, 20, 30, 40] or it % 50 == 0:
            logger.info("ucitao %d primjera u %.3f it/s", it, it / (perf_counter() - t0))


parser = argparse.ArgumentParser()
parser.add_argument("num", help="module name", default=3, type=int)
args = parser.parse_args()

print("running %d evals" % args.num)
for _ in range(args.num):
    p = multiprocessing.Process(target=f)
    p.start()
