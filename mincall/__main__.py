import argparse
from tqdm import tqdm
import logging
from mincall import train
import os


class TqdmWriteWrapper():
    def write(self, s):
        tqdm.write(s, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("mincall")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--logdir", help="Directory for all the logs")
    subparsers = parser.add_subparsers()

    train.add_args(subparsers.add_parser("train"))

    args = parser.parse_args()
    if hasattr(args, 'func'):
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        h = (logging.StreamHandler(TqdmWriteWrapper()))
        if args.verbose:
            h.setLevel(logging.DEBUG)
        else:
            h.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)5s]:%(name)20s: %(message)s")
        h.setFormatter(formatter)
        root_logger.addHandler(h)

        if args.logdir:
            fn = os.path.join(args.logdir,f"{getattr(args, 'name', 'mincall')}.log")
            h = (logging.FileHandler(fn))
            h.setLevel(logging.DEBUG)
            h.setFormatter(formatter)
            root_logger.addHandler(h)
            logging.info(f"Added handler to {fn}")
        logging.info("Initialized logging handlers")
        args.func(args)
    else:
        parser.print_help()
