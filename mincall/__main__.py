import argparse
from tqdm import tqdm
import logging
from mincall import train

class TqdmWriteWrapper():
    def write(self, s):
        tqdm.write(s, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("mincall")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--log-file", help="Where to store log file")
    subparsers = parser.add_subparsers()

    train.add_args(subparsers.add_parser("train"))

    args = parser.parse_args()
    if hasattr(args, 'func'):
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        h = (logging.StreamHandler(
            TqdmWriteWrapper()
        ))
        if args.verbose:
            h.setLevel(logging.DEBUG)
        else:
            h.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)5s]:%(name)20s: %(message)s"
        )
        h.setFormatter(formatter)
        root_logger.addHandler(h)

        if args.log_file:
            h = (logging.FileHandler(
                args.log_file,
            ))
            h.setLevel(logging.DEBUG)
            h.setFormatter(formatter)
            root_logger.addHandler(h)
        logging.info("Initialized logging handlers")
        args.func(args)
    else:
        parser.print_help()
