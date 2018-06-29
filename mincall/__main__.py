import argparse
from tqdm import tqdm
import logging
from mincall import train, basecall, embedding, eval
from mincall.hyperparam import _hyperparam
import graypy
import os


class TqdmWriteWrapper():
    def write(self, s):
        tqdm.write(s, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("mincall")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--logdir", help="Directory for all the logs")
    parser.add_argument("--gelf-udp", help="gelf udp host:port")
    subparsers = parser.add_subparsers()

    train.add_args(subparsers.add_parser("train"))
    basecall.add_args(subparsers.add_parser("basecall"))
    embedding.add_args(subparsers.add_parser("embed"))
    _hyperparam.add_args(subparsers.add_parser("hyperparam"))
    eval.add_args(subparsers.add_parser("eval"))

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
            "%(asctime)s [%(levelname)5s]:%(name)20s: %(message)s"
        )
        h.setFormatter(formatter)
        root_logger.addHandler(h)

        if args.logdir:
            os.makedirs(args.logdir, exist_ok=True)
            fn = os.path.join(
                args.logdir, f"{getattr(args, 'name', 'mincall')}.log"
            )
            h = (logging.FileHandler(fn))
            h.setLevel(logging.DEBUG)
            h.setFormatter(formatter)
            root_logger.addHandler(h)
            logging.info(f"Added handler to {fn}")
        logging.info("Initialized logging handlers")
        if args.gelf_udp:
            host, port = args.gelf_udp.split(":")
            port = int(port)
            handler = graypy.GELFHandler(host, port, extra_fields=True)
            handler.setLevel(logging.DEBUG)
            logging.getLogger().addHandler(handler)
            logging.info(f"Added gelf handler @ {host}:{port}")
        args.func(args)
    else:
        parser.print_help()
