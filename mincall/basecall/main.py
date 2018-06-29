import argparse
import yaml
from ._types import *
import sys
from voluptuous.humanize import humanize_error
from ._basecall import run
import logging
import voluptuous
import cytoolz as toolz
from pprint import pformat

logger = logging.getLogger("mincall.basecall")


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--config", "-c", help="config file", required=False)
    parser.add_argument("--in", "-i", nargs="*", dest='basecall.input_dir')
    parser.add_argument("--out", "-o", dest='basecall.output_fasta')
    parser.add_argument(
        "--model", "-m", dest='basecall.model', help="model savepoint"
    )
    parser.add_argument(
        "--batch_size", "-b", dest='basecall.batch_size', type=int
    )
    parser.add_argument(
        "--num-threads", "-t", dest='basecall.threads', type=int
    )
    parser.add_argument(
        "--seq_length",
        "-l",
        dest='basecall.seq_length',
        type=int,
        help="segment length"
    )
    parser.add_argument("--jump", "-j", dest='basecall.jump', type=int)
    parser.add_argument(
        '--beam',
        dest='basecall.beam_width',
        type=int,
        default=50,
        help=
        "Beam width used in beam search decoder, default is 50, set to 0 to use a greedy decoder. Large beam width give better decoding result but require longer decoding time."
    )
    parser.add_argument(
        "--gzip",
        "-z",
        default=None,
        action="store_true",
        dest="basecall.gzip",
        help="gzip the output"
    )
    parser.set_defaults(func=run_args)
    parser.set_defaults(name="mincall_basecall")


def run_args(args):
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.load(f)
    else:
        config = {
            'basecall': {},
            'version': "v0.1",
        }

    for k, v in vars(args).items():
        if v is not None and "." in k:
            config = toolz.assoc_in(config, k.split("."), v)

    if args.logdir is not None:
        config['basecall']['logdir'] = args.logdir
    try:
        cfg = voluptuous.Schema({
            'basecall': BasecallCfg.schema,
            'version': str,
        },
                                extra=voluptuous.REMOVE_EXTRA,
                                required=True)(config)
        logger.info(f"Parsed config\n{pformat(cfg)}")
        run(cfg['basecall'])
    except voluptuous.error.Error as e:
        logger.error(humanize_error(config, e))
        sys.exit(1)
