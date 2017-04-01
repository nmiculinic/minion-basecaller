import os
from dotenv import load_dotenv, find_dotenv
import model_utils
import argparse
load_dotenv(find_dotenv())


def load_model(module_name, model_dir):
    return model_utils.Model(**model_utils.load_model_parms(module_name, model_dir))


def eval_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("module_name", help="module name", type=str)
    parser.add_argument("model_dir", help="increase output verbosity", type=str)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-c", "--checkpoint", help="Checkpoint to restore", type=str, default=None)
    parser.add_argument("count", nargs='?', type=int, default=-1, help='Number of evaluation count from test set. Default -1 meaning whole test set')

    args = parser.parse_args()

    model = load_model(args.module_name, os.path.abspath(args.model_dir))
    try:
        model.init_session(start_queues=False)
        model.restore(checkpoint=args.checkpoint)
        count = args.count
        if count == -1:
            count = 1.0
        model.run_validation_full(frac=count, verbose=args.verbose)
    finally:
        model.close_session()


if __name__ == "__main__":
    eval_model()
