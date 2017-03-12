import os
from dotenv import load_dotenv, find_dotenv
import model_utils
import json
import argparse
import importlib
load_dotenv(find_dotenv())


def load_model(module_name, model_dir):
    model_module = importlib.import_module(module_name)
    model_dir = os.path.abspath(model_dir)
    with open(os.path.join(model_dir, 'model_hyperparams.json'), 'r') as f:
        hyper = json.load(f)

    params = model_module.model_setup_params(hyper)
    print(params, type(params))
    params['reuse'] = True
    params['overwrite'] = False
    params['log_dir'] = model_dir
    params['run_id'] = model_dir.split('/')[-1]
    return model_utils.Model(**params)


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
