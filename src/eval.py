import tensorflow as tf
import os
from dotenv import load_dotenv, find_dotenv
import model_utils
import dill
import argparse
import deep_residual_gated
load_dotenv(find_dotenv())


def load_model(model_dir):
    model_dir = os.path.abspath(model_dir)
    with open(os.path.join(model_dir, 'model_params.pickle'), 'rb') as f:
        params = dill.load(f)

    print(params, type(params))
    params['reuse'] = True
    params['overwrite'] = False
    params['log_dir'] = os.path.abspath('/' + os.path.join(*model_dir.split('/')[:-1]))
    params['run_id'] = model_dir.split('/')[-1]
    params['model_fn'] = deep_residual_gated.model_fn
    return model_utils.Model(g=tf.Graph(), **params)


def eval_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="increase output verbosity", type=str)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-c", "--checkpoint", help="Checkpoint to restore", type=str, default=None)
    parser.add_argument("count", nargs='?', type=int, default=-1, help='Number of evaluation count from test set. Default -1 meaning whole test set')

    args = parser.parse_args()

    model = load_model(os.path.abspath(args.model_dir))
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
