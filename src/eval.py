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
    parser.add_argument("--fasta_out", "-o", type=str, default=None, help='Directory for output fasta files from processed fast5 files')
    parser.add_argument("--ref", type=str, default=None, help='Path to reference string')

    args = parser.parse_args()

    model = load_model(args.module_name, os.path.abspath(args.model_dir))
    try:
        model.init_session(start_queues=False)
        model.restore(checkpoint=args.checkpoint)
        count = args.count
        if count == -1:
            count = 1.0

        if args.fasta_out is None and args.ref is not None:
            args.fasta_out = os.path.join(model.log_dir, 'fasta')
            os.makedirs(args.fasta_out, exist_ok=True)
        model.run_validation_full(frac=count, verbose=args.verbose, fasta_out_dir=args.fasta_out, ref=args.ref)
    finally:
        model.close_session()


if __name__ == "__main__":
    eval_model()
