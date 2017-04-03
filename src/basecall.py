import os
from dotenv import load_dotenv, find_dotenv
import argparse
import eval
import sys
load_dotenv(find_dotenv())


def basecall_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("module_name", help="module name", type=str)
    parser.add_argument("model_dir", help="increase output verbosity", type=str)
    parser.add_argument("fast5_in", help="Fast5 file to basecall or dir of fast5 files", default='.', type=str)
    parser.add_argument("out_dir", type=str, default='.', help='Directory for output fasta files from processed fast5 files')
    parser.add_argument("-c", "--checkpoint", help="Checkpoint to restore", type=str, default=None)

    args = parser.parse_args()
    if os.path.isfile(args.fast5_in):
        file_list = [args.fast5_in]
    elif os.path.isdir(args.fast5_in):
        file_list = [os.path.join(args.fast5_in, path) for path in os.listdir(args.fast5_in) if os.path.splitext(path)[1] == ".fast5"]
    else:
        print("Not file not dir %s, exiting!!!" % args.fast5_in)
        sys.exit(1)

    model = eval.load_model(args.module_name, os.path.abspath(args.model_dir))

    try:
        model.init_session(start_queues=False)
        model.restore(checkpoint=args.checkpoint)
        os.makedirs(args.out_dir, exist_ok=True)
        for f in file_list:
            out = os.path.splitext(f)[0].split('/')[-1] + ".fasta"
            out = os.path.join(args.out_dir, out)
            model.basecall_sample(f, fasta_out=out)
    finally:
        model.close_session()


if __name__ == "__main__":
    basecall_model()
