import os
from dotenv import load_dotenv, find_dotenv
import argparse
import sys
import util
import h5py
import subprocess
load_dotenv(find_dotenv())


def basecall_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("fast5_in", help="Fast5 file to basecall or dir of fast5 files", default='.', type=str)
    parser.add_argument("out_dir", type=str, default='.', help='Directory for output fasta files from processed fast5 files')
    parser.add_argument("--ref", type=str, default=None, help='Path to reference string')

    args = parser.parse_args()
    if os.path.isfile(args.fast5_in):
        file_list = [args.fast5_in]
    elif os.path.isdir(args.fast5_in):
        file_list = [os.path.join(args.fast5_in, path) for path in os.listdir(args.fast5_in) if os.path.splitext(path)[1] == ".fast5"]
    else:
        print("Not file not dir %s, exiting!!!" % args.fast5_in)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    for f in file_list:
        out = os.path.splitext(f)[0].split('/')[-1] + ".fasta"
        out = os.path.join(args.out_dir, out)

        with h5py.File(f, 'r') as h5:
            basecalled_events = h5['/Analyses/Basecall_1D_000/BaseCalled_template/Events']
            basecalled = util.get_basecalled_sequence(basecalled_events)

            with open(out, 'w') as fn:
                print("> " + f, file=fn)
                n = 80
                for i in range(0, len(basecalled), n):
                    print(basecalled[i:i+n], file=fn)

            if args.ref is not None:
                sam_out = os.path.splitext(out)[0] + ".sam"
                print("Sam out %s" % sam_out)
                subprocess.Popen(["graphmap", "align", "-r", args.ref, "-d", out, "-o", sam_out, "-v", "0", "--extcigar"])


if __name__ == "__main__":
    basecall_model()
