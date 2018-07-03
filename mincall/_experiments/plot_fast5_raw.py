import seaborn as sns
import matplotlib.pyplot as plt
import h5py
import numpy as np
import argparse


def read_fast5_signal(fname: str) -> np.ndarray:
    with h5py.File(fname, 'r') as input_data:
        raw_attr = input_data['Raw/Reads/']
        read_name = list(raw_attr.keys())[0]
        raw_signal = np.array(raw_attr[read_name + "/Signal"].value)
        return raw_signal

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fn", help="file name")
    parser.add_argument("out", help="save fig out")
    args = parser.parse_args()

    signal = read_fast5_signal(args.fn)
    fig, ax = plt.subplots()
    fig: plt.Figure = fig
    fig.set_size_inches(12, 8)
    ax: plt.Axes = ax
    ax.set_title("Fast5 raw signal")
    ax.set_ylabel("current")
    ax.set_xlabel("sampling step")
    ax.plot(np.arange(len(signal)), signal)
    fig.savefig(args.out)


