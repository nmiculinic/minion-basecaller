import h5py
from mincall import input_readers
from mincall import errors
import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import sys
import argparse


def fast5_fix(fast5_path, ax=None):
    with h5py.File(fast5_path, 'r') as h5:
        reads = h5['Analyses/EventDetection_000/Reads']
        target_read = list(reads.keys())[0]
        sampling_rate = h5['UniqueGlobalKey/channel_id'].attrs['sampling_rate']
        signal = h5['Raw/Reads/' + target_read]['Signal']
        raw_start_time = h5['Raw/Reads/' + target_read].attrs['start_time']

        basecalled_hmm, _ = input_readers.HMMAlignedRaw.get_basecalled_data({}, h5, target_read, sampling_rate)

        hmm_times = (basecalled_hmm[0]['start'], basecalled_hmm[-1]['start'] + basecalled_hmm[-1]['length'])

        ax.plot(signal)
        ax.axvline(sampling_rate * hmm_times[0], color='g')
        ax.axvline(sampling_rate * hmm_times[1], 0.1, 0.9, color='g')
        hmm_patch = mpatches.Patch(color='g', label='HMM events')

        events = h5['/Analyses/EventDetection_000/Reads/' + target_read]
        events_times_unscaled = \
            events['Events']['start'][0] - raw_start_time,\
            events['Events']['length'][-1] + events['Events']['start'][-1] - raw_start_time

        ax.axvline(events_times_unscaled[0], color='c')
        ax.axvline(events_times_unscaled[1], 0.1, 0.9, color='c')
        events_patch = mpatches.Patch(color='c', label='events')

        try:
            basecalled_rnn, _ = input_readers.RNNAlignedRaw.get_basecalled_data({}, h5, target_read, sampling_rate)

            rnn_times = (basecalled_rnn[0]['start'], basecalled_rnn[-1]['start'] + basecalled_rnn[-1]['length'])

            ax.axvline(sampling_rate * rnn_times[0], color='r', alpha=0.5)
            ax.axvline(sampling_rate * rnn_times[1], 0.1, 0.9, color='r', alpha=0.5)
            rnn_patch = mpatches.Patch(color='r', label='RNN events')
            ax.legend(handles=[hmm_patch, rnn_patch, events_patch])
        except errors.MissingRNN1DBasecall:
            ax.legend(handles=[hmm_patch, events_patch])

        ax.set_title(fast5_path.split(os.sep)[-1])
        return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input fast5 file", type=str)
    args = parser.parse_args()

    fig, ax = plt.subplots()
    fast5_fix(args.input_file, ax)
    fig.set_size_inches(8, 12)
    fig.savefig(os.path.splitext(args.input_file)[0] + ".png")
