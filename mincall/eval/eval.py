import argparse
import logging
from typing import *
import voluptuous
import os

import pysam
import matplotlib.pyplot as plt
import seaborn as sns

from mincall.eval.align_utils import filter_aligments_in_sam, read_len_filter, secondary_aligments_filter, only_mapped_filter, supplementary_aligments_filter
from mincall.bioinf_utils import error_rates_for_sam
from mincall.bioinf_utils import error_positions_report
from .consensus import get_consensus_report

logger = logging.getLogger(__name__)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("sam_file")
    parser.add_argument("-r", "--reference", type=voluptuous.IsFile(), required=True)
    parser.add_argument("-w", "--work-dir", default=".", type=voluptuous.IsDir())
    parser.set_defaults(func=run_args)


def run_args(args):
    print(args)
    sam_path = args.sam_file
    filtered_path = os.path.join(args.work_dir, "filtered.sam")

    ### Filtering
    # define list of filters (functions that take pysam.AlignedSegment and return boolean)
    filters: List[Callable[[pysam.AlignedSegment]], bool] = [
        #read_len_filter(max_len=400),# read_len_filter(max_len=400),
        only_mapped_filter(), #secondary_aligments_filter(), supplementary_aligments_filter(),
    ]

    n_kept, n_discarded = filter_aligments_in_sam(sam_path, filtered_path, filters)
    logger.info(f"Kept {n_kept}, discarded {n_discarded}")

    ### Analize error rates

    df = error_rates_for_sam(sam_path)
    logger.info(f"Error rates:\n{df.describe(percentiles=[]).transpose()}")

    position_report = error_positions_report(sam_path)
    logger.info(f"Error position report\n{position_report.head(20)}")

    fig = plot_error_distributions(position_report)
    fig.savefig(os.path.join(args.work_dir, "position_report.png"))


def plot_error_distributions(position_report) -> plt.Figure:
    n_cols = 2
    n_rows = 2

    label = {
        '=': 'Match',
        'X': 'Missmatch',
        'I': "Insertion",
        'D': 'Deletion'
    }
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16,10))
    for ax, op in zip(axes.ravel(), ['=', 'X', 'I', 'D']):
        ax: plt.Axes = ax
        data = position_report[position_report.operation == op]['relative_position']
        sns.distplot(data, ax=ax, label=op)
        # sns.kdeplot(data.relative_position, weights=data.op_count, shade=False, label=op, alpha=0.5, ax=ax)
        ax.set_xlim(0, 1)
        ax.set_xlabel('relative position')
        ax.set_ylabel('percentage %')
        ax.set_title(label[op])
    return fig
