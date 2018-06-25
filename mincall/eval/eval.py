import argparse
import logging
from typing import *
import voluptuous
from voluptuous.humanize import humanize_error
import os

import pysam
import matplotlib.pyplot as plt
import seaborn as sns
import cytoolz as toolz
import pandas as pd
import numpy as np

from mincall.eval.align_utils import filter_aligments_in_sam, read_len_filter, secondary_aligments_filter, only_mapped_filter, supplementary_aligments_filter
from mincall.bioinf_utils import error_rates_for_sam
from mincall.bioinf_utils import error_positions_report
from .consensus import get_consensus_report
from mincall.common import named_tuple_helper

logger = logging.getLogger(__name__)


class EvalCfg(NamedTuple):
    sam_path: str
    work_dir: str
    reference: str
    is_circular:bool = False
    coverage_threshold: int = 0

    @classmethod
    def schema(cls, data):
        return named_tuple_helper(
            cls, {}, data
        )


def run_args(args):
    config = {}
    for k, v in vars(args).items():
        if v is not None and "." in k:
            config = toolz.assoc_in(config, k.split("."), v)
            print(k, v)
    try:
        cfg = voluptuous.Schema({"eval": EvalCfg.schema},
                extra=voluptuous.REMOVE_EXTRA,
              required=True)(config)
        run(cfg['eval'])
    except voluptuous.error.Error as e:
        logger.error(humanize_error(config, e))
        raise

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("eval.sam_path")
    parser.add_argument("-r", "--reference", dest="eval.reference", type=voluptuous.IsFile(), required=True)
    parser.add_argument("-w", "--work-dir", dest="eval.work_dir", type=voluptuous.IsDir())
    parser.set_defaults(func=run_args)

def run(cfg: EvalCfg):
    print(cfg)
    filtered_path = os.path.join(cfg.work_dir, "filtered.sam")

    ### Filtering
    # define list of filters (functions that take pysam.AlignedSegment and return boolean)
    filters: List[Callable[[pysam.AlignedSegment]], bool] = [
        #read_len_filter(max_len=400),# read_len_filter(max_len=400),
        only_mapped_filter(), #secondary_aligments_filter(), supplementary_aligments_filter(),
    ]

    n_kept, n_discarded = filter_aligments_in_sam(cfg.sam_path, filtered_path, filters)
    logger.info(f"Kept {n_kept}, discarded {n_discarded}")

    ### Analize error rates

    df = error_rates_for_sam(cfg.sam_path)
    export_dataframe(df.describe(percentiles=[]).transpose(), cfg.work_dir, "error rates")

    position_report = error_positions_report(cfg.sam_path)
    logger.info(f"Error position report\n{position_report.head(20)}")

    fig = plot_error_distributions(position_report)
    fig.savefig(os.path.join(cfg.work_dir, "position_report.png"))

    report = get_consensus_report('mincall', cfg.sam_path, cfg.reference, cfg.is_circular, cfg.coverage_threshold)
    export_dataframe(report.transpose(), cfg.work_dir, "consensus_report")


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

def export_dataframe(df: pd.DataFrame, workdir: str, name: str):
    logger.info(f"{name}:\n{df}")
    df.to_latex(os.path.join(workdir, f"{name}.tex"))
    with open(os.path.join(workdir, f"{name}.txt"), "w") as f:
        df.to_string(f)
    with open(os.path.join(workdir, f"{name}.csv"), "w") as f:
        df.to_csv(f)
    df.to_pickle(os.path.join(workdir, f"{name}.pickle"))
