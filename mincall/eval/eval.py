import argparse
import logging
import voluptuous
import os
from collections import defaultdict
from typing import *
from voluptuous.humanize import humanize_error

import matplotlib
matplotlib.use('agg')

import pysam
import matplotlib.pyplot as plt
import seaborn as sns
import cytoolz as toolz
import pandas as pd

from mincall.eval.align_utils import filter_aligments_in_sam, read_len_filter, secondary_aligments_filter, only_mapped_filter, supplementary_aligments_filter
from mincall.bioinf_utils import error_rates_for_sam
from mincall.bioinf_utils import error_positions_report
from .consensus import get_consensus_report
from mincall.common import named_tuple_helper
logger = logging.getLogger(__name__)


class EvalCfg(NamedTuple):
    sam_path: List[str]
    work_dir: str
    reference: str
    is_circular: bool = False
    coverage_threshold: int = 0

    @classmethod
    def schema(cls, data):
        return named_tuple_helper(cls, {"sam_path": [str]}, data)


def run_args(args):
    config = {}
    for k, v in vars(args).items():
        if v is not None and "." in k:
            config = toolz.assoc_in(config, k.split("."), v)
    try:
        print(config)
        cfg = voluptuous.Schema({
            "eval": EvalCfg.schema
        },
                                extra=voluptuous.REMOVE_EXTRA,
                                required=True)(config)
        run(cfg['eval'])
    except voluptuous.error.Error as e:
        logger.error(humanize_error(config, e))
        raise


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("eval.sam_path", nargs="+")
    parser.add_argument(
        "-r",
        "--reference",
        dest="eval.reference",
        type=voluptuous.IsFile(),
        required=True
    )
    parser.add_argument(
        "-w",
        "--work-dir",
        dest="eval.work_dir",
        type=voluptuous.IsDir(),
        default="."
    )
    parser.set_defaults(func=run_args)


def run(cfg: EvalCfg):
    error_rates_dfs = {}
    consensus_reports = []
    for sam_path in cfg.sam_path:
        basename, ext = os.path.splitext(os.path.basename(sam_path))
        filtered_sam = os.path.join(cfg.work_dir, f"{basename}_filtered.sam")
        logger.info(f"Starting filtering {sam_path} to {filtered_sam}")
        ### Filtering
        # define list of filters (functions that take pysam.AlignedSegment and return boolean)
        filters: List[Callable[[pysam.AlignedSegment]], bool] = [
            # read_len_filter(max_len=400),
            # read_len_filter(max_len=400),
            # secondary_aligments_filter(),
            # supplementary_aligments_filter(),
            only_mapped_filter(),
        ]
        n_kept, n_discarded = filter_aligments_in_sam(
            sam_path, filtered_sam, filters
        )
        logger.info(
            f"{basename} Kept {n_kept}, discarded {n_discarded} after .sam filtering"
        )

        ### Analize error rates
        error_rates_df = error_rates_for_sam(filtered_sam)
        export_dataframe(
            error_rates_df.describe(percentiles=[]).transpose(),
            cfg.work_dir,
            f"error_rates_{basename}"
        )
        error_rates_df['basecaller'] = basename
        error_rates_dfs[basename] = error_rates_df

        # TODO: This is slow and crashes everything
        # position_report = error_positions_report(filtered_sam)
        # logger.info(
        #     f"{basename} Error position report\n{position_report.head(20)}"
        # )
        # fig = plot_error_distributions(position_report)
        # fig.savefig(
        #     os.path.join(cfg.work_dir, f"{basename}_position_report.png")
        # )

        consensus_wdir = os.path.join(cfg.work_dir, basename)
        os.makedirs(consensus_wdir, exist_ok=True)
        report = get_consensus_report(
            basename, filtered_sam, cfg.reference, cfg.is_circular,
            cfg.coverage_threshold, tmp_files_dir=consensus_wdir,
        )
        report.drop(columns=["alignments_file", "mpileup_file"], inplace=True)
        export_dataframe(
            report.transpose(), cfg.work_dir, f"{basename}_consensus_report"
        )
        consensus_reports.append(report)

    for name, fig in plot_read_error_stats(error_rates_dfs).items():
        name: str = name
        name = name.replace(" ", "").replace("%", "")
        fig.savefig(os.path.join(cfg.work_dir, f"read_{name}.png"))

    combined_error_df: pd.DataFrame = pd.concat(error_rates_dfs.values())
    for metric in [
        'Error %',
        'Match %',
        'Mismatch %',
        'Insertion %',
        'Deletion %',
        'Identity %',
        'Read length',
    ]:
        fig, ax = plt.subplots()
        sns.violinplot(
            x='basecaller',
            y=metric,
            ax=ax,
            data=combined_error_df,
            inner="box"
        )
        fig.savefig(
            os.path.join(
                cfg.work_dir,
                f"read_violin_{metric.replace('%', '').replace(' ', '')}.png"
            )
        )
    export_dataframe(
        combined_error_df.groupby("basecaller").mean().drop(
            columns=["Is reversed"]
        ),
        cfg.work_dir,
        f"error_rates"
    )

    consensus = pd.concat(consensus_reports)
    export_dataframe(
        consensus.transpose(), cfg.work_dir, f"all_consensus_report"
    )


def plot_error_distributions(position_report) -> plt.Figure:
    n_cols = 2
    n_rows = 2

    label = {'=': 'Match', 'X': 'Missmatch', 'I': "Insertion", 'D': 'Deletion'}
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16, 10))
    for ax, op in zip(axes.ravel(), ['=', 'X', 'I', 'D']):
        ax: plt.Axes = ax
        data = position_report[position_report.operation == op
                              ]['relative_position']
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


def plot_read_error_stats(error_rates: Dict[str, pd.DataFrame]
                         ) -> Dict[str, plt.Figure]:
    """

    :param error_rates: dictionary of name -> dataframe describing the stats.
        columns are the interesting fields, Error %, Match %, etc.
    :return:
    """
    figs = {}
    axes = {}
    skip_colums = ["Query name"]
    for name, df in error_rates.items():
        for idx, row in df.transpose().iterrows():
            if idx in skip_colums:
                continue
            if idx not in axes:
                fig, ax = plt.subplots()
                figs[idx] = fig
                axes[idx] = ax
                ax.set_title(idx)
            else:
                ax = axes[idx]
            try:
                sns.kdeplot(data=row, ax=ax, label=name)
            except ValueError as e:
                logger.warning(f"{e} occured during plotting read error stats")
                pass
    return figs
