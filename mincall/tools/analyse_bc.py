import os
import pandas as pd
import logging
import subprocess
import argparse
import mincall.align_utils as align_utils
from mincall.align_utils import filter_aligments_in_sam, read_len_filter
from mincall.bioinf_utils import error_rates_for_sam, error_positions_report, CIGAR_OPERATIONS
import seaborn as sns
import matplotlib.pyplot as plt
from mincall.consensus import get_consensus_report


log_fmt = '\r[%(levelname)s] %(name)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

args = argparse.ArgumentParser()
args.add_argument("input_folder", help="Input fast5 folder to basecall")
args.add_argument("out_folder", help="Output folder for all analysis")
args.add_argument("--ref", help="Reference genome")
args.add_argument("--name", help="Run name")
args.add_argument("--min_length", type=int, default=500, help="Minimum read lenght for further analysis")
args.add_argument("-c", "--circular", help="Is genome circular", action="store_true")
args.add_argument("--coverage_threshold", help="Minimal coverage threshold for consensus", type=float, default=0.0)
args = args.parse_args()

basecallers = {
    "mincall_m270": ["nvidia-docker", "run", "--rm", "-v", "%s:/data" % args.input_folder, "-u=%d" % os.getuid(), "nmiculinic/mincall:9947283"],
    "nanonet": ["nanonetcall", args.input_folder, "--chemistry", "r9", "--platforms", "nvidia:0:20", "--exc_opencl"],
    "albacore": None,
    # "nanonet": ["nanonetcall", args.input_folder, "--chemistry", "r9", "--jobs", "8"],
    'metrichorn': ["poretools", "fastq", "--type fwd", args.input_folder]
}

os.makedirs(args.out_folder, exist_ok=True)
dfs = {}
for name, cmd in basecallers.items():
    logger = logging.getLogger(name)
    fasta_path = os.path.join(args.out_folder, name + ".fa")
    fastq_path = os.path.join(args.out_folder, name + ".fastq")
    if os.path.isfile(fasta_path) or os.path.isfile(fastq_path):
        logger.info("%s exists, skipping", fasta_path)
    else:
        logger.info("Basecalling %s with %s", args.input_folder, name)
        logger.info("Output file %s", fasta_path)
        logger.info("Full command: %s", " ".join(cmd))
        with open(fasta_path, 'w') as f:
            subprocess.check_call(cmd, stdout=f)

    sam_path = os.path.join(args.out_folder, name + ".sam")
    if os.path.isfile(sam_path):
        logger.info("%s exists, skipping", sam_path)
    else:
        path = fasta_path
        if os.path.exists(fastq_path):
            path = fastq_path
        logger.info("Aligning %s to reference %s with graphmap", path, args.ref)
        align_utils.align_with_graphmap(path, args.ref, args.circular, sam_path)

    filtered_sam = os.path.join(args.out_folder, name + "_filtered.sam")
    if os.path.isfile(filtered_sam):
        logger.info("%s exists, skipping", filtered_sam)
    else:
        filters = [read_len_filter(min_len=args.min_length)]
        n_kept, n_discarded = filter_aligments_in_sam(sam_path, filtered_sam, filters)
        logger.info("Outputed filtered sam to %s\n%d kept, %d discarded",
                 filtered_sam, n_kept, n_discarded)

    reads_csv = os.path.join(args.out_folder, name + "_read_data.csv")
    if os.path.isfile(reads_csv):
        logger.info("%s file exists, loading", reads_csv)
        df = pd.read_pickle(os.path.join(args.out_folder, name + "_read_summary.pkl"))
    else:
        df = error_rates_for_sam(filtered_sam)
        df.to_csv(reads_csv)
        df.to_pickle(os.path.join(args.out_folder, name + "_read_summary.pkl"))
        desc = df.describe()
        desc.to_latex(os.path.join(args.out_folder, name + "_read_summary.tex"))
        desc.to_string(os.path.join(args.out_folder, name + "_read_summary.txt"))
    df = error_rates_for_sam(filtered_sam)
    logger.info("%s\n%s", name, df.describe())
    dfs[name] = df

    consensus_report_path = os.path.join(args.out_folder, name + "consensus_report.csv")
    if os.path.isfile(consensus_report_path):
        consensus_report = pd.read_csv(consensus_report_path)
        logger.info("%s exists, loading", consensus_report_path)
    else:
        consensus_report = get_consensus_report(filtered_sam, args.ref, args.coverage_threshold)
        consensus_report.to_csv(consensus_report_path)
        with open(os.path.splitext(consensus_report_path)[0] + ".txt", 'w') as f:
            consensus_report.to_string(f)
    logger.info("%s consensus_report:\n%s", name, consensus_report)


columns = list(iter(next(iter(dfs.values()))._get_numeric_data()))
df_prep = []
names = []
for name, df in dfs.items():
    names.append(name)
    df_prep.append({col: df[col].mean() for col in columns})
df_summary = pd.DataFrame(df_prep, index=names)
df_summary.to_csv(os.path.join(args.out_folder, args.name + "_summary.csv"))
df_summary.to_latex(os.path.join(args.out_folder, args.name + "_summary.tex"))
del names
del df_prep

fig_kde_path = os.path.join(args.out_folder, args.name + "reads_kde.png")
fig_hist_path = os.path.join(args.out_folder, args.name + "reads_hist.png")

fig_kde, axes_kde = plt.subplots(3, 2)
fig_hist, axes_hist = plt.subplots(3, 2)
fig_kde.set_size_inches(12, 20)
fig_hist.set_size_inches(12, 20)

for col, ax_kde, ax_hist in zip(columns, axes_kde.ravel(), axes_hist.ravel()):
    fig, ax = plt.subplots()
    logging.info("Plotting column %s", col)
    for k in dfs.keys():
        sns.kdeplot(dfs[k][col], shade=True, label=k, alpha=0.5, ax=ax)
        sns.kdeplot(dfs[k][col], shade=True, label=k, alpha=0.5, ax=ax_kde)
        ax_hist.hist(dfs[k][col], label=k, alpha=0.5)
    ax.set_title(args.name + " " + col)
    fig.savefig(os.path.join(args.out_folder, col + ".png"))
    for ax in [ax_kde, ax_hist]:
        ax.legend()
        ax.set_title(col)
        if not col == "Read length":
            ax.set_xlim([0.0, 1.0])

fig_kde.savefig(fig_kde_path)
fig_hist.savefig(fig_hist_path)
