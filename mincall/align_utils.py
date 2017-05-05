import logging
import subprocess
import math
import pysam
import os
import glob
from mincall.bioinf_utils import reverse_complement, decompress_cigar_pairs


def get_target_sequences(sam_out):
    result_dict = {}

    with pysam.AlignmentFile(sam_out, "r") as samfile:
        for x in samfile.fetch():
            name = x.query_name

            if x.is_unmapped:
                logging.warning("%s unmapped" % name)
                continue
            try:
                target = x.get_reference_sequence()
            except ValueError:
                logging.error("%s Mapped but reference len equals 0" % name)
                continue

            ref_name = x.reference_name
            length = x.reference_length
            start_pos = x.reference_start
            cigar_pairs = x.cigartuples

            if x.is_reverse:
                target = reverse_complement(target)
                cigar_pairs = list(reversed(cigar_pairs))

            cigar_str = decompress_cigar_pairs(cigar_pairs, mode='ints')
            result_dict[name] = [target, ref_name, start_pos, length, cigar_str]
    return result_dict


def align_with_graphmap(reads_path, ref_path, is_circular, out_sam):
    args = ["graphmap", "align", "-r", ref_path, "-d", reads_path, "-o", out_sam, "-v", "0", "--extcigar"]
    if is_circular:
        args.append("-C")

    exit_status = subprocess.call(args)
    logging.info("Graphmap exit status %d" % exit_status)

    if exit_status != 0:
        logging.warning("Graphmap exit status %d" % exit_status)


def filter_aligments_in_sam(sam_path, out_path, filters=[]):
    n_reads = 0
    n_kept = 0

    with pysam.AlignmentFile(sam_path, "r") as in_sam:
        with pysam.AlignmentFile(out_path, "w", template=in_sam) as out_sam:

            for x in in_sam.fetch():
                n_reads += 1

                if all([f(x) for f in filters]):
                    n_kept += 1
                    out_sam.write(x)
    return n_kept, n_reads - n_kept


def read_len_filter(min_len=-1, max_len=math.inf):
    def _filter(aligment):
        return min_len < aligment.query_length < max_len

    return _filter


def merge_sam_files(sam_dir_path, out_sam_path):
    sam_files = glob.glob(os.path.join(sam_dir_path, '*.sam'))
    pysam.merge('-f', out_sam_path, *sam_files, catch_stdout=False)


def merge_reads(reads_fastx_root, out_fastx_path):
    dir_content = [os.path.join(reads_fastx_root, f) for f in os.listdir(reads_fastx_root)]
    files = filter(os.path.isfile, dir_content)

    def _copy_to(outfp, input_file_path):
        with open(input_file_path, 'r') as fin:
            for line in fin:
                outfp.write(line)

    with open(out_fastx_path, 'w') as fout:
        for in_path in files:
            _copy_to(fout, in_path)

