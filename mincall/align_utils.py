import logging
import subprocess
import math
import pysam
import os
import glob
from mincall.bioinf_utils import reverse_complement, decompress_cigar_pairs, rtrim_cigar, ltrim_cigar


def get_target_sequences(sam_path):
    result_dict = {}

    with pysam.AlignmentFile(sam_path, "r") as samfile:
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

            if name in result_dict:
                prev_target, _, prev_start_pos, _, prev_cigar_str = result_dict[name]
                merged = _merge_circular_aligment(prev_target, prev_start_pos, prev_cigar_str,
                                                  target, start_pos, cigar_str, x.is_reverse, x.query_name)
                if not merged:
                    continue

                target, start_pos, cigar_str = merged
                length = len(target)

            result_dict[name] = [target, ref_name, start_pos, length, cigar_str]
    return result_dict


def _merge_circular_aligment(target_1, start_pos_1, cigar_str_1,
                             target_2, start_pos_2, cigar_str_2, is_reversed, qname):

    if is_reversed:
        # reverse back both
        cigar_str_1 = ''.join(reversed(cigar_str_1))
        target_1 = reverse_complement(target_1)

        cigar_str_2 = ''.join(reversed(cigar_str_2))
        target_2 = reverse_complement(target_2)

    if start_pos_1 == 0:
        start = start_pos_2
        cigar = rtrim_cigar(cigar_str_2) + ltrim_cigar(cigar_str_1)
        target = target_2 + target_1

    elif start_pos_2 == 0:
        start = start_pos_1
        cigar = rtrim_cigar(cigar_str_1) + ltrim_cigar(cigar_str_2)
        target = target_1 + target_2

    else:
        # not circular, duplicate
        logging.error("Duplicate read with name %s", qname)
        return None

    if is_reversed:
        cigar = ''.join(reversed(cigar))
        target = reverse_complement(target)

    return [target, start, cigar]


def align_with_graphmap(reads_path, ref_path, is_circular, out_sam):
    os.makedirs(os.path.dirname(out_sam), exist_ok=True)
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


def secondary_aligments_filter():
    def _filter(aligment):
        return not aligment.is_secondary
    return _filter


def merge_sam_files(sam_dir_path, out_sam_path):
    sam_files = glob.glob(os.path.join(sam_dir_path, '*.sam'))
    os.makedirs(os.path.dirname(out_sam_path), exist_ok=True)
    pysam.merge('-f', out_sam_path, *sam_files, catch_stdout=False)


def merge_reads(reads_fastx_root, out_fastx_path):
    dir_content = [os.path.join(reads_fastx_root, f) for f in os.listdir(reads_fastx_root)]
    files = filter(os.path.isfile, dir_content)

    def _copy_to(outfp, input_file_path):
        with open(input_file_path, 'r') as fin:
            for line in fin:
                outfp.write(line)

    os.makedirs(os.path.dirname(out_fastx_path), exist_ok=True)
    with open(out_fastx_path, 'w') as fout:
        for in_path in files:
            _copy_to(fout, in_path)

# sanity check method
# def merge_reads_test(reads_root_dir, single_reads_file):
#     merge_reads(reads_root_dir, single_reads_file)
#     dir_content = [os.path.join(reads_root_dir, f) for f in os.listdir(reads_root_dir)]
#     files = filter(os.path.isfile, dir_content)
#     reads_before = sum(cnt_sequences_in_fastx(p) for p in files)
#     reads_after = cnt_sequences_in_fastx(single_reads_file)
#     assert reads_before == reads_after


def cnt_sequences_in_fastx(fastx_path):
    with pysam.FastxFile(fastx_path, 'r') as fh:
        total = sum(1 for _ in fh)
    return total


def cnt_reads_in_sam(sam_path):
    with pysam.AlignmentFile(sam_path, "r") as samfile:
        total = sum(1 for _ in samfile.fetch())
    return total
