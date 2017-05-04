import logging
import subprocess

import pysam

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
    if exit_status != 0:
        logging.warning("Graphmap exit status %d" % exit_status)


