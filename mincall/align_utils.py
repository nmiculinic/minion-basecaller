import logging
import subprocess
import math
import pysam
import os
import glob
import multiprocessing
from mincall import bioinf_utils as butil
import shutil
import tempfile
from tqdm import tqdm


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
                logging.error("%s Mapped but reference len equals 0, md tag: %s", name, x.has_tag('MD'))
                continue

            ref_name = x.reference_name
            length = x.reference_length
            start_pos = x.reference_start
            cigar_pairs = x.cigartuples

            if x.is_reverse:
                target = butil.reverse_complement(target)
                cigar_pairs = list(reversed(cigar_pairs))

            cigar_str = butil.decompress_cigar_pairs(cigar_pairs, mode='ints')

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
        target_1 = butil.reverse_complement(target_1)

        cigar_str_2 = ''.join(reversed(cigar_str_2))
        target_2 = butil.reverse_complement(target_2)

    if start_pos_1 == 0:
        start = start_pos_2
        cigar = butil.rtrim_cigar(cigar_str_2) + butil.ltrim_cigar(cigar_str_1)
        target = target_2 + target_1

    elif start_pos_2 == 0:
        start = start_pos_1
        cigar = butil.rtrim_cigar(cigar_str_1) + butil.ltrim_cigar(cigar_str_2)
        target = target_1 + target_2

    else:
        # not circular, duplicate
        logging.error("Duplicate read with name %s", qname)
        return None

    if is_reversed:
        cigar = ''.join(reversed(cigar))
        target = butil.reverse_complement(target)

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


def align_with_bwa_mem(reads_path, ref_path, is_circular, out_sam, extended_cigar=True):
    os.makedirs(os.path.dirname(out_sam), exist_ok=True)
    if os.path.exists(out_sam):
        logging.info("Removing %s", out_sam)
        os.remove(out_sam)

    n_threads = multiprocessing.cpu_count()
    args = ["bwa", "mem", "-x", "ont2d", "-t", str(n_threads),
            ref_path, reads_path]
    args_index = ["bwa", "index", ref_path]

    def _align():
        with open(out_sam, 'w') as f:
            ex_status = subprocess.call(args, stdout=f)
        return ex_status

    if is_circular:
        logging.warning("BWA mem circular reference flag not implemented")

    def _log_exit_status(msg, ex_status):
        log = logging.warning if ex_status != 0 else logging.info
        log("%s exit status %d", msg, ex_status)

    exit_status = _align()
    _log_exit_status("bwa mem align", exit_status)

    if exit_status != 0:
        # build index if needed
        logging.info("bwa mem error recovery, trying to rebuild index")
        exit_status = subprocess.call(args_index)
        _log_exit_status("bwa index", exit_status)
        if exit_status == 0:
            exit_status = _align()
            _log_exit_status("bwa mem align", exit_status)

    if exit_status == 0:
        extend_cigars_in_sam(out_sam, ref_path, reads_path)


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


def supplementary_aligments_filter():
    def _filter(aligment):
        return not aligment.is_supplementary
    return _filter


def only_mapped_filter():
    def _filter(x):
        return not x.is_unmapped
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


def extend_cigar(read_seq, ref_seq, cigar_pairs, mode='ints'):
    cigar_str = butil.decompress_cigar_pairs(cigar_pairs, mode)

    ref_seq = butil.reference_align_string(ref_seq, cigar_pairs)
    read_seq = butil.query_align_string(read_seq, cigar_pairs)

    assert len(ref_seq) == len(cigar_str) and len(read_seq) == len(cigar_str)

    def _resolve_m(i, op):
        if op.upper() == 'M':
            return '=' if ref_seq[i].upper() == read_seq[i].upper() else 'X'
        return op.upper()

    cigar_str = ''.join(_resolve_m(*p) for p in enumerate(cigar_str))
    pairs = butil.compress_cigar(cigar_str)
    cigar = butil.cigar_pairs_to_str(pairs, 'chars')
    return cigar


def extend_cigars_in_sam(sam_in, ref_path, fastx_path, sam_out=None):
    tmp_dir = None
    tmp_sam_out = sam_out
    inplace = sam_out is None

    if inplace:
        # inplace change using tmp file
        tmp_dir = tempfile.mkdtemp()
        tmp_sam_out = os.path.join(tmp_dir, 'tmp.sam')

    ref = butil.read_fasta(ref_path)
    reads = {}

    with pysam.FastxFile(fastx_path, 'r') as fh:
        for r in fh:
            reads[r.name] = r

    with pysam.AlignmentFile(sam_in, "r") as in_sam, \
            pysam.AlignmentFile(tmp_sam_out, "w", template=in_sam) as out_sam:

        for x in tqdm(in_sam.fetch(), unit='reads'):
            if x.query_name not in reads:
                logging.warning("read %s in sam not found in .fastx", x.query_name)
                continue

            if x.is_unmapped:
                logging.warning("read %s is unmapped, copy to out sam as is", x.query_name)
                out_sam.write(x)
                continue

            read_seq = reads[x.query_name].sequence
            ref_seq = ref[x.reference_start:x.reference_end]
            cigar_pairs = x.cigartuples

            if x.is_reverse:
                read_seq = butil.reverse_complement(read_seq)

            x.cigarstring = extend_cigar(read_seq, ref_seq, cigar_pairs)
            out_sam.write(x)

    if inplace:
        # clear tmp files
        shutil.move(tmp_sam_out, sam_in)
        shutil.rmtree(tmp_dir)
