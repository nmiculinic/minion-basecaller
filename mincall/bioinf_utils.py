import itertools
import logging
import pysam
import re
import csv
import numpy as np
from collections import Counter, defaultdict
import pandas as pd
from tqdm import tqdm

CIGAR_TO_BYTE = {
    'M': 0,
    'I': 1,
    'D': 2,
    'N': 3,
    'S': 4,
    'H': 5,
    'P': 6,
    '=': 7,
    'X': 8
}

CIGAR_OPERATIONS = 'MIDNSHP=X'
CIGAR_MATCH_MISSMATCH = 'M=X'
CIGAR_MATCH = ['=']
CIGAR_MISSMATCH = ['X']
CIGAR_INSERTION = 'ISH'
CIGAR_DELETION = 'DNP'
CIGAR_SOFTCLIP = ['S']
CIGAR_HARDCLIP = ['H']
CIGAR_CLIP = 'SH'

BYTE_TO_CIGAR = {v: k for k, v in CIGAR_TO_BYTE.items()}

COMPLEMENT = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
"""
    CIGAR utils
"""


def cigar_c_to_int(c):
    ret = CIGAR_TO_BYTE.get(c, None)
    if ret is None:
        logging.error('cigar_c_to_int invalid base %s' % c)
    return ret


def cigar_int_to_c(b):
    ret = BYTE_TO_CIGAR.get(b, None)
    if ret is None:
        logging.error('cigar_int_to_c invalid byte %d' % b)
    return ret


def cigar_pairs_to_str(cigar_pairs, mode='ints'):
    if mode == 'ints':
        convert = cigar_int_to_c
    elif mode == 'chars':
        convert = lambda x: x
    else:
        raise ValueError('Invalid mode argument. Expected ints or chars')

    cigar = ('%d%s' % (cnt, convert(b)) for b, cnt in cigar_pairs)
    return ''.join(cigar)


def cigar_str_to_pairs(cigar):
    split_locations = []
    for i, c in enumerate(cigar):
        if c in CIGAR_OPERATIONS:
            split_locations.append(i)

    cigar_pairs = []
    for i, end in enumerate(split_locations):
        start = split_locations[i - 1] + 1 if i > 0 else 0

        cnt = int(cigar[start:end])
        char = cigar[end].upper()
        cigar_pairs.append((char, cnt))
    return cigar_pairs


def decompress_cigar_pairs(cigar_pairs, mode='ints'):
    if mode == 'ints':
        convert = cigar_int_to_c
    elif mode == 'chars':
        convert = lambda x: x
    else:
        raise ValueError('Invalid mode argument. Expected ints or chars')

    extended = (cnt * [convert(b)] for b, cnt in cigar_pairs)
    return ''.join(itertools.chain.from_iterable(extended))


def decompress_cigar(cigar_str):
    cigar_pairs = cigar_str_to_pairs(cigar_str)
    return decompress_cigar_pairs(cigar_pairs, mode='chars')


def compress_cigar(cigar_str, ret_mode='chars'):
    if ret_mode == 'ints':
        convert = cigar_c_to_int
    elif ret_mode == 'chars':
        convert = lambda x: x
    else:
        raise ValueError('Invalid mode argument. Expected ints or chars')

    pairs = []
    count = 1

    for i in range(1, len(cigar_str)):
        if cigar_str[i - 1] == cigar_str[i]:
            count += 1
        else:
            pairs.append((convert(cigar_str[i - 1]), count))
            count = 1
    pairs.append((convert(cigar_str[-1]), count))
    return pairs


def ltrim_cigar(cigar):
    for op in CIGAR_CLIP:
        cigar = re.sub(r'^%s*' % op, r'', cigar)
    return cigar


def rtrim_cigar(cigar):
    for op in CIGAR_CLIP:
        cigar = re.sub(r'%s*$' % op, r'', cigar)
    return cigar


def get_ref_len_from_cigar_pairs(cigar_pairs):
    ref_len = 0
    for b, cnt in cigar_pairs:
        sym = cigar_int_to_c(b) if isinstance(b, int) else b
        if sym in CIGAR_MATCH_MISSMATCH or sym in CIGAR_DELETION:
            ref_len += cnt
    return ref_len


def get_ref_len_from_cigar(decompressed_cigar):
    ref_len = 0
    for op in decompressed_cigar:
        if op in CIGAR_MATCH_MISSMATCH or op in CIGAR_DELETION:
            ref_len += 1
    return ref_len


def get_read_len_from_cigar(decompressed_cigar):
    ref_len = 0
    for op in decompressed_cigar:
        if op in CIGAR_MATCH_MISSMATCH or op in CIGAR_INSERTION or op in CIGAR_CLIP:
            ref_len += 1
    return ref_len


def reference_align_string(ref, cigar_int_pairs):
    """
    Debug method that creates align string with '-' character for insertions
    for passed reference string and cigar passed in tuple form (cnt, byte)
    """
    out_ref = []
    ref_index = 0

    for b, cnt in cigar_int_pairs:
        sym = cigar_int_to_c(b) if isinstance(b, int) else b
        if sym in CIGAR_MATCH_MISSMATCH or sym in CIGAR_DELETION:
            assert ref_index + cnt <= len(ref)
            out_ref.extend(ref[ref_index:ref_index + cnt])
            ref_index += cnt

        elif sym in CIGAR_INSERTION:
            out_ref.extend(['-'] * cnt)
    return ''.join(out_ref)


def query_align_string(ref, cigar_int_pairs):
    """
    Debug method that creates align string with '-' character for insertions
    for passed query string and cigar passed in tuple form (cnt, byte)
    """
    out_ref = []
    ref_index = 0

    for b, cnt in cigar_int_pairs:
        sym = cigar_int_to_c(b) if isinstance(b, int) else b
        if sym in CIGAR_MATCH_MISSMATCH or sym in CIGAR_INSERTION:
            assert ref_index + cnt <= len(ref)
            out_ref.extend(ref[ref_index:ref_index + cnt])
            ref_index += cnt

        elif sym in CIGAR_DELETION:
            out_ref.extend(['-'] * cnt)
    return ''.join(out_ref)


def generate_md_tag(ref, cigar_pairs=None):
    md = []
    ref_index = 0
    prev_cnt = 0

    for b, cnt in cigar_pairs:
        sym = cigar_int_to_c(b) if isinstance(b, int) else b
        if sym in CIGAR_MATCH:
            ref_index += cnt
            prev_cnt += cnt

        elif sym in CIGAR_MISSMATCH:
            assert ref_index + cnt <= len(ref)

            for c in ref[ref_index:ref_index + cnt]:
                md.append(str(prev_cnt))
                md.append(c.upper())
                prev_cnt = 0

            ref_index += cnt

        elif sym in CIGAR_DELETION:
            md.append(str(prev_cnt))
            prev_cnt = 0
            assert ref_index + cnt <= len(ref)
            md.extend('^' + ref[ref_index:ref_index + cnt].upper())
            ref_index += cnt

    if prev_cnt:
        md.append(str(prev_cnt))
    return ''.join(md)


def reverse_complement(seq):
    bases = reversed([COMPLEMENT.get(b, b) for b in seq])
    return ''.join(bases)


def error_rates_from_cigar(cigar_full_str):
    cntr = Counter(cigar_full_str)

    def get_cnt(keys):
        return sum([cntr[c] for c in keys])

    cigar_len = len(cigar_full_str)
    n_deletions = get_cnt(CIGAR_DELETION)
    n_all_insertions = get_cnt(CIGAR_INSERTION)
    n_insertions = n_all_insertions - get_cnt(CIGAR_SOFTCLIP
                                             ) - get_cnt(CIGAR_HARDCLIP)
    n_missmatches = get_cnt(CIGAR_MISSMATCH)
    n_matches = get_cnt(CIGAR_MATCH)

    read_len = n_insertions + n_missmatches + n_matches
    target_len = n_deletions + n_missmatches + n_matches
    noncliped_len = n_all_insertions + n_missmatches + n_matches
    if cigar_len != n_deletions + n_all_insertions + n_missmatches + n_matches:
        raise ValueError(
            "cigar_len != n_deletions + n_all_insertions + n_missmatches + n_matches"
            ";Expected extended cigar format"
        )
    if not read_len:
        logging.warning("Clipped read len == 0")
        return None

    n_errors = n_missmatches + n_insertions + n_deletions

    error_rate = 100 * n_errors / read_len
    match_rate = 100 * n_matches / read_len
    missmatch_rate = 100 * n_missmatches / read_len
    ins_rate = 100 * n_insertions / read_len
    del_rate = 100 * n_deletions / read_len
    identity_rate = 100 * n_matches / cigar_len

    return error_rate, match_rate, missmatch_rate, ins_rate, del_rate, identity_rate, noncliped_len


ERROR_RATES_COLUMNS = [
    'Query name', 'Error %', 'Match %', 'Mismatch %', 'Insertion %',
    'Deletion %', 'Identity %', 'Read length', 'Is reversed'
]


def error_rates_for_sam(sam_path):
    all_errors = []
    unmapped = 0
    with pysam.AlignmentFile(sam_path, "r") as samfile:
        for x in tqdm(samfile.fetch(), unit='read'):
            if x.is_unmapped:
                logging.debug("%s is unmapped", x.query_name)
                unmapped += 1
                continue

            qname = x.query_name
            cigar_pairs = x.cigartuples

            full_cigar = decompress_cigar_pairs(cigar_pairs, mode='ints')
            rates = error_rates_from_cigar(full_cigar)
            if rates:
                all_errors.append([qname] + list(rates) + [x.is_reverse])
            else:
                logging.warning('Problem with read %s', qname)
    if unmapped > 0:
        logging.error("%d reads were unmapped", unmapped)
    return pd.DataFrame(all_errors, columns=ERROR_RATES_COLUMNS)


def error_positions_report(sam_path, n_buckets=1000):
    bucket_width = 1 / n_buckets

    def _update_positions(cigar_str, positions_dict):
        cigar_len = len(cigar_str)
        for i, c in enumerate(cigar_str.upper()):
            relative_pos = i / cigar_len
            bucket_index = relative_pos // bucket_width
            bucket_pos = bucket_index * bucket_width
            positions_dict[bucket_pos][c] += 1

    positions = defaultdict(lambda: defaultdict(int))
    n_reads = 0
    with pysam.AlignmentFile(sam_path, "r") as samfile:
        for x in samfile.fetch():
            cigar_pairs = x.cigartuples
            if not cigar_pairs:
                logging.error("%s No cigar string found", x.query_name)
                continue
            full_cigar = decompress_cigar_pairs(cigar_pairs, mode='ints')
            _update_positions(full_cigar, positions)
            n_reads += 1

    data = [[pos, op, cnt]
            for pos, count in positions.items()
            for op, cnt in count.items()]

    df = pd.DataFrame(
        data, columns=['relative_position', 'operation', 'op_count']
    )
    df.sort_values(by='relative_position', inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


def read_fasta(fp):
    def rr(f):
        return "".join(
            line.strip() for line in f.readlines() if ">" not in line
        )

    if not hasattr(fp, 'readlines'):
        with open(fp, 'r') as f:
            return rr(f)
    else:
        return rr(fp)
