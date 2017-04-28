import itertools
import logging
from collections import Counter

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


def cigar_pairs_to_str(cigar_pairs):
    cigar = ('%d%s' % (cnt, cigar_int_to_c(b)) for b, cnt in cigar_pairs)
    return ''.join(cigar)


def cigar_str_to_pairs(cigar):
    split_locations = []
    for i, c in enumerate(cigar):
        if c in CIGAR_OPERATIONS:
            split_locations.append(i)

    cigar_pairs = []
    for i, end in enumerate(split_locations):
        start = split_locations[i-1] + 1 if i > 0 else 0

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
        raise Exception('Invalid mode argument. Expected ints or chars')

    extended = (cnt * [convert(b)] for b, cnt in cigar_pairs)
    return ''.join(itertools.chain.from_iterable(extended))


def decompress_cigar(cigar_str):
    cigar_pairs = cigar_str_to_pairs(cigar_str)
    return decompress_cigar_pairs(cigar_pairs, mode='chars')


def get_ref_len_from_cigar(cigar_pairs):
    ref_len = 0

    for b, cnt in cigar_pairs:
        sym = cigar_int_to_c(b)
        if sym in 'MX=DNP':
            ref_len += cnt
    return ref_len


def reference_align_string(ref, cigar_int_pairs):
    """
    Debug method that creates align string with '-' character for insertions
    for passed reference string and cigar passed in tuple form (cnt, byte)
    """
    out_ref = []
    ref_index = 0

    for b, cnt in cigar_int_pairs:
        sym = cigar_int_to_c(b)
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
        sym = cigar_int_to_c(b)
        if sym in CIGAR_MATCH_MISSMATCH or sym in CIGAR_INSERTION:
            assert ref_index + cnt <= len(ref)
            out_ref.extend(ref[ref_index:ref_index + cnt])
            ref_index += cnt

        elif sym in CIGAR_DELETION:
            out_ref.extend(['-'] * cnt)
    return ''.join(out_ref)


def reverse_complement(seq):
    bases = reversed([COMPLEMENT.get(b, b) for b in seq])
    return ''.join(bases)


def error_rates_from_cigar(cigar_full_str):
    cntr = Counter(cigar_full_str)

    def get_cnt(keys):
        return sum([cntr[c] for c in keys])

    cigar_len = len(cigar_full_str)
    n_deletions = get_cnt(CIGAR_DELETION)
    n_insertions = get_cnt(CIGAR_INSERTION)
    n_missmatches = get_cnt(CIGAR_MISSMATCH)
    n_matches = get_cnt(CIGAR_MATCH)

    ref_len = n_insertions + n_missmatches + n_matches
    read_len = n_deletions + n_missmatches + n_matches

    assert cigar_len == n_deletions + n_insertions + n_missmatches + n_matches

    n_errors = n_missmatches + n_insertions + n_deletions

    error_rate = n_errors / read_len
    match_rate = n_matches / read_len
    missmatch_rate = n_missmatches / read_len
    ins_rate = n_insertions / read_len
    del_rate = n_deletions / read_len

    return error_rate, match_rate, missmatch_rate, ins_rate, del_rate


def read_fasta(fp):
    def rr(f):
        return "".join(line.strip() for line in f.readlines() if ">" not in line)

    if not hasattr(fp, 'readlines'):
        with open(fp, 'r') as f:
            return rr(f)
    else:
        return rr(fp)