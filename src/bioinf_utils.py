import itertools
import logging

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


def decompress_cigar(cigar_pairs):
    extended = (cnt * [cigar_int_to_c(b)] for b, cnt in cigar_pairs)
    return ''.join(itertools.chain.from_iterable(extended))


def get_ref_len_from_cigar(cigar_pairs):
    ref_len = 0

    for b, cnt in cigar_pairs:
        sym = cigar_int_to_c(b)
        if sym in 'MX=DNP':
            ref_len += cnt
    return ref_len


def reference_align_string(ref, cigar_pairs):
    """
    Debug method that creates align string with '-' character for insertions
    for passed reference string and cigar passed in tuple form (cnt, byte)
    """
    out_ref = []
    ref_index = 0

    for b, cnt in cigar_pairs:
        sym = cigar_int_to_c(b)
        if sym in 'MX=DNP':
            assert ref_index + cnt <= len(ref)
            out_ref.extend(ref[ref_index:ref_index + cnt])
            ref_index += cnt

        elif sym in 'ISH':
            out_ref.extend(['-'] * cnt)
    return ''.join(out_ref)


def query_align_string(ref, cigar_pairs):
    """
    Debug method that creates align string with '-' character for insertions
    for passed query string and cigar passed in tuple form (cnt, byte)
    """
    out_ref = []
    ref_index = 0

    for b, cnt in cigar_pairs:
        sym = cigar_int_to_c(b)
        if sym in 'MX=ISH':
            assert ref_index + cnt <= len(ref)
            out_ref.extend(ref[ref_index:ref_index + cnt])
            ref_index += cnt

        elif sym in 'DNP':
            out_ref.extend(['-'] * cnt)
    return ''.join(out_ref)


def reverse_complement(seq):
    bases = reversed([COMPLEMENT.get(b, b) for b in seq])
    return ''.join(bases)


def read_fasta(fp):
    def rr(f):
        return "".join(line.strip() for line in f.readlines() if ">" not in line)

    if not hasattr(fp, 'readlines'):
        with open(fp, 'r') as f:
            return rr(f)
    else:
        return rr(fp)