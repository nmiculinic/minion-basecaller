import edlib
import numpy as np
import re
import unittest
from mincall.errors import TooLargeEditDistance, BlockSizeYTooSmall, ZeroLenY
from mincall.bioinf_utils import CIGAR_MATCH_MISSMATCH, CIGAR_INSERTION, CIGAR_DELETION

def dump_fasta(name, fasta, fd):
    print(">" + name, file=fd)
    n = 80
    for i in range(0, len(fasta), n):
        print(fasta[i:i + n], file=fd)


def next_num(prev, symbol):
    val = {
        'A': 0,
        'G': 1,
        'T': 2,
        'C': 3,
    }[symbol]
    if symbol == prev:
        return 'N', val + 4
    else:
        return symbol, val


def read_fasta(fp):
    def rr(f):
        return "".join(line.strip() for line in f.readlines() if ">" not in line)

    if not hasattr(fp, 'readlines'):
        with open(fp, 'r') as f:
            return rr(f)
    else:
        return rr(fp)


def decode(arr):
    arr = arr.astype(np.int32)
    y_out = np.zeros_like(arr, dtype=np.unicode)
    for i, a in enumerate("AGTCAGTCN"):
        y_out[arr == i] = a
    return y_out


def decode_sparse(arr, pad=None):
    indices = arr.indices
    if pad is None:
        pad = np.max(indices[:, 1]) + 1
    values = decode(arr.values)
    shape = arr.shape

    tmp = np.array([[' '] * pad] * shape[0])
    for ind, val in zip(indices, values):
        r, c = ind
        tmp[r, c] = val
    sol = np.array([' ' * pad] * shape[0])
    for row in range(shape[0]):
        sol[row] = "".join([c for c in tmp[row]])
    return sol


def decode_example(Y, Y_len, num_blocks, block_size_y, pad=None):
    gg = []
    for blk in range(num_blocks):
        gg.append("".join([str(x) for x in decode(Y[blk * block_size_y:blk * block_size_y + Y_len[blk]].ravel())]))

    if pad is None:
        pad = np.max(list(map(len, gg)))

    return list(map(lambda x: x.ljust(pad, ' '), gg))


cigar_re = re.compile(r"\d+[ID=X]")


def breakCigar(cigar):
    return map(lambda g: (int(g[:-1]), g[-1]), cigar_re.findall(cigar))


def correct_basecalled(bucketed_basecall, reference, nedit_tol=0.2):
    basecalled = "".join(bucketed_basecall)
    origin = np.zeros(len(basecalled), dtype=np.int32)
    idx = 0
    for i, b in enumerate(bucketed_basecall):
        origin[idx:idx + len(b)] = i
        idx += len(b)

    result_set = edlib.align(basecalled, reference, task="path")
    nedit = result_set['editDistance'] / len(reference)
    if nedit > nedit_tol:
        raise TooLargeEditDistance("Normalized edit distance is large...%.3f" % nedit)

    result = ["" for _ in bucketed_basecall]
    idx_ref = 0
    idx_bcalled = 0

    for num, op in breakCigar(result_set['cigar']):
        for _ in range(num):
            if op in CIGAR_MATCH_MISSMATCH:
                result[origin[idx_bcalled]] += reference[idx_ref]
                idx_bcalled = min(idx_bcalled + 1, len(basecalled) - 1)
                idx_ref += 1
            elif op in CIGAR_INSERTION:
                idx_bcalled += 1
            elif op in CIGAR_DELETION:
                result[origin[idx_bcalled]] += reference[idx_ref]
                idx_ref += 1
    return result


def prepare_y(bucketed_basecall, block_size_y):
    num_blocks = len(bucketed_basecall)
    y = np.zeros([num_blocks * block_size_y], dtype=np.uint8)
    y_len = np.zeros([num_blocks], dtype=np.int32)

    prev = "N"
    for i, seq in enumerate(bucketed_basecall):
        y_len[i] = len(seq)
        if y_len[i] > block_size_y:
            raise BlockSizeYTooSmall("On block {}, got {}".format(i, y_len[i]))

        if y_len[i] == 0:
            raise ZeroLenY()

        for j, c in enumerate(seq):
            prev, y[i * block_size_y + j] = next_num(prev, c)

    return y, y_len


class TestCorrectedBasecalled(unittest.TestCase):
    def test_eq(self):
        self.assertEqual(correct_basecalled(["AA"], "AA"), ["AA"])

    def test_del(self):
        self.assertEqual(correct_basecalled(["AA"], "AAC", nedit_tol=1.0), ["AAC"])

    def test_ins(self):
        self.assertEqual(correct_basecalled(["AAD"], "AA", nedit_tol=1.0), ["AA"])

    def test_mis(self):
        self.assertEqual(correct_basecalled(["AAD"], "AAC", nedit_tol=1.0), ["AAC"])


def sigopt_numeric(type, name, min, max):
    return dict(
        name=name,
        type=type,
        bounds=dict(
            min=min,
            max=max
        )
    )


def sigopt_int(name, min, max):
    return sigopt_numeric('int', name, min, max)


def sigopt_double(name, min, max):
    return sigopt_numeric('double', name, min, max)


if __name__ == '__main__':
    unittest.main()
