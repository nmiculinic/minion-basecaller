import edlib
import re
from minion_data import dataset_pb2
from typing import *
from collections import defaultdict
import tensorflow as tf
import voluptuous
from mincall import bioinf_utils

__all__ = [
    "decode", "tensor_default_summaries", "squggle", "named_tuple_helper",
    "ext_cigar_stats", "TOTAL_BASE_PAIRS"
]

TOTAL_BASE_PAIRS = 4  # Total number of bases (A, C, T, G)  # Total number of bases (A, C, T, G)


def decode(x):
    return "".join([
        dataset_pb2.BasePair.Name(int(yy) % TOTAL_BASE_PAIRS) for yy in x
    ])


cigar_type = {}
for x in bioinf_utils.CIGAR_DELETION:
    cigar_type[x] = dataset_pb2.DELETION
for x in bioinf_utils.CIGAR_INSERTION:
    cigar_type[x] = dataset_pb2.INSERTION
for x in bioinf_utils.CIGAR_MATCH:
    cigar_type[x] = dataset_pb2.MATCH
for x in bioinf_utils.CIGAR_MISSMATCH:
    cigar_type[x] = dataset_pb2.MISMATCH


def ext_cigar_stats(ext_cigar: str) -> Dict[int, int]:
    sol = defaultdict(int)
    for x in re.findall(r"\d+[=XIDSHM]", ext_cigar):
        cnt = int(x[:-1])
        op = x[-1]
        sol[cigar_type[op]] += cnt
    return sol


def squggle(query: str, target: str) -> Tuple[str, str, Dict]:
    if len(query) > 0:
        alignment = edlib.align(query, target, task='path')
    else:
        alignment = {
            'editDistance': len(target),
            'cigar': f"{len(target)}D",
            'alphabetLength': 4,
            'locations': [(0, len(target))],
        }

    cigar = alignment['cigar']
    q_idx = 0
    t_idx = 0

    qq, tt = "", ""
    for x in re.findall(r"\d+[=XIDSHM]", cigar):
        cnt = int(x[:-1])
        op = x[-1]
        if op in ["=", "X"]:
            qq += query[q_idx:q_idx + cnt]
            q_idx += cnt

            tt += target[t_idx:t_idx + cnt]
            t_idx += cnt
        elif op == "D":
            qq += "-" * cnt

            tt += target[t_idx:t_idx + cnt]
            t_idx += cnt
        elif op == "I":
            qq += query[q_idx:q_idx + cnt]
            q_idx += cnt

            tt += "-" * cnt
        else:
            ValueError(f"Unknown op {op}")
    assert len(target) == t_idx, "Not all target base pairs used"
    assert len(query) == q_idx, "Not all target base pairs used"
    return qq, tt, alignment


def tensor_default_summaries(name, tensor, family=None,
                             full=False) -> List[tf.Summary]:
    mean, var = tf.nn.moments(tensor, axes=list(range(len(tensor.shape))))
    ret = [
        tf.summary.scalar(name + '/mean', mean, family=family),
        tf.summary.histogram(name + '/histogram', tensor, family=family),
    ]
    if full:
        ret += [
            tf.summary.scalar(name + '/stddev', tf.sqrt(var), family=family),
            tf.summary.scalar(
                name + '/max', tf.reduce_max(tensor), family=family
            ),
            tf.summary.scalar(
                name + '/min', tf.reduce_min(tensor), family=family
            ),
        ]
    return ret


def named_tuple_helper(cls, known, data):
    for k, v in cls.__annotations__.items():
        if k not in known:
            if k in cls._field_defaults:
                known[voluptuous.Optional(k)] = voluptuous.Coerce(v)
            else:
                known[k] = voluptuous.Coerce(v)
    schema = voluptuous.Schema(
        {
            **known,
        },
        required=True,
    )
    return cls(**schema(data))
