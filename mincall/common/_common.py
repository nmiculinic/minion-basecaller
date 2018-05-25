import edlib
from minion_data import dataset_pb2
from typing import *
import tensorflow as tf
import voluptuous

__all__ = [
    "decode",
    "tensor_default_summaries",
    "squggle",
    "named_tuple_helper",
]


def decode(x):
    return "".join(map(dataset_pb2.BasePair.Name, x))


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


def tensor_default_summaries(name, tensor) -> List[tf.Summary]:
    mean, var = tf.nn.moments(tensor, axes=list(range(len(tensor.shape))))
    return [
        tf.summary.scalar(name + '/mean', mean),
        tf.summary.scalar(name + '/stddev', tf.sqrt(var)),
        tf.summary.scalar(name + '/max', tf.reduce_max(tensor)),
        tf.summary.scalar(name + '/min', tf.reduce_min(tensor)),
        tf.summary.histogram(name + '/histogram', tensor),
    ]


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
