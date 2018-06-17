import tensorflow as tf
import uuid
import os
import pandas as pd
from collections import defaultdict
import numpy as np
from minion_data import dataset_pb2
from mincall.common import *
import edlib
import logging
from pprint import pformat
import cytoolz as toolz
import random

logger = logging.getLogger(__name__)

aligment_stats_ordering = [
    dataset_pb2.MATCH, dataset_pb2.MISMATCH, dataset_pb2.INSERTION,
    dataset_pb2.DELETION
]


def alignment_stats(
    lable_ind, label_val, pred_ind, pred_val, batch_size, debug=False
):
    """Returns a list of numpy array representing alignemnt stats. First N elements are
    in aligment_stats_ordering and the last one in identity.

    The return is like this due to tf.py_func requirements --> this function is made for
    embedding as tf operation via tf.py_func

    :param lable_ind:
    :param label_val:
    :param pred_ind:
    :param pred_val:
    :param batch_size:
    :param debug:
    :return:
    """

    prefix = os.environ.get("MINCALL_LOG_DATA", None)
    if prefix:
        fname = os.path.abspath(os.path.join(prefix, f"{uuid.uuid4().hex}.npz"))
        with open(fname, "wb") as f:
            np.savez(f, **{
                "label_val": label_val,
                "lable_ind": lable_ind,
                "pred_val": pred_val,
                "pred_ind": pred_ind,
                "batch_size": batch_size,
            })
        logger.debug(f"Saves alignment stats input data to {fname}")

    yt = defaultdict(list)
    for ind, val in zip(lable_ind, label_val):
        yt[ind[0]].append(val)

    yp = defaultdict(list)
    for ind, val in zip(pred_ind, pred_val):
        yp[ind[0]].append(val)

    sol = defaultdict(list)
    identities = []
    for x in range(batch_size):
        query = decode(yp[x])
        target = decode(yt[x])
        if len(target) == 0:
            raise ValueError("Empty target sequence")
        if len(query) == 0:
            logger.warning(f"Empty query sequence\n"
                           f"Target: {target}")
            continue
        edlib_res = edlib.align(query, target, task='path')
        stats = ext_cigar_stats(edlib_res['cigar'])

        read_len = stats[dataset_pb2.MISMATCH
                        ] + stats[dataset_pb2.MATCH
                                 ] + stats[dataset_pb2.INSERTION]

        #  https://github.com/isovic/samscripts/blob/master/src/errorrates.py
        identities.append(stats[dataset_pb2.MATCH] / sum(stats.values()))

        for op in aligment_stats_ordering:
            sol[op].append(stats[op] / read_len)
        if True:
            msg = "edlib results\n"
            s_query, s_target, _ = squggle(query, target)
            exp_cigar = expand_cigar(edlib_res['cigar'])

            for i in range(0, len(s_query), 80):
                msg += "query:  " + s_query[i:i + 80] + "\n"
                msg += "target: " + s_target[i:i + 80] + "\n"
                msg += "cigar : " + exp_cigar[i:i + 80] + "\n"
                msg += "--------" + 80 * "-" + "\n"

            msg += "query:  " + query + "\n"
            msg += "target: " + target + "\n"
            msg += "full cigar:  " + edlib_res['cigar'] + "\n"
            msg += pformat({
                dataset_pb2.Cigar.Name(k): v
                for k, v in stats.items()
            }) + "\n"
            msg += "readl:  " + str(read_len) + "\n"
            df = pd.DataFrame({
                "query":
                    toolz.merge(
                        toolz.frequencies(query),
                        toolz.keymap(
                            "".join,
                            toolz.frequencies(toolz.sliding_window(2, query))
                        ),
                    ),
                "target":
                    toolz.merge(
                        toolz.frequencies(target),
                        toolz.keymap(
                            "".join,
                            toolz.frequencies(toolz.sliding_window(2, target))
                        ),
                    ),
            })
            df["delta"] = 100 * (df['target'] / df['query'] - 1)
            df = df[['query', 'target', 'delta']]
            msg += "Stats\n" + str(df) + "\n"
            msg += "==================\n"
            logging.info(msg)
    sol = [
        np.array(sol[op], dtype=np.float32) for op in aligment_stats_ordering
    ]
    sol_data = {
        dataset_pb2.Cigar.Name(k): v
        for k, v in zip(aligment_stats_ordering, sol)
    }
    sol_data["IDENTITY"] = identities
    logging.info(f"sol: \n{pd.DataFrame(sol_data)}")
    return sol + [np.array(identities, dtype=np.float32)]


if __name__ == "__main__":
    labels = tf.sparse_placeholder(tf.float32, name="labels")
    pred = tf.sparse_placeholder(tf.float32, name="pred")
    *stats, _ = tf.py_func(
        alignment_stats,
        [
            labels.indices, labels.values, pred.indices, pred.values,
            tf.constant(5)
        ],
        5 * [tf.float32],
        stateful=False,
    )
    match, mismatch, insertion, deletion = stats

    with tf.Session() as sess:
        sol = sess.run(
            stats,
            feed_dict={
                pred:
                    tf.SparseTensorValue(
                        indices=np.array([[0, 0], [1, 0], [1, 1], [2, 0],
                                          [3, 0], [3, 1], [4, 0]],
                                         dtype=np.int64),
                        values=np.array([0, 0, 1, 2, 1, 1, 3],
                                        dtype=np.float32),
                        dense_shape=np.array([1, 5], dtype=np.int64),
                    ),
                labels:
                    tf.SparseTensorValue(
                        indices=np.array([[0, 0], [1, 0], [1, 1], [2, 0],
                                          [2, 1], [3, 0], [4, 0]],
                                         dtype=np.int64),
                        values=np.array([0, 0, 0, 2, 2, 1, 3],
                                        dtype=np.float32),
                        dense_shape=np.array([1, 5], dtype=np.int64),
                    ),
            }
        )
        df = pd.DataFrame({
            dataset_pb2.Cigar.Name(x): sol[i]
            for i, x in enumerate(aligment_stats_ordering)
        })
        print(df)
