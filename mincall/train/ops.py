import tensorflow as tf
import pandas as pd
from collections import defaultdict
import numpy as np
from minion_data import dataset_pb2
from mincall.common import *
import edlib
import logging
from pprint import pformat
import random

aligment_stats_ordering = [
    dataset_pb2.MATCH, dataset_pb2.MISMATCH, dataset_pb2.INSERTION,
    dataset_pb2.DELETION
]


def alignment_stats(
    lable_ind, label_val, pred_ind, pred_val, batch_size, debug=False
):
    yt = defaultdict(list)
    for ind, val in zip(lable_ind, label_val):
        yt[ind[0]].append(val)

    yp = defaultdict(list)
    for ind, val in zip(pred_ind, pred_val):
        yp[ind[0]].append(val)

    sol = defaultdict(list)
    for x in range(batch_size):
        query = decode(yp[x])
        target = decode(yt[x])
        if len(target) == 0:
            raise ValueError("Empty target sequence")
        edlib_res = edlib.align(query, target, task='path')
        stats = ext_cigar_stats(edlib_res['cigar'])

        read_len = stats[dataset_pb2.MISMATCH
                        ] + stats[dataset_pb2.MATCH
                                 ] + stats[dataset_pb2.DELETION]

        for op in aligment_stats_ordering:
            sol[op].append(stats[op] / read_len)
        if x < 5:
            msg = "edlib results\n"
            msg += "query:  " + query + "\n"
            msg += "target: " + target + "\n"
            msg += "cigar:  " + edlib_res['cigar'] + "\n"
            msg += pformat({dataset_pb2.Cigar.Name(k): v
                            for k, v in stats.items()}) + "\n"
            msg += "readl:  " + str(read_len) + "\n"
            msg += "==================\n"
            logging.info(msg)
    sol = [
        np.array(sol[op], dtype=np.float32) for op in aligment_stats_ordering
    ]
    sol_data = {
        dataset_pb2.Cigar.Name(k): v for k, v in zip(aligment_stats_ordering, sol)
    }
    logging.info(f"sol: \n{pd.DataFrame(sol_data)}")
    return sol


if __name__ == "__main__":
    labels = tf.sparse_placeholder(tf.float32, name="labels")
    pred = tf.sparse_placeholder(tf.float32, name="pred")
    stats = tf.py_func(
        alignment_stats,
        [
            labels.indices, labels.values, pred.indices, pred.values,
            tf.constant(5)
        ],
        4 * [tf.float32],
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
