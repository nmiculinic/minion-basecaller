import tensorflow as tf
import numpy as np
from util import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import sparse
import sys
import os
from tensorflow.python.client import timeline
import time
import multiprocessing
from threading import Thread
import input_readers
import model_utils

trace_level = tf.RunOptions.NO_TRACE

def model_fn(net, X_len, max_reach, block_size, out_classes, batch_size, reuse=False, **kwargs):
    """
        Args:
        net -> Input tensor shaped (batch_size, max_reach + block_size + max_reach, 3)
        Returns:
        logits -> Unscaled logits tensor in time_major form, (block_size, batch_size, out_classes)
    """
    state_size = 50  # For RNN

    print("model in", net.get_shape())
    with tf.name_scope("model"):
        for i, no_channel in zip([1, 2], [32, 64, 128, 256, 512]):
            with tf.variable_scope("atrous_conv1d_%d" % i):
                tf.Print(net, [tf.shape(net)], message="during convs")
                filter = tf.get_variable("W", shape=(3, net.get_shape()[-1], no_channel))
                bias = tf.get_variable("b", shape=(no_channel,))
                net = atrous_conv1d(net, filter, i, padding="VALID") + bias
                net = tf.nn.relu(net)
        tf.Print(net, [tf.shape(net)], message="After convs")
        print("after conv", net.get_shape())
        # net = tf.slice(net, [0, 0, 0], [-1, block_size, -1])
        net = tf.transpose(net, [1, 0, 2], name="Shift_to_time_major")
        with tf.name_scope("RNN"):
            cell = tf.nn.rnn_cell.GRUCell(state_size)
            init_state = cell.zero_state(batch_size, dtype=tf.float32)
            outputs, final_state = tf.nn.dynamic_rnn(cell, net, initial_state=init_state, sequence_length=X_len, time_major=True)

        with tf.variable_scope("Output"):
            outputs = tf.reshape(outputs, [-1, state_size])
            W = tf.get_variable("W", shape=[state_size, out_classes])
            b = tf.get_variable("b", shape=[out_classes])
            outputs = tf.matmul(outputs, W) + b
            logits = tf.reshape(outputs, [block_size, batch_size, out_classes])
    print("model out", logits.get_shape())
    return {
        'logits': logits,
        'init_state': init_state,
        'final_state': final_state
    }


model = model_utils.Model(
    tf.get_default_graph(),
    block_size=50,
    num_blocks=2,
    batch_size=16,
    max_reach=3,
    model_fn=model_fn
)
model.init_session()
model.train_minibatch()
print("closing session")
model.close_session()

print("finishing")
sys.exit(0)


# if __name__ == "__main__":
#     try:
#         batch_time = 0
#         for i in range(10001):
#
#             def print_d(idx):
#                 yy, yy_len = sess.run([input_vars[2], input_vars[3]])
#                 print("%13sTarget:" % "", decode_example(yy[idx], yy_len[idx], num_blocks, block_size))
#
#             tt = time.clock()
#             state = sess.run(init_state)
#             for blk in range(num_blocks):
#                 run_metadata = tf.RunMetadata()
#                 loss_val, _, state = sess.run([loss, train_op, final_state], feed_dict={
#                     block_idx: blk,
#                     init_state: state
#                 }, options=tf.RunOptions(trace_level=trace_level),
#                 run_metadata=run_metadata)
#
#                 if trace_level > tf.RunOptions.NO_TRACE:
#                     trace = timeline.Timeline(step_stats=run_metadata.step_stats)
#                     trace_file = open('timeline.ctf_loss.json', 'w')
#                     trace_file.write(trace.generate_chrome_trace_format())
#             batch_time = 0.8*batch_time + 0.2 * (time.clock() - tt)
#             if (i % 20 == 0):
#                 print("avg time per batch %.3f" % batch_time)
#                 state = sess.run(init_state)
#                 gg = []
#                 for blk in range(num_blocks):
#                     run_metadata = tf.RunMetadata()
#
#                     ff, state = sess.run([pred, final_state], feed_dict={
#                         block_idx: blk,
#                         init_state: state
#                     }, options=tf.RunOptions(trace_level=trace_level),
#                     run_metadata=run_metadata)
#                     gg.append("".join([str(x) for x in decode(ff[0].ravel())]))
#
#                     if trace_level > tf.RunOptions.NO_TRACE:
#                         trace = timeline.Timeline(step_stats=run_metadata.step_stats)
#                         trace_file = open('timeline.ctf_decode.json', 'w')
#                         trace_file.write(trace.generate_chrome_trace_format())
#
#                 print("%4d %6.3f" % (i, np.sum(loss_val)), "decoded:", gg)
#                 print_d(0)
#                 t0 = time.clock()
#                 print("loading_time %.3f" % (time.clock() - t0))
