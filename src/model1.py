import tensorflow as tf
import numpy as np
from util import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import sparse
import sys
import os
from tensorflow.python.client import timeline
import time

max_reach = 32  # How many extra elements I have to fetch for convolutions
state_size = 50  # For RNN
out_classes = 4 + 4 + 1  # A,G,T,C plus LAST state for blank. Last due to CTC implementation
trace_level = tf.RunOptions.NO_TRACE

# For slicing input (due to data locality it's more efficient to keep as much data as possible on GPU, thus slicing):


with tf.variable_scope("input"):
    block_size = 25   # Training block size
    num_blocks = 2
    batch_size = 10

    input_vars = [
        tf.get_variable("X", initializer=tf.zeros_initializer([batch_size, block_size * num_blocks, 3], tf.float32), trainable=False),
        tf.get_variable("X_len", initializer=tf.zeros_initializer([batch_size], tf.int32), trainable=False),
        tf.get_variable("Y", initializer=tf.zeros_initializer([batch_size, block_size * num_blocks], tf.uint8), trainable=False),
        tf.get_variable("Y_len", initializer=tf.zeros_initializer([batch_size, num_blocks], tf.int32), trainable=False),
    ]
    order = [x.name[6:-2] for x in input_vars]
    shapes = [x.get_shape()[1:] for x in input_vars]
    types = [x.dtype.base_dtype for x in input_vars]
    queue = tf.RandomShuffleQueue(5 * batch_size, 2 * batch_size, types, shapes=shapes)

    input_var_dict = {}
    for x, qx in zip(input_vars, queue.dequeue_many(batch_size)):
        name = x.name[6:-2]
        input_var_dict[name] = x
        feed = tf.placeholder_with_default(qx, shape=x.get_shape(), name=name + "_feed")
        input_var_dict[name + "_feed"] = feed
        input_var_dict[name + "_assign"] = tf.assign(x, feed, name=name + "_assign")
        print(x.dtype)
        input_var_dict[name + "_enqueue_val"] = tf.placeholder(x.dtype.base_dtype, x.get_shape(), name=name + "_enqueue_val")

    enqueue_op = queue.enqueue_many([input_var_dict[name + "_enqueue_val"] for name in order])

    print(*[v for k, v in input_var_dict.items() if '_assign' in k])
    input_var_dict['load_queue'] = tf.group(*[v for k, v in input_var_dict.items() if '_assign' in k])

    block_idx = tf.placeholder(dtype=tf.int32, shape=[], name="block_idx")
    begin = block_idx * block_size

with tf.name_scope("model"):

    X = input_var_dict['X']
    with tf.control_dependencies([
        tf.assert_less_equal(begin + block_size, tf.shape(X)[1], message="Cannot request that many elements from X"),
        tf.assert_non_negative(begin, message="Beginning slice must be >=0"),
    ]):
        left = tf.maximum(0, begin - max_reach)
        right = tf.minimum(tf.shape(X)[1], begin + block_size + max_reach)
        X_len = tf.clip_by_value(input_var_dict['X_len'] - block_idx * block_size, 0, block_size)

        Y_len = tf.squeeze(tf.slice(input_var_dict['Y_len'], [0, block_idx], [batch_size, 1]), [1])
        Y = dense2d_to_sparse(tf.slice(input_var_dict['Y'], [0, begin], [batch_size, block_size]), Y_len, dtype=tf.int32)


    net = tf.slice(X, [0, left, 0], [batch_size, right - left, -1])
    net.set_shape([batch_size, None, 3])
    for i, no_channel in zip([], [32, 64, 128, 256, 512]):
        with tf.variable_scope("atrous_conv1d_%d" % i):
            filter = tf.get_variable("W", shape=(3, net.get_shape()[-1], no_channel))
            bias = tf.get_variable("b", shape=(no_channel,))
            net = atrous_conv1d(net, filter, i) + bias
            net = tf.nn.relu(net)
    net = tf.slice(net, [0, begin - left, 0], [-1, block_size, -1])
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

    print("logits: ", logits.get_shape())

    loss = tf.nn.ctc_loss(inputs=logits, labels=Y, sequence_length=X_len, time_major=True)
    loss = tf.reduce_mean(loss)

    predicted, prdicted_logprob = tf.nn.ctc_beam_search_decoder(logits, X_len, merge_repeated=True, top_paths=1)
    pred = tf.sparse_tensor_to_dense(tf.cast(predicted[0], tf.int32))

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)
    grads = optimizer.compute_gradients(loss)

if __name__ == "__main__":

    ds = np.load(os.path.expanduser('~/dataset.npz'))
    keys = list(ds.keys())

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        batch_time = 0
        for i in range(10001):
            if i == 0:
                for _ in range(3):
                    feed = {
                            input_var_dict[name + "_enqueue_val"]: ds[name] for name in keys
                    }
                    feed[input_var_dict['X_len_enqueue_val']] = np.array([500] * batch_size)
                    sess.run(enqueue_op, feed_dict=feed)

                sess.run(input_var_dict['load_queue'])

                perm = np.arange(batch_size)
                # perm = np.random.permutation(batch_size)
                # feed = {
                #     input_var_dict[x + "_feed"]: ds[x][:batch_size][perm] for x in keys
                # }
                # feed[input_var_dict["X_len_feed"]] = np.array([500] * batch_size)
                # sess.run([input_var_dict[x + "_assign"] for x in keys], feed_dict=feed)

            def print_d(idx):
                yy, yy_len = sess.run([input_vars[2], input_vars[3]])
                print("%13sTarget:" % "", decode_example(yy[idx], yy_len[idx], num_blocks, block_size))

            tt = time.clock()
            state = sess.run(init_state)
            for blk in range(num_blocks):
                run_metadata = tf.RunMetadata()
                loss_val, _, state = sess.run([loss, train_op, final_state], feed_dict={
                    block_idx: blk,
                    init_state: state
                }, options=tf.RunOptions(trace_level=trace_level),
                run_metadata=run_metadata)

                if trace_level > tf.RunOptions.NO_TRACE:
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    trace_file = open('timeline.ctf_loss.json', 'w')
                    trace_file.write(trace.generate_chrome_trace_format())
            batch_time = 0.8*batch_time + 0.2 * (time.clock() - tt)
            if (i % 20 == 0):
                print("avg time per batch %.3f" % batch_time)
                state = sess.run(init_state)
                gg = []
                for blk in range(num_blocks):
                    run_metadata = tf.RunMetadata()

                    ff, state = sess.run([pred, final_state], feed_dict={
                        block_idx: blk,
                        init_state: state
                    }, options=tf.RunOptions(trace_level=trace_level),
                    run_metadata=run_metadata)
                    gg.append("".join([str(x) for x in decode(ff[0].ravel())]))

                    if trace_level > tf.RunOptions.NO_TRACE:
                        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                        trace_file = open('timeline.ctf_decode.json', 'w')
                        trace_file.write(trace.generate_chrome_trace_format())

                print("%4d %6.3f" % (i, np.sum(loss_val)), "decoded:", gg)
                print_d(0)

                # Refresh queue a bit
                feed = {
                        input_var_dict[name + "_enqueue_val"]: ds[name] for name in keys
                }
                feed[input_var_dict['X_len_enqueue_val']] = np.array([500] * batch_size)
                sess.run(enqueue_op, feed_dict=feed)
                sess.run(input_var_dict['load_queue'])

                # if np.sum(loss_val) < 0.1:
                    # break

    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
