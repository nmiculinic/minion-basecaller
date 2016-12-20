import tensorflow as tf
import numpy as np
from util import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import sparse
import sys
import os


max_reach = 32  # How many extra elements I have to fetch for convolutions
state_size = 50  # For RNN
out_classes = 4 + 4 + 1  # A,G,T,C plus LAST state for blank. Last due to CTC implementation

# For slicing input (due to data locality it's more efficient to keep as much data as possible on GPU, thus slicing):


with tf.variable_scope("input"):
    block_size = 25   # Training block size
    num_blocks = 2
    batch_size = 1
    num_examples = 10

    input_vars = [
        tf.get_variable("X", initializer=tf.zeros_initializer([num_examples, block_size * num_blocks, 3], tf.float32), trainable=False),
        tf.get_variable("Y", initializer=tf.zeros_initializer([num_examples, block_size * num_blocks], tf.uint8), trainable=False),
        tf.get_variable("X_len", initializer=tf.zeros_initializer([num_examples, num_blocks], tf.int32), trainable=False),
        tf.get_variable("Y_len", initializer=tf.zeros_initializer([num_examples, num_blocks], tf.int32), trainable=False),
    ]
    input_var_dict = {}
    for x in input_vars:
        name = x.name[6:-2]
        input_var_dict[name] = x
        feed = tf.placeholder(x.dtype, shape=x.get_shape(), name=name + "_feed")
        input_var_dict[name + "_feed"] = feed
        input_var_dict[name + "_assign"] = tf.assign(x, feed, name=name + "_assign")

    block_idx = tf.placeholder(dtype=tf.int32, shape=[], name="block_idx")
    start_batch_idx = tf.placeholder(dtype=tf.int32, shape=[], name="start_batch_idx")
    begin = block_idx * block_size

X = input_var_dict['X']
with tf.control_dependencies([
    tf.assert_less_equal(begin + block_size, tf.shape(X)[1], message="Cannot request that many elements from X"),
    tf.assert_non_negative(begin, message="Beginning slice must be >=0"),
    tf.assert_non_negative(start_batch_idx, message="Start batch idx must be >=0"),
    tf.assert_less_equal(start_batch_idx + batch_size, tf.shape(X)[0], message="end batch out of bounds")
]):
    left = tf.maximum(0, begin - max_reach)
    right = tf.minimum(tf.shape(X)[1], begin + block_size + max_reach)
    X_len = tf.squeeze(tf.slice(input_var_dict['X_len'], [start_batch_idx, block_idx], [batch_size, 1]), [0])

    Y_len = tf.squeeze(tf.slice(input_var_dict['Y_len'], [start_batch_idx, block_idx], [batch_size, 1]), [0])
    Y = dense2d_to_sparse(tf.slice(input_var_dict['Y'], [start_batch_idx, begin], [batch_size, block_size]), Y_len, dtype=tf.int32)


net = tf.slice(X, [start_batch_idx, left, 0], [batch_size, right - left, -1])
net.set_shape([batch_size, None, 3])
for i, no_channel in zip([1], [8, 16, 16, 16]):
    with tf.variable_scope("atrous_conv1d_%d" % i):
        print(net.get_shape())
        filter = tf.get_variable("W", shape=(3, net.get_shape()[-1], no_channel))
        bias = tf.get_variable("b", shape=(no_channel,))
        net = atrous_conv1d(net, filter, i) + bias
        net = tf.nn.relu(net)
net = tf.slice(net, [0, begin - left, 0], [-1, block_size, -1])

with tf.name_scope("RNN"):
    cell = tf.nn.rnn_cell.GRUCell(state_size)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, net, initial_state=init_state, sequence_length=X_len)

with tf.variable_scope("Output"):
    outputs = tf.reshape(outputs, [-1, state_size])
    W = tf.get_variable("W", shape=[state_size, out_classes])
    b = tf.get_variable("b", shape=[out_classes])
    outputs = tf.matmul(outputs, W) + b
    logits = tf.reshape(outputs, [batch_size, block_size, out_classes])

print("logits: ", logits.get_shape())
logits_time_major = tf.transpose(logits, [1, 0, 2])

loss = tf.nn.ctc_loss(inputs=logits_time_major, labels=Y, sequence_length=Y_len, time_major=True)

loss = tf.reduce_mean(loss)

predicted, prdicted_logprob = tf.nn.ctc_beam_search_decoder(logits_time_major, Y_len, merge_repeated=True)

pred = tf.sparse_tensor_to_dense(tf.cast(predicted[0], tf.int32))

optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)
grads = optimizer.compute_gradients(loss)

if __name__ == "__main__":

    ds = np.load(os.path.expanduser('~/dataset.npz'))
    keys = list(ds.keys())

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    perm = np.random.permutation(num_examples)
    feed = {
        input_var_dict[x + "_feed"]: ds[x][perm] for x in keys
    }

    sess.run([input_var_dict[x + "_assign"] for x in keys], feed_dict=feed)

    print("Target: ", decode_example(ds['Y'][perm][0], ds['Y_len'][perm][0], num_blocks, block_size))

    for i in range(1001):
        loss_val = 0
        state = sess.run(init_state)
        for blk in range(num_blocks):
            loss_val, _, state = sess.run([loss, train_op, final_state], feed_dict={
                block_idx: blk,
                start_batch_idx: 0,
                init_state: state
            })
        if (i % 20 == 0):
            state = sess.run(init_state)
            gg = []
            for blk in range(num_blocks):
                ff, state = sess.run([pred, final_state], feed_dict={
                    block_idx: blk,
                    start_batch_idx: 0,
                    init_state: state
                })
                gg.append("".join([str(x) for x in decode(ff.ravel())]))

            print("%4d %.3f" % (i, np.sum(loss_val)), "decoded:", gg)
            print("Target: ", decode_example(ds['Y'][perm][0], ds['Y_len'][perm][0], num_blocks, block_size))

            if np.sum(loss_val) < 0.1:
                break
