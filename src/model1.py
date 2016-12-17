import tensorflow as tf
import numpy as np
from util import atrous_conv1d

sess = tf.Session()


batch_size = 2
begin = tf.placeholder(dtype=tf.int32, shape=[], name="begin")
length = 3  # Fixed due to dynamic_rnn
max_reach = 2  # How many extra elements I have to fetch for convolutions

X = tf.placeholder(tf.float32, shape=(batch_size, None, 1), name="X")
X_len = tf.placeholder(tf.int32, shape=(batch_size,), name="X_len")
Y = tf.placeholder(tf.float32, shape=(batch_size, 1), name="Y")
Y_len = tf.placeholder(tf.int32, shape=(batch_size,), name="Y_len")
filter = np.array([100, 10, 1.0], dtype=np.float32).reshape(3, 1, 1)

net = X
with tf.control_dependencies([
    tf.assert_less_equal(begin + length, tf.shape(X)[1], message="Cannot request that many elements from X"),
    # tf.assert_less_equal(begin + length, tf.shape(Y)[1], message="Cannot request that many elements from Y"),
    tf.assert_non_negative(begin, message="Beginning slice must be >=0")
]):
    left = tf.maximum(0, begin - max_reach)
    right = tf.minimum(tf.shape(X)[1], begin + length + max_reach)

net = tf.slice(X, [0, left, 0], [-1, right - left, -1])
net = atrous_conv1d(net, filter, 2)
net = tf.slice(net, [0, begin - left, 0], [-1, length, -1])


cell = tf.nn.rnn_cell.GRUCell(50)
init_state = cell.zero_state(batch_size, dtype=tf.float32)
print(init_state.get_shape())
seq_len = tf.maximum(0, X_len - begin)
print("net", net.get_shape())
net.set_shape([batch_size, length, 1])
print("net", net.get_shape())
print("seq_len", seq_len.get_shape())
outputs, final_state = tf.nn.dynamic_rnn(cell, net, initial_state=init_state, sequence_length=seq_len)

# print(outputs.get_shape())

# loss = tf.nn.seq2seq.sequence_loss_by_example(outputs, targets, weights)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
gg = sess.run(net, feed_dict={X: 1 + np.arange(14).reshape(2, 7, 1), begin:4, X_len:[3,4]})
print(gg, gg.shape)
