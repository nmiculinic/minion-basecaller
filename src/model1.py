import tensorflow as tf
import numpy as np
from util import atrous_conv1d
from sklearn.preprocessing import LabelEncoder, StandardScaler

max_reach = 32  # How many extra elements I have to fetch for convolutions
state_size = 50  # For RNN
out_classes = 5  # A,G,T,C plus LAST state for blank. Last due to CTC implementation

# For slicing input (due to data locality it's more efficient to keep as much data as possible on GPU, thus slicing):
begin = tf.placeholder(dtype=tf.int32, shape=[], name="begin")
length = 50  # Fixed due to dynamic_rnn

# TODO, once I have data pipeline in place

batch_size = 16

X = tf.placeholder(tf.float32, shape=[batch_size, 5000, 3], name="X")
X_len = tf.placeholder(tf.int32, shape=[batch_size], name="X_len")
Y = tf.sparse_placeholder(tf.int32, shape=[batch_size, length], name="Y") # Won't cut for now
Y_len = tf.placeholder(tf.int32, shape=[batch_size], name="Y_len")

net = X
with tf.control_dependencies([
    tf.assert_less_equal(begin + length, tf.shape(X)[1], message="Cannot request that many elements from X"),
    tf.assert_non_negative(begin, message="Beginning slice must be >=0")
]):
    left = tf.maximum(0, begin - max_reach)
    right = tf.minimum(tf.shape(X)[1], begin + length + max_reach)

net = tf.slice(X, [0, left, 0], [-1, right - left, -1])
net.set_shape([None, None, 3])
for i, no_channel in zip([2, 4, 8, 16], [8, 16, 16, 16]):
    with tf.variable_scope("atrous_conv1d_%d" % i):
        print(net.get_shape())
        filter = tf.get_variable("W", shape=(3, net.get_shape()[-1], no_channel))
        bias = tf.get_variable("b", shape=(no_channel,))
        net = atrous_conv1d(net, filter, i) + bias
        net = tf.nn.relu(net)
net = tf.slice(net, [0, begin - left, 0], [-1, length, -1])

with tf.name_scope("RNN"):
    cell = tf.nn.rnn_cell.GRUCell(state_size)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    seq_len = tf.maximum(0, X_len - begin)
    outputs, final_state = tf.nn.dynamic_rnn(cell, net, initial_state=init_state, sequence_length=seq_len)

with tf.variable_scope("Output"):
    outputs = tf.reshape(outputs, [-1, state_size])
    W = tf.get_variable("W", shape=[state_size, out_classes])
    b = tf.get_variable("b", shape=[out_classes])
    outputs = tf.matmul(outputs, W) + b
    logits = tf.reshape(outputs, [batch_size, length, out_classes])

# yy = tf.slice(Y, [0, begin], [-1, length])  ##Damn sparse vs dense matrix.
yy_len = tf.maximum(0, Y_len - begin)

print(logits.get_shape())

loss = tf.nn.ctc_loss(logits, Y, yy_len, preprocess_collapse_repeated=False, ctc_merge_repeated=False)

predicted, predicted_logprob = tf.nn.ctc_beam_search_decoder(tf.transpose(logits, [1, 0, 2]), yy_len, merge_repeated=False)

pred = tf.cast(predicted[0], tf.int32)

optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)
grads = optimizer.compute_gradients(loss)

if __name__ == "__main__":
    # ds = np.load('/home/lpp/Desktop/dataset.npz')
    # x_train, y_train = ds['X'], ds['Y']
    x_train_len = np.repeat(5000, 100)
    y_train_len = np.repeat(5000, 100)

    # label_encoder = LabelEncoder()
    # label_encoder.classes_ = np.array(['A', 'G', 'T', 'C', 'N'])

    # y_train = label_encoder.transform(y_train.reshape(-1)).reshape(-1, 5000)

    # preprocess_x = StandardScaler()
    # x_train = preprocess_x.fit_transform(x_train.reshape(-1, 5000 * 3)).reshape(-1, 5000, 3)

    x_train = np.random.uniform(shape=[100, 5000, 3])
    y_train = np.random.randint(5, shape=[100, 5000])


    # Testing stuff
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    gg = sess.run(logits, feed_dict=
                  {
                  X: x_train[:32],
                  begin: 0,
                  X_len: x_train_len[:32],
                  Y: y_train[:32],
                  Y_len:y_train_len[:32]
                  })
    print(gg, gg.shape)
