import tensorflow as tf
import numpy as np
import tflearn


def running_mean(net, sizes, penalties, out_classes, name=None):
    """
        Args:
        net -> (max_len, batch_size, num_classes)
        sizes -> sizes of running mean window
        penalties -> penalty per size
        out_classes -> number of output classes
    """
    with tf.name_scope("running_mean"):
        net = tf.transpose(net, [1, 0, 2])
        net = tf.nn.log_softmax(net)
        # net.set_shape([4, 2, 5])
        # (batch_size, max_len, num_classes)
        out = []

        for size, penalty in zip(sizes, penalties):
            with tf.name_scope("size_%d" % size):

                filters = np.zeros(
                    [size, out_classes, out_classes], dtype=np.float32)
                for i in range(out_classes):
                    filters[:, i, i] = 1

                reg = tf.nn.conv1d(net, filters, 1, 'VALID')
                reg = tf.exp(
                    reg
                )  # Likelihood of having size consecutive symbols on output
                reg = tf.reduce_sum(reg, axis=[2])  # Sum over all symbols
                reg = tf.reduce_mean(
                    reg)  # and find mean per sequence per batch
                out.append(penalty * reg)

        return tf.reduce_sum(out)


def read_model_vars(model, *args, **kwargs):
    try:
        model.init_session()
        model.restore(*args, **kwargs)
        sol = {}
        with model.g.as_default():
            vars = tf.trainable_variables()
            for var, val in zip(vars, model.sess.run(vars)):
                sol[var.name] = val
            return sol
    finally:
        model.close_session()


def __running_mean_test():
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32)
        net = running_mean(x, [1], [1], 5)

        # batch_size = 2
        # max_len = 4
        # num_classes = 5

        tin = np.arange(4 * 2 * 5).reshape([4, 2, 5])
        print(tin)
        print(sess.run(net, feed_dict={x: tin}))


def central_cut(net, block_size, shrink_factor):
    output_shape = net.get_shape().as_list()
    output_shape[1] = None
    cut_size = tf.shape(net)[1] - tf.div(block_size, shrink_factor)
    with tf.control_dependencies([
        tf.cond(
            tflearn.get_training_mode(),
            lambda: tf.assert_equal(
                tf.mod(cut_size, 2), 0, name="cut_size_assert"),
            lambda: tf.no_op()
        ),
        tf.assert_non_negative(cut_size)
    ]
    ):
        cut_size = tf.div(cut_size, 2)

    net = tf.slice(
        net, [0, cut_size, 0], [-1, tf.div(block_size, shrink_factor), -1],
        name="Cutting")

    net.set_shape(output_shape)
    return net


def atrous_conv1d(value, filters, rate, padding="SAME", name=None):
    with tf.name_scope(name, "atrous_conv1d", [value, filters]) as name:
        value = tf.convert_to_tensor(value, name="value")
        filters = tf.convert_to_tensor(filters, name="filters")

        if rate == 1:
            return tf.nn.conv1d(value, filters, 1, padding)

        if value.get_shape().is_fully_defined():
            value_shape = value.get_shape().as_list()
        else:
            value_shape = tf.shape(value)

        add = (-value_shape[1] % rate + rate) % rate
        pad = [[0, add]]
        crop = [[0, add]]

        value = tf.space_to_batch_nd(
            input=value, paddings=pad, block_shape=[rate])

        value = tf.nn.conv1d(value, filters, 1, padding, name=name)

        value = tf.batch_to_space_nd(
            input=value, crops=crop, block_shape=[rate])

        return value


def dense2d_to_sparse(dense_input, length, name=None, dtype=None):
    with tf.name_scope(name, "dense2d_to_sparse"):
        num_batches = dense_input.get_shape()[0]

        indices = [
            tf.stack([tf.fill([length[x]], x),
                      tf.range(length[x])], axis=1) for x in range(num_batches)
        ]
        indices = tf.concat(axis=0, values=indices)
        indices = tf.to_int64(indices)

        values = [
            tf.squeeze(
                tf.slice(dense_input, [x, 0], [1, length[x]]), axis=[0])
            for x in range(num_batches)
        ]
        values = tf.concat(axis=0, values=values)

        if dtype is not None:
            values = tf.cast(values, dtype)

        return tf.SparseTensor(indices, values,
                               tf.to_int64(tf.shape(dense_input)))


# (2) use SELUs
def selu(x):
    with tf.name_scope('selu'):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


# (3) initialize weights with stddev sqrt(1/n)
# e.g. use:
# selu_initializer = tf.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')

if __name__ == '__main__':
    from model_small import model
    print(read_model_vars(model))

    X = tf.constant(
        np.array([1, 2, 3, 4, 5, 6, 7]).reshape(1, 7, 1), dtype=tf.float32)
    kernel = tf.constant(
        np.array([100, 10, 1]).reshape(3, 1, 1), dtype=tf.float32)
    y = atrous_conv1d(X, kernel, 2, "SAME")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    gg = sess.run(y)
    print(gg, gg.shape)
