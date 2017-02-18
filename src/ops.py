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

                filters = np.zeros([size, out_classes, out_classes], dtype=np.float32)
                for i in range(out_classes):
                    filters[:, i, i] = 1

                reg = tf.nn.conv1d(net, filters, 1, 'VALID')
                reg = tf.exp(reg)  # Likelihood of having size consecutive symbols on output
                reg = tf.reduce_sum(reg, axis=[2])  # Sum over all symbols
                reg = tf.reduce_mean(reg)  # and find mean per sequence per batch
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
        )
    ]
    ):
        cut_size = tf.div(cut_size, 2)

    net = tf.slice(net, [0, cut_size, 0],
                   [-1, tf.div(block_size, shrink_factor), -1], name="Cutting")

    net.set_shape(output_shape)
    return net


if __name__ == '__main__':
    from model_small import model
    print(read_model_vars(model))
