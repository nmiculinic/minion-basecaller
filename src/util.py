import tensorflow as tf
import numpy as np


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

        value = tf.space_to_batch_nd(input=value,
                                     paddings=pad,
                                     block_shape=[rate])

        value = tf.nn.conv1d(value, filters, 1, padding, name=name)

        value = tf.batch_to_space_nd(input=value,
                                     crops=crop,
                                     block_shape=[rate])

        return value


if __name__ == "__main__":
    X = tf.constant(np.array([1, 2, 3, 4, 5, 6, 7]).reshape(1, 7, 1), dtype=tf.float32)
    kernel = tf.constant(np.array([100, 10, 1]).reshape(3, 1, 1), dtype=tf.float32)
    y = atrous_conv1d(X, kernel, 2, "SAME")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    gg = sess.run(y)
    print(gg, gg.shape)
