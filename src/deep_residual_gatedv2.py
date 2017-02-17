import tensorflow as tf
from tflearn.layers.conv import max_pool_1d, conv_1d
import os
from tflearn.layers.normalization import batch_normalization
from dotenv import load_dotenv, find_dotenv
from tflearn.initializations import variance_scaling_initializer
from runners import sigopt_runner
load_dotenv(find_dotenv())


def model_fn(net, X_len, max_reach, block_size, out_classes, batch_size, dtype, **kwargs):
    """
        Args:
        net -> Input tensor shaped (batch_size, max_reach + block_size + max_reach, 3)
        Returns:
        logits -> Unscaled logits tensor in time_major form, (block_size, batch_size, out_classes)
    """

    with tf.name_scope("model"):
        print("model in", net.get_shape())
        for block in range(1, 4):
            with tf.variable_scope("block%d" % block):
                for layer in range(1, 1 + 1):
                    with tf.variable_scope('layer_%d' % layer):
                        res = net
                        for sublayer in [1, 2]:
                            res = batch_normalization(res, scope='bn_%d' % sublayer)
                            res = tf.nn.relu(res)
                            res = conv_1d(
                                res,
                                64,
                                3,
                                scope="conv_1d_%d" % sublayer,
                                weights_init=variance_scaling_initializer(dtype=dtype)
                            )
                        k = tf.get_variable("k", initializer=tf.constant_initializer(1.0), shape=[])
                        net = tf.nn.relu(k) * res + net
                net = max_pool_1d(net, 2)

        cut_size = tf.shape(net)[1] - tf.div(block_size, 8)
        with tf.control_dependencies([tf.assert_equal(tf.mod(cut_size, 2), 0)]):
            cut_size = tf.div(cut_size, 2)

        net = tf.slice(net, [0, cut_size, 0],
                       [-1, tf.div(block_size, 8), -1], name="Cutting")
        print("after slice", net.get_shape())

        net = tf.transpose(net, [1, 0, 2], name="Shift_to_time_major")

        state_size = 64
        outputs = net
        print("outputs", outputs.get_shape())

        with tf.variable_scope("Output"):
            outputs = tf.reshape(
                outputs, [tf.div(block_size, 8) * batch_size, state_size], name="flatten")
            W = tf.get_variable("W", shape=[state_size, out_classes])
            b = tf.get_variable("b", shape=[out_classes])
            outputs = tf.matmul(outputs, W) + b
            logits = tf.reshape(
                outputs, [tf.div(block_size, 8), batch_size, out_classes], name="logits")
    print("model out", logits.get_shape())
    return {
        'logits': logits,
        'init_state': tf.constant(0),
        'final_state': tf.constant(0),
    }


model_setup_params = dict(
    block_size_x=8 * 3 * 600 // 2,
    block_size_y=630,
    in_data="ALIGNED_RAW",
    num_blocks=1,
    batch_size=16,
    max_reach=8 * 20,  # 160
    queue_cap=300,
    overwrite=False,
    reuse=False,
    shrink_factor=8,
    dtype=tf.float32,
)

params = [
    dict(name='initial_lr', type='double',
         bounds=dict(min=1e-5, max=1e-3)),
    dict(name='decay_factor', type='double',
         bounds=dict(min=1e-3, max=0.5)),
]

default_params = {
    'initial_lr': 1e-4,
    'decay_factor': 0.1
}


if __name__ == "__main__":
    sigopt_runner(__file__[:-3].split('/')[-1])
