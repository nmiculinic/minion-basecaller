import tensorflow as tf
import numpy as np
from util import atrous_conv1d
import os
import model_utils
from tflearn.layers.conv import max_pool_1d
import logging

def model_fn(net, X_len, max_reach, block_size, out_classes, batch_size, reuse=False, **kwargs):
    """
        Args:
        net -> Input tensor shaped (batch_size, max_reach + block_size + max_reach, 3)
        Returns:
        logits -> Unscaled logits tensor in time_major form, (block_size, batch_size, out_classes)
    """

    print("model in", net.get_shape())
    with tf.name_scope("model"):
        for j in range(3):
            with tf.variable_scope("block%d" % (j + 1)):
                for i, no_channel in zip([1, 2, 4], np.array([64, 64, 64])):
                    with tf.variable_scope("atrous_conv1d_%d" % i):
                        filter = tf.get_variable("W", shape=(5, net.get_shape()[-1], no_channel))
                        bias = tf.get_variable("b", shape=(no_channel,))
                        net = atrous_conv1d(net, filter, i, padding="VALID") + bias
                        net = tf.nn.relu(net)
                net = tf.Print(net, [tf.shape(net)], first_n=5, message="net, pre_pool")
                net = max_pool_1d(net, 2)
        print("after conv", net.get_shape())
        net = tf.transpose(net, [1, 0, 2], name="Shift_to_time_major")

        outputs = net
        state_size = 64  # outputs.get_shape()[-1]  # Number of output filters
        init_state = tf.constant(0.1, dtype=tf.float32)
        final_state = tf.constant(0.1, dtype=tf.float32)

        outputs = tf.Print(outputs, [tf.shape(outputs)], first_n=1, message="outputs_pre_w")
        print("outputs", outputs.get_shape())
        with tf.variable_scope("Output"):
            outputs = tf.reshape(outputs, [-1, state_size])
            W = tf.get_variable("W", shape=[state_size, out_classes])
            b = tf.get_variable("b", shape=[out_classes])
            outputs = tf.matmul(outputs, W) + b
            logits = tf.reshape(outputs, [block_size // 8, batch_size, out_classes])
    print("model out", logits.get_shape())
    return {
        'logits': logits,
        'init_state': init_state,
        'final_state': final_state
    }


if __name__ == "__main__":
    model = model_utils.Model(
        tf.get_default_graph(),
        block_size_x=8 * 50,
        block_size_y=50,
        in_data="RAW",
        num_blocks=10,
        batch_size=32,
        max_reach=98,
        model_fn=model_fn,
        queue_cap=300,
        overwrite=False,
        reuse=True,
        shrink_factor=8,
        run_id=__file__[:-3],
    )

    lr_schedule = {
        0: 1e-3,
        50000: 5e-4,
        100000: 3e-4,
        125000: 1e-4,
        150000: 5e-5,
    }

    model.init_session()
    iter_step = 0
    iter_step = model.restore()
    lr = 1e-3
    for k, v in lr_schedule.items():
        if iter_step > k:
            lr = min(v, lr)
    model.set_lr(lr)

    for i in range(iter_step + 1, 200001):
        if i in lr_schedule:
            model.set_lr(lr_schedule[i])
        model.train_minibatch(i)
        if i % 200 == 0:
            model.run_validation(i)
            model.summarize(i, write_example=False)
        if i % 2000 == 0:
            model.save(i)
    model.close_session()