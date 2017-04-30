import tensorflow as tf
import numpy as np
from util import atrous_conv1d
import os
import model_utils
from tflearn.layers.conv import max_pool_1d
import logging
import ops

def model_fn(net, X_len, max_reach, block_size, out_classes, batch_size, k, reuse=False):
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
                for i, no_channel in zip([1, 4, 16], np.array([64, 64, 128]) * (2**j)):
                    with tf.variable_scope("atrous_conv1d_%d" % i):
                        filter = tf.get_variable("W", shape=(
                            3, net.get_shape()[-1], no_channel))
                        bias = tf.get_variable("b", shape=(no_channel,))
                        net = atrous_conv1d(
                            net, filter, i, padding="VALID") + bias
                        net = tf.nn.relu(net)
                net = tf.Print(net, [tf.shape(net)],
                               first_n=5, message="net, pre_pool")
                net = max_pool_1d(net, 2)
        print("after conv", net.get_shape())
        net = tf.transpose(net, [1, 0, 2], name="Shift_to_time_major")

        outputs = net

        state_size = 128

        with tf.name_scope("RNN"):
            cell = tf.nn.rnn_cell.GRUCell(state_size)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell for _ in range(3)])
            init_state = cell.zero_state(batch_size, dtype=tf.float32)
            outputs, final_state = tf.nn.dynamic_rnn(cell, net, initial_state=init_state, sequence_length=X_len, time_major=True, parallel_iterations=128)

        outputs = tf.Print(
            outputs, [tf.shape(outputs)], first_n=1, message="outputs_pre_w")
        print("outputs", outputs.get_shape())
        with tf.variable_scope("Output"):
            outputs = tf.reshape(outputs, [-1, state_size])
            W = tf.get_variable("W", shape=[state_size, out_classes])
            b = tf.get_variable("b", shape=[out_classes])
            outputs = tf.matmul(outputs, W) + b
            logits = tf.reshape(
                outputs, [block_size // 8, batch_size, out_classes])
    print("model out", logits.get_shape())
    return {
        'logits': logits,
        'init_state': init_state,
        'final_state': final_state,
        'reg': k * ops.running_mean(logits, [5, 6, 7], [4, 8, 16], out_classes)
    }


if __name__ == "__main__":
    model = model_utils.Model(
        tf.get_default_graph(),
        block_size_x=8 * 50,
        block_size_y=50,
        in_data="RAW",
        num_blocks=6,
        batch_size=16,
        max_reach=147,
        model_fn=model_fn,
        queue_cap=300,
        overwrite=True,
        reuse=False,
        shrink_factor=8,
        run_id=__file__[:-3],
        hyper={'k': 2000.0},
    )

    try:
        model.init_session()
        for i in range(model.restore(must_exist=False) + 1, 100001):
            model.train_minibatch()
            if i % 200 == 0:
                model.run_validation()
                model.summarize(write_example=False)
            if i % 2000 == 0:
                model.save()
    finally:
        model.close_session()
