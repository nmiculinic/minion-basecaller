import tensorflow as tf
import numpy as np
from util import atrous_conv1d
import sys
import os
import model_utils
import time
import input_readers
from tflearn.layers.conv import max_pool_1d
import ops

def model_fn(net, X_len, max_reach, block_size, out_classes, batch_size, reuse=False, **kwargs):
    """
        Args:
        net -> Input tensor shaped (batch_size, max_reach + block_size + max_reach, 3)
        Returns:
        logits -> Unscaled logits tensor in time_major form, (block_size, batch_size, out_classes)
    """

    print("model in", net.get_shape())
    # net = tf.Print(net, [tf.shape(net), tf.shape(X_len)], message="netty")
    for j in range(3):
        with tf.variable_scope("block%d" % (j + 1)):
            for i, no_channel in zip([1, 2, 4], [16, 16, 16]):
                with tf.variable_scope("atrous_conv1d_%d" % i):
                    filter = tf.get_variable("W", shape=(3, net.get_shape()[-1], no_channel))
                    bias = tf.get_variable("b", shape=(no_channel,))
                    net = atrous_conv1d(net, filter, i, padding="VALID") + bias
                    net = tf.nn.relu(net)
                    tf.get_default_graph().add_to_collection("activations", net)
            net = tf.Print(net, [tf.shape(net)], first_n=10, message="net, pre_pool")
            net = max_pool_1d(net, 2)
    print("after conv", net.get_shape())
    net = tf.transpose(net, [1, 0, 2], name="Shift_to_time_major")

    state_size = 32  # outputs.get_shape()[-1]  # Number of output filters
    with tf.name_scope("RNN"):
        cell = tf.nn.rnn_cell.GRUCell(state_size)
        init_state = cell.zero_state(batch_size, dtype=tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell, net, initial_state=init_state, sequence_length=X_len, time_major=True)
    # outputs = net
    # init_state = tf.constant(0.1, dtype=tf.float32)
    # final_state = tf.constant(0.1, dtype=tf.float32)

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
        'final_state': final_state,
        'reg': ops.running_mean(logits, [3, 4, 5], [1, 4, 16], out_classes)
    }


model = model_utils.Model(
    tf.Graph(),
    block_size_x=8 * 20,
    block_size_y=20,
    in_data="RAW",
    num_blocks=2,
    batch_size=64,
    max_reach=49,
    model_fn=model_fn,
    queue_cap=100,
    shrink_factor=8,
    overwrite=False,
    reuse=True,
    run_id="init_model",
    lr_fn=lambda gs: tf.train.exponential_decay(1e-3, gs, 10000, 0.01)
)

if __name__ == "__main__":

    print(__file__[:-3])
    try:
        model.init_session()
        iter_step = model.restore(must_exist=False)
        for i in range(iter_step + 1, 10001):
            model.train_minibatch(log_every=5)
            assert i == model.get_global_step()
            if i % 20 == 0 or i in [1, 10, 20, 30, 40]:
                model.summarize(write_example=True)
                _, avg_edit_distance = model.run_validation()
                print(avg_edit_distance)
            if i % 100 == 0:
                model.save()
    finally:
        model.close_session()
