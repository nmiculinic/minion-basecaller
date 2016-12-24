import tensorflow as tf
import numpy as np
from util import atrous_conv1d
import sys
import os
import model_utils
import time
import input_readers


def model_fn(net, X_len, max_reach, block_size, out_classes, batch_size, reuse=False, **kwargs):
    """
        Args:
        net -> Input tensor shaped (batch_size, max_reach + block_size + max_reach, 3)
        Returns:
        logits -> Unscaled logits tensor in time_major form, (block_size, batch_size, out_classes)
    """

    print("model in", net.get_shape())
    # net = tf.Print(net, [tf.shape(net), tf.shape(X_len)], message="netty")
    with tf.name_scope("model"):
        for i, no_channel in zip([1, 2], [16, 32]):
            with tf.variable_scope("atrous_conv1d_%d" % i):
                filter = tf.get_variable("W", shape=(3, net.get_shape()[-1], no_channel))
                bias = tf.get_variable("b", shape=(no_channel,))
                net = atrous_conv1d(net, filter, i, padding="VALID") + bias
                net = tf.nn.relu(net)
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

        with tf.variable_scope("Output"):
            outputs = tf.reshape(outputs, [-1, state_size])
            W = tf.get_variable("W", shape=[state_size, out_classes])
            b = tf.get_variable("b", shape=[out_classes])
            outputs = tf.matmul(outputs, W) + b
            logits = tf.reshape(outputs, [block_size, batch_size, out_classes])
    print("model out", logits.get_shape())
    return {
        'logits': logits,
        'init_state': init_state,
        'final_state': final_state
    }


if __name__ == "__main__":
    model = model_utils.Model(
        tf.get_default_graph(),
        block_size=20,
        num_blocks=2,
        batch_size=4,
        max_reach=3,
        model_fn=model_fn,
        queue_cap=100
        # overwrite=False,
        # run_id="init_model"
    )
    dummy_input = input_readers.get_feed_yield2(block_size=model.block_size, num_blocks=model.num_blocks, batch_size=4)
    model.init_session(num_workers=2, proc=False)

    for i in range(100001):
        model.train_minibatch()
        if i % 50 == 0 or i in [0, 10, 20, 30, 40]:
            model.summarize(i, write_example=True)
            d = next(dummy_input)
            X = d['X_enqueue_val']
            X_len = d['X_len_enqueue_val']
            print("Eval X:", X.shape, X_len.shape)
            print(model.eval_x(X, X_len))
        if i % 1000 == 0:
            model.save(i)

    print("closing session")
    model.close_session()
    print("finishing")
    time.sleep(120)
    sys.exit(0)
