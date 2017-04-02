import tensorflow as tf
import numpy as np
from ops import atrous_conv1d
import model_utils
from tflearn.layers.conv import max_pool_1d
from util import sigopt_double
import input_readers


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
                        net = atrous_conv1d(net, filter, i, padding="SAME") + bias
                        net = tf.nn.relu(net)
                net = max_pool_1d(net, 2)
        print("after conv", net.get_shape())
        net = tf.transpose(net, [1, 0, 2], name="Shift_to_time_major")

        outputs = net
        state_size = 64  # outputs.get_shape()[-1]  # Number of output filters
        init_state = tf.constant(0.1, dtype=tf.float32)
        final_state = tf.constant(0.1, dtype=tf.float32)

        print("outputs", outputs.get_shape())
        with tf.variable_scope("Output"):
            W = tf.get_variable("W", shape=[state_size, out_classes])
            b = tf.get_variable("b", shape=[out_classes])
            logits = tf.nn.conv1d(outputs, tf.reshape(W, (1, state_size, out_classes)), 1, padding='SAME')
            logits += b
    print("model out", logits.get_shape())
    return {
        'logits': logits,
        'init_state': init_state,
        'final_state': final_state
    }


def model_setup_params(hyper):
    print("Requesting %s hyperparams" % __file__)
    return dict(
        g=tf.Graph(),
        block_size_x=8 * 3 * 600 // 2,
        block_size_y=630,
        in_data=input_readers.AlignedRaw(),
        num_blocks=3,
        batch_size=16,
        max_reach=98,  # 240
        queue_cap=300,
        overwrite=False,
        reuse=True,
        shrink_factor=8,
        dtype=tf.float32,
        model_fn=model_fn,
        lr_fn=lambda global_step: tf.train.exponential_decay(
            hyper['initial_lr'], global_step, 100000, hyper['decay_factor']),
        hyper=hyper,
    )


params = [
    sigopt_double('initial_lr', 1e-5, 1e-3),
    sigopt_double('decay_factor', 1e-3, 0.5),
]

default_params = {
    'initial_lr': 0.000965352400196344,
    'decay_factor': 0.0017387361908150767,
}


if __name__ == "__main__":
    params = default_params
    model = model_utils.Model(
        **model_setup_params(params)
    )

    model.init_session()
    model.simple_managed_train_model(25000)
    model.close_session()
