import tensorflow as tf
from tflearn.layers.conv import max_pool_1d, conv_1d
from tflearn.layers.normalization import batch_normalization
from dotenv import load_dotenv, find_dotenv
from train import sigopt_runner
from ops import central_cut
import numpy as np
load_dotenv(find_dotenv())


def model_fn(net, X_len, max_reach, block_size, out_classes, batch_size, reuse=False, **kwargs):
    """
        Args:
        net -> Input tensor shaped (batch_size, max_reach + block_size + max_reach, 3)
        Returns:
        logits -> Unscaled logits tensor in time_major form, (block_size, batch_size, out_classes)
    """

    print("model in", net.get_shape())
    for j in range(3):
        with tf.variable_scope("block%d" % (j + 1)):
            for i, no_channel in zip([1, 4, 16], np.array([64, 64, 128])*(2**j)):
                with tf.variable_scope("atrous_conv1d_%d" % i):
                    filter = tf.get_variable("W", shape=(3, net.get_shape()[-1], no_channel))
                    bias = tf.get_variable("b", shape=(no_channel, ))
                    net = tf.nn.convolution(net, filter, padding="SAME", dilation_rate=[i]) + bias
                    net = tf.nn.relu(net)
            net = max_pool_1d(net, 2)
    net = central_cut(net, block_size, 8)
    net = tf.transpose(net, [1, 0, 2], name="Shift_to_time_major")
    net = conv_1d(net, 9, 1, scope='logits')
    print("model out", net.get_shape())
    return {
        'logits': net,
        'init_state': tf.constant(0),
        'final_state': tf.constant(0),
    }


def model_setup_params(hyper):
    print("Requesting %s hyperparams" % __file__)
    return dict(
        g=tf.Graph(),
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
        model_fn=model_fn,
        lr_fn=lambda global_step: tf.train.exponential_decay(
            hyper['initial_lr'], global_step, 100000, hyper['decay_factor']),
        hyper=hyper,
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
