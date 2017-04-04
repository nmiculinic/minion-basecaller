import tensorflow as tf
from tflearn.layers.conv import max_pool_1d, conv_1d
import input_readers
from tflearn.layers.normalization import batch_normalization
from dotenv import load_dotenv, find_dotenv
from tflearn.initializations import variance_scaling_initializer
from train import sigopt_runner
from ops import central_cut
from util import sigopt_int, sigopt_double
load_dotenv(find_dotenv())


def model_fn(net, X_len, max_reach, block_size, out_classes, batch_size, dtype, **kwargs):
    """
        Args:
        net -> Input tensor shaped (batch_size, max_reach + block_size + max_reach, 3)
        Returns:
        logits -> Unscaled logits tensor in time_major form, (block_size, batch_size, out_classes)
    """

    print("model in", net.get_shape())
    for block in range(1, 4):
        with tf.variable_scope("block%d" % block):
            if block > 1:
                net = tf.expand_dims(net, 3)
                net = tf.layers.max_pooling2d(net, [1, 2], [1, 2])
                net = tf.squeeze(net, axis=3)

            for layer in range(kwargs['num_layers']):
                with tf.variable_scope('layer_%d' % layer):
                    res = net
                    for sublayer in range(kwargs['num_sub_layers']):
                        res = batch_normalization(
                            res, scope='bn_%d' % sublayer)
                        res = tf.nn.relu(res)
                        res = conv_1d(
                            res,
                            32 * 2**(4 - block),
                            3,
                            scope="conv_1d_%d" % sublayer,
                            weights_init=variance_scaling_initializer(
                                dtype=dtype)
                        )
                    k = tf.get_variable(
                        "k", initializer=tf.constant_initializer(1.0), shape=[])
                    net = tf.nn.relu(k) * res + net
            net = max_pool_1d(net, 2)
        net = tf.nn.relu(net)

    net = central_cut(net, block_size, 8)
    print("after slice", net.get_shape())
    net = tf.transpose(net, [1, 0, 2], name="Shift_to_time_major")
    print("after transpose", net.get_shape())
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
        in_data=input_readers.AlignedRaw(),
        num_blocks=3,
        batch_size=16,
        max_reach=8 * 20,  # 240
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
    sigopt_double('initial_lr', 1e-5, 1e-3),
    sigopt_double('decay_factor', 1e-3, 0.5),
    sigopt_int('num_layers', 10, 20),
    sigopt_int('num_sub_layers', 1, 2),
]

default_params = {
    'initial_lr': 0.000965352400196344,
    'decay_factor': 0.0007387361908150767,
    'num_layers': 10,
    'num_sub_layers': 1
}


if __name__ == "__main__":
    sigopt_runner(__file__[:-3].split('/')[-1])
