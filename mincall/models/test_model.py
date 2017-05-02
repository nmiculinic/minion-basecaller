import tensorflow as tf
from tflearn.layers.conv import max_pool_1d, conv_1d
from tflearn.layers.normalization import batch_normalization
from dotenv import load_dotenv, find_dotenv
from tflearn.initializations import variance_scaling_initializer
from mincall.ops import central_cut
from mincall.util import sigopt_double
from mincall.model_utils import Model
from mincall.controller import control
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
            for layer in range(1, 1 + 1):
                with tf.variable_scope('layer_%d' % layer):
                    res = net
                    for sublayer in [1, 2]:
                        res = batch_normalization(
                            res, scope='bn_%d' % sublayer)
                        res = tf.nn.relu(res)
                        res = conv_1d(
                            res,
                            64,
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


def create_train_model(hyper, **kwargs):
    print("Requesting %s hyperparams" % __file__)
    model_setup = dict(
        g=tf.Graph(),
        block_size_x=8 * 3 * 50 // 2,
        block_size_y=80,
        num_blocks=1,
        batch_size=32,
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
    model_setup.update(kwargs)
    return Model(**model_setup)


def create_test_model(hyper, **kwargs):
    return create_train_model(hyper, **kwargs)


sigopt_params = [
    sigopt_double('initial_lr', 1e-5, 1e-3),
    sigopt_double('decay_factor', 1e-3, 0.5),
]

default_params = {
    'initial_lr': 1e-3,
    'decay_factor': 0.1,
}

default_name = "test_model"

if __name__ == "__main__":
    control(globals())
