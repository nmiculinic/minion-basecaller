import tensorflow as tf
from tflearn.layers.conv import max_pool_1d, conv_1d
from tflearn.layers.normalization import batch_normalization
from dotenv import load_dotenv, find_dotenv
from tflearn.initializations import variance_scaling_initializer

from mincall.ops import central_cut
from mincall import input_readers
from mincall.util import sigopt_int, sigopt_double
from mincall.controller import control
from mincall.model_utils import Model
load_dotenv(find_dotenv())


def model_fn(net, X_len, max_reach, block_size, out_classes, batch_size, dtype, **kwargs):
    """
        Args:
        net -> Input tensor shaped (batch_size, max_reach + block_size + max_reach, 3)
        Returns:
        logits -> Unscaled logits tensor in time_major form, (block_size, batch_size, out_classes)
    """

    net = batch_normalization(net, decay=0.99, scope="Initial_bn")
    for block in range(1, 3):
        with tf.variable_scope("block%d" % block):
            for layer in range(kwargs['num_layers']):
                with tf.variable_scope("layer%d" % layer):
                    net = conv_1d(
                        net,
                        64,
                        9,
                        scope='conv1d'
                    )
                    net = batch_normalization(net, scope='bn')
                    net = tf.nn.relu(net)
            net = max_pool_1d(net, 2)
        net = tf.nn.relu(net)

    net = central_cut(net, block_size, 4)
    net = tf.transpose(net, [1, 0, 2], name="Shift_to_time_major")
    net = conv_1d(net, 9, 1, scope='logits')
    return {
        'logits': net,
        'init_state': tf.constant(0),
        'final_state': tf.constant(0),
    }


def model_setup_params(hyper):
    return dict(
        g=tf.Graph(),
        block_size_x=6 * 600,
        block_size_y=630,
        in_data=input_readers.AlbacoreAlignedRaw(0.2),
        num_blocks=5,
        batch_size=16,
        max_reach=400,
        queue_cap=300,
        overwrite=False,
        reuse=False,
        shrink_factor=4,
        dtype=tf.float32,
        model_fn=model_fn,
        n_samples_per_ref=5,
        lr_fn=lambda global_step: tf.train.exponential_decay(
            hyper['initial_lr'], global_step, 100000, hyper['decay_factor']),
        hyper=hyper,
    )


sigopt_params = [
    sigopt_double('initial_lr', 1e-5, 1e-3),
    sigopt_double('decay_factor', 1e-3, 0.5),
    sigopt_int('num_layers', 10, 20),
]

default_params = {
    'initial_lr': 0.000965352400196344,
    'decay_factor': 0.0017387361908150767,
    'num_layers': 15,
}


def create_train_model(hyper, **kwargs):
    model_setup = model_setup_params(hyper)
    model_setup.update(kwargs)
    return Model(**model_setup)


def create_test_model(hyper, **kwargs):
    return create_train_model(hyper, **kwargs)


default_name = "r15"


if __name__ == "__main__":
    control(globals())
