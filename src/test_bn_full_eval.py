import tensorflow as tf
import model_utils
from tflearn.layers.conv import max_pool_1d, conv_1d
from tflearn.layers.normalization import batch_normalization
from dotenv import load_dotenv, find_dotenv
from time import monotonic
from tflearn.initializations import variance_scaling_initializer
import numpy as np
load_dotenv(find_dotenv())


def model_fn(net, X_len, max_reach, block_size, out_classes, batch_size, dtype, **kwargs):
    """
        Args:
        net -> Input tensor shaped (batch_size, max_reach + block_size + max_reach, 3)
        Returns:
        logits -> Unscaled logits tensor in time_major form, (block_size, batch_size, out_classes)
    """

    net = tf.Print(net, [tf.shape(net), X_len], message="input shape")
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

        # print("block_size", block_size // 8)
        # cut_size = int(net.get_shape()[1] - block_size // 8)
        # assert cut_size % 2 == 0
        # cut_size //= 2
        # print("after conv", net.get_shape())
        # net = tf.slice(net, [0, cut_size, 0],
        #                [-1, block_size // 8, -1], name="Cutting")
        # print("after slice", net.get_shape())

        net = tf.transpose(net, [1, 0, 2], name="Shift_to_time_major")

        with tf.variable_scope("Output"):
            logits = conv_1d(net, 9, 1)
        print("logits", logits.get_shape())

    return {
        'logits': logits,
        'init_state': tf.constant(0),
        'final_state': tf.constant(0),
    }


def model_run(run_no, experiment_id, hyper):
    print("Running hyper parameters")
    for k in sorted(hyper.keys()):
        print("%-20s: %7s" % (k, str(hyper[k])))

    start_timestamp = monotonic()
    model = model_utils.Model(
        tf.Graph(),
        block_size_x=8 * 3 * 50 // 2,
        block_size_y=50,
        in_data="ALIGNED_RAW",
        num_blocks=1,
        batch_size=16,
        max_reach=8 * 20,  # 160
        model_fn=model_fn,
        queue_cap=300,
        overwrite=False,
        reuse=True,
        shrink_factor=8,
        run_id="testing",
        lr_fn=lambda global_step: tf.train.exponential_decay(
            hyper['initial_lr'], global_step, 100000, hyper['decay_factor']),
        dtype=tf.float32,
        hyper=hyper,
    )

    # avg_loss, avg_edit = model.simple_managed_train_model(20, final_val_samples=100, summarize=False)
    model.init_session()
    model.restore()

    edit_distances = model.run_validation_full(1.0)

    model.close_session()

    # with model.g.as_default():
    #     for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    #         print(var.name, var.dtype, var.get_shape())

    return np.mean(edit_distances), {
        'time[h]': (monotonic() - start_timestamp) / 3600.0,
        'logdir': model.log_dir,
    }


default_params = {
    'initial_lr': 1e-4,
    'decay_factor': 0.1
}


if __name__ == "__main__":
    for k, v in model_run(0, 0, default_params)[1].items():
        print(k, v)
