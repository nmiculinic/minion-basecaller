import tensorflow as tf
import model_utils
from tflearn.layers.conv import max_pool_1d, conv_1d
import os
from sigopt import Connection
from tflearn.layers.normalization import batch_normalization
from dotenv import load_dotenv, find_dotenv
from time import monotonic
from tflearn.initializations import variance_scaling_initializer
import tflearn
load_dotenv(find_dotenv())

# ~0.225 +- 0.001 tested on 500 samples from

def model_fn(net, X_len, max_reach, block_size, out_classes, batch_size, dtype, **kwargs):
    """
        Args:
        net -> Input tensor shaped (batch_size, max_reach + block_size + max_reach, 3)
        Returns:
        logits -> Unscaled logits tensor in time_major form, (block_size, batch_size, out_classes)
    """

    with tf.name_scope("model"):
        print("model in", net.get_shape())
        for block in range(1, 4):
            with tf.variable_scope("block%d" % block):
                for layer in range(1, 20 + 1):
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

        cut_size = tf.shape(net)[1] - tf.div(block_size, 8)
        with tf.control_dependencies([
            tf.cond(
                tflearn.get_training_mode(),
                lambda: tf.assert_equal(
                    tf.mod(cut_size, 2), 0, name="cut_size_assert"),
                lambda: tf.no_op()
            )
        ]
        ):
            cut_size = tf.div(cut_size, 2)

        net = tf.slice(net, [0, cut_size, 0],
                       [-1, tf.div(block_size, 8), -1], name="Cutting")
        print("after slice", net.get_shape())

        net = tf.transpose(net, [1, 0, 2], name="Shift_to_time_major")

        state_size = 64
        outputs = net
        print("outputs", outputs.get_shape())

        with tf.variable_scope("Output"):
            outputs = tf.reshape(
                outputs, [block_size // 8 * batch_size, state_size], name="flatten")
            W = tf.get_variable("W", shape=[state_size, out_classes])
            b = tf.get_variable("b", shape=[out_classes])
            outputs = tf.matmul(outputs, W) + b
            logits = tf.reshape(
                outputs, [block_size // 8, batch_size, out_classes], name="logits")
    print("model out", logits.get_shape())
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
        block_size_x=8 * 3 * 600 // 2,
        block_size_y=630,
        in_data="ALIGNED_RAW",
        num_blocks=1,
        batch_size=16,
        max_reach=8 * 20,  # 160
        model_fn=model_fn,
        queue_cap=300,
        overwrite=False,
        reuse=True,
        shrink_factor=8,
        run_id="deep_residual_15757_01",
        lr_fn=lambda global_step: tf.train.exponential_decay(
            hyper['initial_lr'], global_step, 100000, hyper['decay_factor']),
        dtype=tf.float32,
        hyper=hyper,
    )

    avg_loss, avg_edit = model.simple_managed_train_model(200001, summarize=False)
    return avg_edit, {
        'time[h]': (monotonic() - start_timestamp) / 3600.0,
        'logdir': model.log_dir,
        'average_loss_cv': avg_loss
    }


params = [
    dict(name='initial_lr', type='double',
         bounds=dict(min=1e-5, max=1e-3)),
    dict(name='decay_factor', type='double',
         bounds=dict(min=1e-3, max=0.5)),
]


def verify_hyper(hyper):
    return True


if __name__ == "__main__":
    conn = Connection(client_token=os.environ["SIGOPT_KEY"])

    if os.environ["EXPERIMENT_ID"] == "NEW":
        experiment = conn.experiments().create(
            name='MinION basecaller residual',
            parameters=params,
            observation_budget=20
        )
        print("Created experiment: https://sigopt.com/experiment/" + experiment.id)
        experiment_id = experiment.id
    else:
        experiment_id = os.environ["EXPERIMENT_ID"]
        print("Using experiment: https://sigopt.com/experiment/" + experiment_id)

    run_no = 0
    while True:
        run_no += 1

        suggestion = conn.experiments(experiment_id).suggestions().create()
        hyper = dict(suggestion.assignments)

        while not verify_hyper(hyper):
            print("Rejecting suggestion:")
            for k in sorted(hyper.keys()):
                print("%-20s: %7s" % (k, str(hyper[k])))
            conn.experiments(experiment_id).observations().create(
                suggestion=suggestion.id,
                metadata=dict(
                    hostname=model_utils.hostname,
                ),
                failed=True
            )
            suggestion = conn.experiments(experiment_id).suggestions().create()
            hyper = dict(suggestion.assignments)

#### ADJUST
        if os.environ['SIGOPT_KEY'].startswith("TJEAVRLBP"):
            print("DEVELOPMENT MODE!!!")
            hyper['initial_lr'] = 1e-4
            hyper['decay_factor'] = 0.1
####

        value, metadata = model_run(run_no, experiment_id, hyper)
        conn.experiments(experiment_id).observations().create(
            suggestion=suggestion.id,
            value=-value,
            metadata=dict(
                hostname=model_utils.hostname,
                run_no=run_no,
                **metadata
            )
        )
