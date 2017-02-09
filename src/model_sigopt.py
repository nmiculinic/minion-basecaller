import tensorflow as tf
import model_utils
from tflearn.layers.conv import max_pool_1d, conv_1d
import os
from sigopt import Connection
from tflearn.layers.normalization import batch_normalization
from dotenv import load_dotenv, find_dotenv
from time import monotonic
load_dotenv(find_dotenv())


def model_fn(net, X_len, max_reach, block_size, out_classes, batch_size, **kwargs):
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
                for layer in range(1, 1 + kwargs['b%d_layers' % block]):
                    with tf.variable_scope('layer_%d' % layer):
                        net = conv_1d(
                            net,
                            2**kwargs['l%d_lower_lg' % block],
                            1 + 2 * kwargs['l%d_receptive_field_l' % block],
                            scope='conv_1d_lower'
                        )
                        net = conv_1d(
                            net,
                            2**kwargs['l%d_upper_lg' % block],
                            1 + 2 * kwargs['l%d_receptive_field_l' % block] + 2 * kwargs['l%d_receptive_field_u' % block],
                            scope='conv_1d_upper'
                        )
                        net = batch_normalization(net, scope='batch_norm')
                        net = tf.nn.relu(net)
                net = tf.Print(net, [tf.shape(net)],
                               first_n=5, message="net, pre_pool")
                net = max_pool_1d(net, 2)

        print("block_size", block_size // 8)
        cut_size = int(net.get_shape()[1] - block_size // 8)
        assert cut_size % 2 == 0
        cut_size //= 2
        print("after conv", net.get_shape())
        net = tf.slice(net, [0, cut_size, 0],
                       [-1, block_size // 8, -1], name="Cutting")
        print("after slice", net.get_shape())

        net = tf.transpose(net, [1, 0, 2], name="Shift_to_time_major")

        print(net.get_shape)
        state_size = 2**kwargs['l3_upper_lg']
        outputs = net
        outputs = tf.Print(
            outputs, [tf.shape(outputs), block_size // 8 * batch_size, state_size], first_n=1, message="outputs_pre_w")
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
        block_size_x=8 * 256,
        block_size_y=256,
        in_data="ALIGNED_RAW",
        num_blocks=5,
        batch_size=16,
        max_reach=8 * 20,  # 160
        model_fn=model_fn,
        queue_cap=300,
        overwrite=True,
        reuse=False,
        shrink_factor=8,
        run_id="sigopt_%s_%02d" % (experiment_id, run_no),
        lr_fn=lambda global_step: tf.train.exponential_decay(
            hyper['initial_lr'], global_step, 100000, hyper['decay_factor']),
        hyper=hyper,
    )

    avg_loss, avg_edit = model.simple_managed_train_model(100000)
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

for layer in [1, 2, 3]:
    params.extend([
        {"name": "l%d_lower_lg" % layer, "type": 'int',
            "bounds": {'min': 3, 'max': 6}},
        {"name": "l%d_upper_lg" % layer, "type": 'int',
            "bounds": {'min': 5, 'max': 8}},
        {"name": "b%d_layers" % layer, "type": 'int',
            "bounds": {'min': 2, 'max': 4}},
        {"name": "l%d_receptive_field_l" % layer, "type": 'int',
            "bounds": {'min': 0, 'max': 2}},
        {"name": "l%d_receptive_field_u" % layer, "type": 'int',
            "bounds": {'min': 0, 'max': 2}},
    ])


def verify_hyper(hyper):
    for i in [1, 2, 3]:
        if hyper['l%d_upper_lg' % i] < hyper['l%d_lower_lg' % i]:
            print("Rejecting because upper/lower")
            return False

    if hyper['l1_upper_lg'] > hyper['l2_upper_lg']:
        print("Rejecting beause monotonic channel size")
        return False
    if hyper['l2_upper_lg'] > hyper['l3_upper_lg']:
        print("Rejecting beause monotonic channel size")
        return False

    return True


if __name__ == "__main__":
    conn = Connection(client_token=os.environ["SIGOPT_KEY"])

    if os.environ["EXPERIMENT_ID"] == "NEW":
        experiment = conn.experiments().create(
            name='MinION basecaller',
            parameters=params,
            observation_budget=30
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
