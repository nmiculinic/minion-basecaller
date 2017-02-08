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
        for j in range(3):
            with tf.variable_scope("block%d" % (j + 1)):
                for i in range(kwargs['b%d_layers' % (j + 1)]):
                    with tf.variable_scope('layer_%d' % (i + 1)):
                        net = conv_1d(
                            net,
                            2**kwargs['lower_lg'],
                            1 + 2 * kwargs['receptive_field_l'],
                            scope='conv_1d_lower'
                        )
                        net = conv_1d(
                            net,
                            2**kwargs['upper_lg'],
                            1 + 2 * kwargs['receptive_field_u'],
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
        net = tf.slice(net, [0, cut_size, 0], [-1, block_size // 8, -1], name="Cutting")
        print("after slice", net.get_shape())

        net = tf.transpose(net, [1, 0, 2], name="Shift_to_time_major")

        print(net.get_shape)
        state_size = 2**kwargs['upper_lg']
        outputs = net
        outputs = tf.Print(
            outputs, [tf.shape(outputs), block_size // 8 * batch_size, state_size], first_n=1, message="outputs_pre_w")
        print("outputs", outputs.get_shape())
        with tf.variable_scope("Output"):
            outputs = tf.reshape(outputs, [block_size // 8 * batch_size, state_size], name="flatten")
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


def model_run(run_no, hyper):

    start_timestamp = monotonic()

    model = model_utils.Model(
        tf.Graph(),
        block_size_x=8 * 128,
        block_size_y=128,
        in_data="RAW",
        num_blocks=6,
        batch_size=2**hyper['batch_size_log'],
        max_reach=8 * 20,  # 160
        model_fn=model_fn,
        queue_cap=500,
        overwrite=True,
        reuse=False,
        shrink_factor=8,
        run_id="sigopt_%02d" % run_no,
        lr_fn=lambda global_step: tf.train.exponential_decay(hyper['initial_lr'], global_step, 100000, hyper['decay_factor']),
        hyper=hyper,
    )

    model.logger.info("Running hyper parameters %s", str(hyper))

    try:
        model.init_session()
        for i in range(model.restore(must_exist=False) + 1, 50):
            print('\r%s Step %4d, loss %7.4f' % (model.run_id, i, model.train_minibatch()), end='')
            if i % 100 == 0:
                model.run_validation()
                model.summarize(write_example=False)
            if i % 3000 == 0:
                model.save()

        model.save()
        model.logger.info("Running final validation run")
        avg_loss, avg_edit = model.run_validation(num_batches=10)
        model.logger.info("Average loss %7.4f, Average edit distance %7.4f", avg_loss, avg_edit)
        return avg_edit, {
            'time[h]': (monotonic() - start_timestamp) / 3600.0,
            'logdir': model.log_dir,
            'average_loss_cv': avg_loss
        }
    except:
        model.logger.error("Error happened", exc_info=True)
    finally:
        model.train_writer.flush()
        model.test_writer.flush()
        model.close_session()


if __name__ == "__main__":

    conn = Connection(client_token=os.environ["SIGOPT_KEY"])

    if os.environ["EXPERIMENT_ID"] == "NEW":
        experiment = conn.experiments().create(
            name='MinION basecaller',
            parameters=[
                dict(name='batch_size_log', type='int', bounds=dict(min=4, max=7)),
                dict(name='initial_lr', type='double',
                     bounds=dict(min=1e-5, max=1e-3)),
                dict(name='decay_factor', type='double',
                     bounds=dict(min=1e-3, max=0.5)),

                dict(name="lower_lg", type='int', bounds={'min': 3, 'max': 6}),
                dict(name="upper_lg", type='int', bounds={'min': 5, 'max': 8}),

                dict(name="b1_layers", type='int', bounds={'min': 2, 'max': 4}),
                dict(name="b2_layers", type='int', bounds={'min': 2, 'max': 4}),
                dict(name="b3_layers", type='int', bounds={'min': 2, 'max': 4}),

                dict(name="receptive_field_l", type='int',
                     bounds={'min': 0, 'max': 2}),
                dict(name="receptive_field_u", type='int',
                     bounds={'min': 1, 'max': 3})
            ],
            observation_budget=int(os.environ["OBS"])
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
        print(hyper)
        value, metadata = model_run(run_no, hyper)
        conn.experiments(experiment_id).observations().create(
            suggestion=suggestion.id,
            value=-value,
            metadata=dict(
                hostname=model_utils.hostname,
                run_no=run_no,
                **metadata
            )
        )
