import numpy as np
import util
import socket
import tensorflow as tf
from util import decode_example, decode_sparse
from ops import dense2d_to_sparse
import input_readers
import multiprocessing
from threading import Thread
from time import perf_counter
from tensorflow.python.client import timeline
import os
import string
import random
import shutil
import warpctc_tensorflow
from tflearn.summaries import add_gradients_summary, add_activations_summary
import logging
from tflearn.config import is_training, get_training_mode
from slacker_log_handler import SlackerLogHandler
from edlib import Edlib
import inspect
import json
import importlib


# UGLY UGLY HACK!
for name, logger in logging.root.manager.loggerDict.items():
    logger.disabled=True


hostname = os.environ.get("MINION_HOSTNAME", socket.gethostname())
repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
log_fmt = '\r[%(levelname)s] %(name)s: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_fmt)


def default_lr_fn(global_step):
    return tf.train.exponential_decay(1e-3, global_step, 100000, 0.01)


def load_model_parms(module_name, model_dir):
    model_module = importlib.import_module(module_name)
    model_dir = os.path.abspath(model_dir)
    with open(os.path.join(model_dir, 'model_hyperparams.json'), 'r') as f:
        hyper = json.load(f)

    params = model_module.model_setup_params(hyper)
    print(params, type(params))
    params['reuse'] = True
    params['overwrite'] = False
    params['log_dir'] = model_dir
    params['run_id'] = model_dir.split('/')[-1]
    return params


class Model():
    def __init__(self, g, num_blocks, batch_size, max_reach, model_fn, block_size_x, block_size_y, lr_fn=default_lr_fn, log_dir=None, run_id=None, overwrite=False, reuse=False, queue_cap=None, shrink_factor=1, test_queue_cap=None, in_data=input_readers.AlignedRaw(), dtype=tf.float32, hyper={}, clip_grad=2.0):
        """
            Args:
                max_reach: int, size of contextual window for convolutions etc.
                model_fn: function accepting (batch_size, 2*max_reach + block_size, 3) -> (block_size, batch_size, out_classes). Notice shift to time major as well as reduction in time dimension.
                hyper: dictionary of hyperparameters passed to the model function as **hyper
                clip_grad: Maximum gradient_clipping value, None to disable
        """

        if isinstance(in_data, input_readers.InputReader.__class__):
            raise ValueError("in_data must instance of InputReader")

        self.in_data = in_data
        self.lr_fn = lr_fn or default_lr_fn

        if overwrite and reuse:
            raise ValueError("Cannot overwrite and reuse logdit and checkpoints")

        self._handle_logdir(log_dir, run_id, overwrite, reuse)

        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        valargs = {arg: values[arg] for arg in args}

        for k in sorted(valargs.keys()):
            self.logger.info("%-20s: %7s" % (k, str(valargs[k])))

        fname = os.path.join(self.log_dir, 'model_hyperparams.json')
        with open(fname, 'w') as f:
            json.dump(hyper, f, sort_keys=True, indent=4)
            self.logger.info("Dumping model hyperparams to %s", fname)

        del args
        del values
        del valargs

        self.block_size_y = block_size_y
        self.block_size_x = block_size_x
        self.shrink_factor = shrink_factor
        if self.block_size_x % shrink_factor != 0:
            raise ValueError("shrink factor need to divide block_size_x")

        self.num_blocks = num_blocks
        self.batch_size = batch_size
        self.max_reach = max_reach
        self.train_queue_cap = queue_cap or 5 * self.batch_size
        self.test_queue_cap = test_queue_cap or self.train_queue_cap
        self.dtype = dtype
        self.g = g
        self.clip_grad = clip_grad
        self._setup_graph(model_fn, hyper)

    def _setup_graph(self, model_fn, hyper):
        self._setup_graph_pre()
        with self.g.as_default():
            data = self._setup_logits(model_fn, hyper)
            self.logits = data['logits']
            self.init_state = data['init_state']
            self.final_state = data['final_state']
            self.reg = data.get('reg', tf.constant(0))
            self.loss = self._setup_loss(self.logits)

            self._setup_prediction()
            self._setup_train()
            self._setup_saver()

    def _setup_saver(self):
        self.saver = tf.train.Saver(
            keep_checkpoint_every_n_hours=1,
        )

    def _setup_graph_pre(self):
        """
            Sets up everything before requirnig model_fn parametirzing model.
        """
        with self.g.as_default():
            self.training_mode = get_training_mode()
            self.batch_size_var = tf.placeholder_with_default(tf.convert_to_tensor(self.batch_size, dtype=tf.int32), [])
            self._create_train_input_objects()
            self.block_size_x_tensor = tf.placeholder_with_default(self.block_size_x, [])

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.inc_gs = tf.assign_add(self.global_step, 1)
            self.lr = self.lr_fn(self.global_step)

    def _setup_logits(self, model_fn, hyper):
        """
            Setup logits function. Returns dict containing logits, init_state, final_state and optionally regularization
        """
        with tf.variable_scope("model"):
            data = model_fn(
                self.X_batch,
                self.X_batch_len,
                max_reach=self.max_reach,
                block_size=self.block_size_x_tensor,
                out_classes=9,
                batch_size=self.batch_size_var,
                dtype=self.dtype,
                **hyper
            )

        if data['logits'].get_shape()[2] != 9:
            raise ValueError("Loggits must be tensor with dim 2 = 9\n%s" % str(data['logits'].get_shape()))
        with tf.control_dependencies([
            tf.cond(
                self.training_mode,
                lambda: tf.assert_equal(tf.shape(data['logits']), [self.block_size_x_tensor // self.shrink_factor, self.batch_size_var, 9]),
                lambda: tf.no_op()
            )
        ]):
            data['logits'] = tf.identity(data['logits'])
        return data

    def _setup_loss(self, logits):
        """
            Function returning loss.
        """
        print("Model _setup_loss")
        with tf.name_scope("loss"):
            loss = warpctc_tensorflow.ctc(
                self.logits,
                self.Y_batch.values,
                self.Y_batch_len,
                tf.div(self.X_batch_len, self.shrink_factor),
                blank_label=8
            )
            return tf.reduce_mean(loss)

    def _setup_prediction(self):
        """
            Sets up pred, dense_pred and edit_distance in self
        """
        def dedup_output(sparse_tensor):
            sol_values = sparse_tensor.values - 4 * tf.to_int32(sparse_tensor.values >= 4)
            sol = tf.SparseTensor(
                sparse_tensor.indices,
                tf.Print(sol_values, [sol_values, sparse_tensor.values], first_n=5, message="__dedup", summarize=20),
                sparse_tensor.dense_shape
            )
            return sol

        with tf.name_scope("prediction"):
            predicted = self._predict_from_logits()
            self.pred = dedup_output(tf.cast(predicted[0], tf.int32))
            self.dense_pred = tf.sparse_tensor_to_dense(self.pred, default_value=-1)
            self.edit_distance = tf.edit_distance(
                self.pred,
                dedup_output(self.Y_batch)
            )

    def _predict_from_logits(self):
        predicted, _ = tf.nn.ctc_beam_search_decoder(self.logits, tf.div(self.X_batch_len, self.shrink_factor), merge_repeated=True, top_paths=1)
        return predicted

    def _setup_train(self):
        """
            Sets up rest of training. Assumes self.lr and self.loss are defined
        """
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.grads = optimizer.compute_gradients(self.loss, self.g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model'))
        if self.clip_grad is not None:
            with tf.name_scope("gradient_clipping"):
                self.grads = [(tf.clip_by_value(grad, -2., 2.), var) for grad, var in self.grads]

        self.train_op = optimizer.apply_gradients(self.grads)

        self.grad_summ = tf.summary.merge(add_gradients_summary(self.grads))
        self.activation_summ = tf.summary.merge(
            add_activations_summary(self.g.get_collection("activations")))

        self.train_summ = tf.summary.merge([
            self.train_queue_size,
            tf.summary.scalar("train/learning_rate", self.lr)
        ])

    def _handle_logdir(self, log_dir, run_id, overwrite, reuse):
        if run_id is None:
            run_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
        self.run_id = run_id

        if log_dir is None:
            self.log_dir = os.path.join(repo_root, 'log', hostname, self.run_id)
        else:
            self.log_dir = log_dir

        if os.path.exists(self.log_dir):
            if overwrite:
                print("Clearing %s" % self.log_dir)
                shutil.rmtree(self.log_dir)
            elif not reuse:
                raise ValueError("path " + self.log_dir + " exists")

        os.makedirs(self.log_dir, mode=0o744, exist_ok=reuse or overwrite)
        print("Logdir = ", self.log_dir)
        self.logger = logging.getLogger(run_id)
        self.logger.propagate = False
        self.logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(log_fmt))
        self.logger.addHandler(ch)

        hdlr = logging.FileHandler(os.path.join(self.log_dir, "model.log"))
        file_log_fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        hdlr.setFormatter(file_log_fmt)
        hdlr.setLevel(logging.DEBUG)
        self.logger.addHandler(hdlr)
        self.logger.debug("pl")
        if "SLACK_TOKEN" in os.environ:
            username = "%s@%s" % (run_id, hostname)
            self.logger.info("Adding slack logger")
            slack_handler = SlackerLogHandler(os.environ['SLACK_TOKEN'], hostname, stack_trace=True, username=username)

            slack_handler.setFormatter(file_log_fmt)
            slack_handler.setLevel(logging.INFO)
            print("Ignoring Slack INFO handler")
            # self.logger.addHandler(slack_handler)

            slack_handler = SlackerLogHandler(os.environ['SLACK_TOKEN'], 'error', username=username)

            slack_handler.setFormatter(file_log_fmt)
            slack_handler.setLevel(logging.ERROR)
            self.logger.addHandler(slack_handler)
        self.logger.info("Logdir %s", self.log_dir)

    def _create_train_input_objects(self):
        with tf.variable_scope("input"):
            input_vars = [
                tf.get_variable("X",
                                shape=[self.batch_size, self.block_size_x * self.num_blocks, self.in_data.in_dim],
                                initializer=tf.zeros_initializer(),
                                dtype=self.dtype,
                                trainable=False
                                ),
                tf.get_variable("X_len",
                                shape=[self.batch_size],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.int32,
                                trainable=False
                                ),
                tf.get_variable("Y",
                                initializer=tf.zeros_initializer(),
                                dtype=tf.uint8,
                                shape=[self.batch_size, self.block_size_y * self.num_blocks],
                                trainable=False
                                ),
                tf.get_variable("Y_len",
                                shape=[self.batch_size, self.num_blocks],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.int32,
                                trainable=False
                                ),
            ]
            names = [x.name[6:-2] for x in input_vars]  # TODO, hacky
            shapes = [x.get_shape()[1:] for x in input_vars]
            types = [x.dtype.base_dtype for x in input_vars]

            with tf.name_scope("queue_handling"):
                self.train_queue = tf.FIFOQueue(self.train_queue_cap, types, shapes=shapes)
                self.test_queue = tf.FIFOQueue(self.test_queue_cap, types, shapes=shapes)

                self.train_queue_size = tf.summary.scalar("train_queue_filled", self.train_queue.size())
                self.test_queue_size = tf.summary.scalar("test_queue_filled", self.test_queue.size())

                self.close_queues = tf.group(
                    self.train_queue.close(True),
                    self.test_queue.close(True)
                )

                for name, x in zip(
                    names,
                    input_vars,
                ):
                    self.__dict__[name] = x
                    self.__dict__[name + "_enqueue_val"] = tf.placeholder(x.dtype.base_dtype, shape=[None, *x.get_shape()[1:]], name=name + "_enqueue_val")

                self.enqueue_train = self.train_queue.enqueue_many(
                    [self.__dict__[name + "_enqueue_val"] for name in names])
                self.enqueue_test = self.test_queue.enqueue_many(
                    [self.__dict__[name + "_enqueue_val"] for name in names])

                self.load_train = tf.group(*[
                    tf.assign(x, qx) for x, qx in zip(input_vars, self.train_queue.dequeue_many(self.batch_size))
                ])
                self.load_test = tf.group(*[
                    tf.assign(x, qx) for x, qx in zip(input_vars, self.test_queue.dequeue_many(self.batch_size))
                ])

            self.block_idx = tf.placeholder(dtype=tf.int32, shape=[], name="block_idx")
            begin_x = self.block_idx * self.block_size_x
            begin_y = self.block_idx * self.block_size_y

            with tf.name_scope("X_batch"):
                X = self.X
                with tf.control_dependencies([
                    tf.assert_less_equal(begin_x + self.block_size_x, tf.shape(X)[1], message="Cannot request that many elements from X"),
                    tf.assert_non_negative(self.block_idx, message="Beginning slice must be >=0"),
                ]):
                    max_len_x = tf.shape(X)[1]
                    left = tf.maximum(0, begin_x - self.max_reach)
                    right = tf.minimum(max_len_x, begin_x + self.block_size_x + self.max_reach)
                    self.X_batch_len = tf.clip_by_value(
                        tf.slice(self.X_len, [0], [self.batch_size_var]) - begin_x, 0, self.block_size_x)

                net = tf.slice(X, [0, left, 0], [self.batch_size_var, right - left, -1])
                padding = [
                    [0, 0],
                    [tf.maximum(0, self.max_reach - begin_x), tf.maximum(0, begin_x + self.block_size_x + self.max_reach - max_len_x)],
                    [0, 0]
                ]
                padding = tf.convert_to_tensor(padding)
                net = tf.pad(net, padding)
                net.set_shape([None, 2 * self.max_reach + self.block_size_x, self.in_data.in_dim])
                print(net.get_shape())
                self.X_batch = tf.placeholder_with_default(net, (None, None, self.in_data.in_dim))

            with tf.name_scope("Y_batch"):
                self.Y_batch_len = tf.squeeze(tf.slice(self.Y_len, [0, self.block_idx], [self.batch_size_var, 1]), [1])
                # TODO: migrate to batch_size_var
                self.Y_batch = dense2d_to_sparse(tf.slice(self.Y, [0, begin_y], [self.batch_size, self.block_size_y]), self.Y_batch_len, dtype=tf.int32)
        return self.X_batch

    def _rnn_roll(self, add_fetch=[], add_feed={}, timeline_suffix=""):
        if 'sess' not in self.__dict__:
            raise ValueError("session not initialized")

        for retry in range(1, 4):
            try:
                fetch = [self.final_state]
                fetch.extend(add_fetch)
                run_metadata = tf.RunMetadata()
                sol = [[] for _ in range(len(add_fetch))]
                state = self.sess.run(self.init_state, feed_dict=add_feed)
                for blk in range(self.num_blocks):
                    feed = {
                        self.block_idx: blk,
                        self.init_state: state,
                    }
                    feed.update(add_feed)
                    state, *vals = self.sess.run(
                        fetch,
                        feed_dict=feed,
                        options=tf.RunOptions(trace_level=self.trace_level),
                        run_metadata=run_metadata
                    )

                    if blk == 0 and self.trace_level > tf.RunOptions.NO_TRACE:
                        gs = self.get_global_step()
                        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                        trace_file = open(os.path.join(self.log_dir, 'timeline.%d' % gs + timeline_suffix + '.json'), 'w')
                        trace_file.write(trace.generate_chrome_trace_format())
                        self.train_writer.add_run_metadata(run_metadata, "step%d" % gs, global_step=gs)
                        self.logger.info("%4d running full trace!!!", gs)
                    for i, val in enumerate(vals):
                        sol[i].append(val)
                    return sol
            except Exception as ex:
                if isinstance(ex, KeyboardInterrupt):
                    raise
                else:
                    self.logger.error('\r=== ERROR RNN ROLL, retrying %d ===\n', retry, exc_info=True)
                    continue

    def _inc_gs(self):
        self.sess.run(self.inc_gs)

    def get_global_step(self):
        return self.sess.run(self.global_step)

    def train_minibatch(self, log_every=20, trace_every=10000):
        """
            Trains minibatch and performs all required operations

            Args:
                log_every: how many time steps to log train_writer and stdout
                trace_every: how many time steps to perform full trace execution (slow). It always performs it at global step 25.
        """
        with self.g.as_default():
            is_training(True, session=self.sess)
        tt = perf_counter()
        iter_step, _ = self.sess.run([self.global_step, self.load_train])
        self.sess.run([self.inc_gs])
        self.dequeue_time = 0.8 * self.dequeue_time + 0.2 * (perf_counter() - tt)

        if trace_every > 0:
            if (iter_step > 0 and iter_step % trace_every == 0) or iter_step == 25:
                self.trace_level = tf.RunOptions.FULL_TRACE

        self.bbt = 0.8 * (self.bbt) + 0.2 * (perf_counter() - self.bbt_clock)
        tt = perf_counter()
        fetches = [self.train_op, self.loss, self.reg]
        vals = self._rnn_roll(add_fetch=fetches, timeline_suffix="ctc_loss")
        self.batch_time = 0.8 * self.batch_time + 0.2 * (perf_counter() - tt)
        self.bbt_clock = perf_counter()

        loss = np.sum(vals[1]).item()
        reg_loss = np.sum(vals[2]).item()

        self.train_writer.add_summary(tf.Summary(value=[
            tf.Summary.Value(tag="train/loss", simple_value=loss),
            tf.Summary.Value(tag="train/reg_loss", simple_value=reg_loss),
            tf.Summary.Value(tag="input/batch_time", simple_value=self.batch_time),
            tf.Summary.Value(tag="input/dequeue_time", simple_value=self.dequeue_time),
            tf.Summary.Value(tag="input/between_batch_time", simple_value=self.bbt),
        ]), global_step=iter_step)

        if iter_step % log_every == 0:
            train_summ, y_len = self.sess.run([self.train_summ, self.Y_len])
            self.train_writer.add_summary(train_summ, global_step=iter_step)

            self.logger.info("%4d loss %6.3f reg_loss %6.3f bt %.3f, bbt %.3f, avg_y_len %.3f dequeue %.3fs", iter_step, loss, reg_loss, self.batch_time, self.bbt, np.mean(y_len), self.dequeue_time)

        self.trace_level = tf.RunOptions.NO_TRACE
        return loss

    def run_validation(self, num_batches=5):
        """
        Runs validation.

        Args:
        num_batches: Number of batches to run validation.

        Returns:
            A tuple (average loss, average edit distance) on validation set
        """
        with self.g.as_default():
            is_training(False, session=self.sess)
        losses = []
        reg_losses = []
        edit_distances = []
        for _ in range(num_batches):
            _, iter_step, summ = self.sess.run([self.load_test, self.global_step, self.test_queue_size])
            self.test_writer.add_summary(summ, global_step=iter_step)
            loss, reg_loss, edit_distance = self._rnn_roll(
                add_fetch=[self.loss, self.reg, self.edit_distance],
                timeline_suffix="ctc_val_loss"
            )
            losses.append(np.sum(loss))
            reg_losses.append(np.sum(reg_loss))
            edit_distances.append(edit_distance)
        avg_loss = np.mean(losses).item()
        avg_reg_loss = np.mean(reg_losses).item()
        avg_edit_distance = np.mean(edit_distances).item()
        self.test_writer.add_summary(tf.Summary(value=[
            tf.Summary.Value(tag="train/loss", simple_value=avg_loss),
            tf.Summary.Value(tag="train/reg_loss", simple_value=avg_reg_loss),
            tf.Summary.Value(tag="train/edit_distance", simple_value=avg_edit_distance),
        ]), global_step=iter_step)
        self.logger.info("%4d validation loss %6.3f edit_distance %.3f in %.3fs" % (iter_step, avg_loss, avg_edit_distance, perf_counter() - self.bbt_clock))
        self.bbt_clock = perf_counter()
        return avg_loss, avg_edit_distance

    def basecall_sample(self, fast5_path):
        with self.g.as_default():
            is_training(False, session=self.sess)
        signal = self.in_data.get_signal(fast5_path)
        t = perf_counter()
        basecalled = self.sess.run(
            self.dense_pred,
            feed_dict={
            self.X_batch: signal,
            self.X_batch_len: np.array(signal.shape[1]).reshape([1,]),
            self.block_idx: 0,
            self.batch_size_var: 1,
            self.block_size_x_tensor: signal.shape[1]
        }).ravel()
        self.logger.debug("Basecalled %s in %.3f", fast5_path, perf_counter() - t)

        return "".join(util.decode(basecalled))

    def skim_trainables(self):
        vars = self.g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        vals = self.sess.run(vars)
        for var, val in zip(vars, vals):
            print(var.op.name, val.ravel()[:5])


    def get_aligement(self, fast5_path, ref_path, verbose):
        t = perf_counter()
        basecalled = self.basecall_sample(fast5_path)
        with open(ref_path) as f:
            target = f.readlines()[-1]

        if verbose:
            print("Basecalled:\n", basecalled)
            print("Target\n", target)
        self.logger.debug("fast5_path %s, ref_path %s", fast5_path, ref_path)
        self.logger.debug("Basecalled \n%s", basecalled)
        self.logger.debug("Target \n%s", target)

        result = Edlib().align(basecalled, target)
        self.logger.debug("Aligment %s", "".join(map(str, result.alignment)))
        self.logger.debug("Whole time %.3f", perf_counter() - t)

        acc = np.sum(np.array(result.alignment) == Edlib().EDLIB_EDOP_MATCH) / len(result.alignment)
        nedit = result.edit_distance / len(target)
        return nedit, acc

    def run_validation_full(self, frac, verbose=False):
        """
            Runs full validation on test set with whole sequence_length
            Args:
                frac: fraction of test set to evaluate, or number of test cases if int
        """

        with open(os.path.join(input_readers.root_dir_default, 'test.txt')) as f:
            fnames = np.array(list(map(lambda x: x.strip().split()[0], f.readlines())))
        np.random.shuffle(fnames)

        if isinstance(frac, (int)):
            fnames = fnames[:frac]
        else:
            fnames = fnames[:int(len(fnames) * frac)]
        self.logger.info("Running validation on %d examples", len(fnames))

        nedit = np.zeros(fnames.shape, dtype=np.float32)
        acc = np.zeros(fnames.shape, dtype=np.float32)
        n = len(acc)
        total_time = 0.0
        with self.g.as_default():
            is_training(False, session=self.sess)
            for i, fname in enumerate(fnames):
                t = perf_counter()
                nedit[i], acc[i] = self.get_aligement(
                    os.path.join(
                        input_readers.root_dir_default,
                        'pass',
                        fname + ".fast5"
                    ), os.path.join(
                        input_readers.root_dir_default,
                        'ref',
                        fname + ".ref"
                    ), verbose=verbose)
                total_time += perf_counter() - t
                mu_edit, mu_acc = np.mean(nedit[:i + 1]), np.mean(acc[:i + 1])
                std_edit, std_acc = np.std(nedit[:i + 1]), np.std(acc[:i + 1])
                se_edit, se_acc = std_edit / np.sqrt(i + 1), std_acc / np.sqrt(i + 1)
                if i % 5 == 0 or i < 5:
                    print("\r%4d/%d avg edit %.4f s %.4f CI <%.4f, %.4f> avg_acc %.4f s %.4f CI <%.4f, %.4f> tps %.3f" % (i + 1, n, mu_edit, std_edit, mu_edit - 2*se_edit, mu_edit + 2*se_edit, mu_acc, std_acc, mu_acc - 2*se_acc, mu_acc + 2*se_acc, total_time/(i + 1)), end='')

        mu_edit, mu_acc = np.mean(nedit), np.mean(acc)
        std_edit, std_acc = np.std(nedit), np.std(acc)
        se_edit, se_acc = std_edit / np.sqrt(i + 1), std_acc / np.sqrt(i + 1)

        self.logger.info("step: %d [samples %d] avg edit %.4f s %.4f CI <%.4f, %.4f> avg_acc %.4f s %.4f CI <%.4f, %.4f> tps %.3f", self.get_global_step(), n, mu_edit, std_edit, mu_edit - 2*se_edit, mu_edit + 2*se_edit, mu_acc, std_acc, mu_acc - 2*se_acc, mu_acc + 2*se_acc, total_time/n)

        return {
            'edit': {
                'mu': mu_edit.item(),
                'std': std_edit.item(),
                'se': se_edit.item()
            },
            'accuracy': {
                'mu': mu_acc.item(),
                'std': std_acc.item(),
                'se': se_acc.item()
            }
        }

    def summarize(self, write_example=True):
        with self.g.as_default():
            is_training(False, session=self.sess)
        iter_step = self.get_global_step()
        fetches = [self.loss, self.grad_summ, self.activation_summ, self.edit_distance]
        if write_example:
            fetches.append(self.pred)
        vals = self._rnn_roll(fetches)

        edit_distance = np.mean(vals[3]).item()
        summaries = [x[0] for x in vals[1:3]]
        for summary in summaries:
            self.train_writer.add_summary(summary, global_step=iter_step)

        self.train_writer.add_summary(tf.Summary(value=[
            tf.Summary.Value(tag="train/edit_distance", simple_value=edit_distance)
        ]), global_step=iter_step)

        if write_example:
            out_net = list(map(lambda x: decode_sparse(x)[0], vals[-1]))
            target = self.decode_target(0, pad=self.block_size_y)
            for a, b in zip(out_net, target):
                print("OUTPUT:", a)
                print("TARGET:", b)
                print('----')

    def decode_target(self, idx, pad=None):
        yy, yy_len = self.sess.run([self.Y, self.Y_len])
        return decode_example(yy[idx], yy_len[idx], self.num_blocks, self.block_size_y, pad=pad)

    def eval_x(self, X, X_len):
        with self.g.as_default():
            is_training(False, session=self.sess)
        batch_size = X.shape[0]

        feed = {
            self.batch_size_var: batch_size,
            self.X: X,
            self.X_len: X_len
        }

        vals = self._rnn_roll([self.pred], feed, timeline_suffix="eval_x")
        return np.array([decode_sparse(ff) for ff in vals]).T

    def save(self):
        self.saver.save(
            self.sess,
            os.path.join(self.log_dir, 'model.ckpt'),
            global_step=self.get_global_step()
        )
        self.logger.info("%4d saved checkpoing", self.get_global_step())

    def restore(self, checkpoint=None, must_exist=True):
        """
            Args:
                checkpoint: filename to restore, default to last checkpoint
        """
        checkpoint = checkpoint or tf.train.latest_checkpoint(self.log_dir)
        self.logger.info("Checkpoint %s", checkpoint)
        if checkpoint is None:
            if must_exist:
                raise ValueError("No checkpoints found")
            iter_step = self.get_global_step()
            self.logger.info("%4d Restored to checkpoint %s" % (iter_step, checkpoint))
        else:
            self.saver.restore(self.sess, checkpoint)
            iter_step = self.get_global_step()
            self.logger.info("%4d Restored to checkpoint %s" % (iter_step, checkpoint))
        return iter_step


    def _start_queues(self, num_workers, proc):
        self.logger.info("Using %s reading class", type(self.in_data).__name__)

        def __queue_feeder_thread(enqueue_op, fun, args, proc):
            """ Proc = True is GIL workaround """
            def thread_fn():
                if proc:
                    q = multiprocessing.Queue(5)
                    p = multiprocessing.Process(target=input_readers.proc_wrapper, args=(q, fun, *args))
                    p.start()
                    gen_next = lambda: q.get()
                else:
                    gen_fn = fun(*args)
                    gen_next = lambda: next(gen_fn)
                while not self.coord.should_stop():
                    feed = gen_next()
                    feed = {self.__dict__[k]: v for k, v in feed.items()}
                    try:
                        self.sess.run(enqueue_op, feed_dict=feed)
                    except tf.errors.CancelledError:
                        print("closing queue_feeder")
                        break
                if proc:
                    p.terminate()
            return Thread(target=thread_fn, daemon=True)

        def data_thread_fn(enqueue_op, file_list):
            return __queue_feeder_thread(
                enqueue_op,
                self.in_data.input_fn(),
                self.in_data.fn_args(self, file_list),
                proc=proc
            )

        self.feed_threads = [data_thread_fn(self.enqueue_train, 'train.txt') for _ in range(num_workers)]
        self.feed_threads.extend([data_thread_fn(self.enqueue_test, 'test.txt') for _ in range(num_workers)])
        for feed_thread in self.feed_threads:
            feed_thread.start()

    def init_session(self, proc=True, num_workers=3, start_queues=True):
        self.trace_level = tf.RunOptions.NO_TRACE
        self.batch_time = 0
        self.dequeue_time = 0
        self.bbt = 0
        self.bbt_clock = perf_counter()

        with self.g.as_default():
            self.sess = tf.Session(
                graph=self.g,
                config=tf.ConfigProto(log_device_placement=False)
            )
            self.sess.run(tf.global_variables_initializer())
            self.coord = tf.train.Coordinator()
            self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        self.g.finalize()
        self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'train'), graph=self.g)
        self.test_writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'test'), graph=self.g)
        self.feed_threads = []
        if start_queues:
            self._start_queues(num_workers, proc)

    def simple_managed_train_model(self, num_steps, val_every=250, save_every=5000, summarize=True, final_val_samples=500, num_workers=3, trace_every=10000, **kwargs):
        try:
            self.logger.info("Training %d steps", num_steps)
            if trace_every < 0:
                self.logger.warn("Profiling tracing is disabled!!!")
            self.init_session(num_workers=num_workers)
            for i in range(self.restore(must_exist=False) + 1, num_steps + 1):
                print('\r%s Step %4d, loss %7.4f batch_time %.3f bbt %.3f dequeue %.3f  ' % (self.run_id, i, self.train_minibatch(trace_every=trace_every), self.batch_time, self.bbt, self.dequeue_time), end='')
                if i > 0 and i % val_every == 0:
                    self.run_validation()
                    if summarize:
                        self.summarize(write_example=False)
                if i > 0 and i % save_every == 0:
                    self.save()

            self.save()
            self.logger.info("Running final validation run")
            return self.run_validation_full(final_val_samples)
        except Exception as ex:
            if not isinstance(ex, KeyboardInterrupt):
                self.logger.error("Error happened", exc_info=1)
            raise
        finally:
            self.train_writer.flush()
            self.test_writer.flush()
            self.close_session()

    def close_session(self):
        self.coord.request_stop()
        self.train_writer.flush()
        self.test_writer.flush()
        self.sess.run(self.close_queues)
        for feed_thread in self.feed_threads:
            feed_thread.join(5)
            if feed_thread.is_alive():
                self.logger.error("Thread is still alive!" + str(feed_thread))

        self.coord.join(self.threads)
        self.sess.close()


#  RNN based teacher not supported
class TeacherStudentModel(Model):
    def __init__(self, teacher_names, teacher_dirs, model_fn, hyper, **kwargs):
        self.teacher_params = [load_model_parms(teacher_name, teacher_dir) for teacher_name, teacher_dir in zip(teacher_names, teacher_dirs)]
        self.ctc_scale = hyper['ctc_scale']
        super().__init__(model_fn=model_fn, hyper=hyper, **kwargs)

    def _setup_teachers(self):
        if hasattr(self, 'teachers'):
            self.logger.warn("_setup_teachers CALLED MORE THAN ONCE")
        self.teachers = []
        for i, teacher_params in enumerate(self.teacher_params):
            with tf.variable_scope("teacher%d" % i):
                teacher = super()._setup_logits(teacher_params['model_fn'], teacher_params['hyper'])

            mapping = {'/'.join(v.op.name.split('/')[1:]): v for v in self.g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'teacher%d' % i)
            }
            self.logger.debug("Teacher %d uses following varibles", i)
            for k in mapping.keys():
                self.logger.debug(k)
            teacher['saver'] = tf.train.Saver(mapping)
            teacher['log_dir'] = teacher_params['log_dir']
            self.logger.info("Configured teacher%d, logdir %s", i, teacher['log_dir'])
            self.teachers.append(teacher)

    def _get_ensamble(self):
        self._setup_teachers()
        ensamble = [tf.nn.softmax(teacher['logits']) for teacher in self.teachers]
        max_len = tf.reduce_max(tf.stack([tf.shape(x)[0] for x in ensamble]))
        min_len = tf.reduce_min(tf.stack([tf.shape(x)[0] for x in ensamble]))

        with tf.control_dependencies([
            tf.assert_less_equal(max_len - min_len, 1, message="Maximum padding limit exceeded")
        ]):
            ensamble = [tf.pad(x, [
                [0, max_len - tf.shape(x)[0]],
                [0, 0],
                [0, 0]
                ]) for x in ensamble]
            ensamble = tf.stack(ensamble)
            ensamble = tf.reduce_mean(ensamble, axis=0)
        return ensamble

    def _setup_loss(self, logits):
        self.ctc_loss = super()._setup_loss(logits)
        ensamble = self._get_ensamble()
        with tf.name_scope("loss"):
            self.ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ensamble, logits=logits))
            return self.ce_loss + self.ctc_scale * self.ctc_loss

    def _restore_teachers(self):
        self.print_k()
        for i, teacher in enumerate(self.teachers):
            checkpoint = tf.train.latest_checkpoint(teacher['log_dir'])
            if checkpoint is None:
                raise ValueError("Teacher checkpoint must exist!")
            self.logger.info("Loading teacher%d weights from %s", i, checkpoint)
            teacher['saver'].restore(self.sess, checkpoint)
        self.print_k()

    def init_session(self, *args, **kwargs):
        super().init_session(*args, **kwargs)
        self._restore_teachers()

    def print_k(self):
        for i in range(len(self.teacher_params)):
            vars = [v for v in self.g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'teacher%d' % i) if '/k' in v.op.name]
            vals = self.sess.run(vars)
            for var, val in zip(vars, vals):
                self.logger.debug("%s: %s", var.op.name, val.ravel()[:5])


class Ensamble(TeacherStudentModel):
    def __init__(self, teacher_names, teacher_dirs):
        from model_small import default_params, model_setup_params
        params = model_setup_params(default_params)
        params['hyper']['ctc_scale'] = 0.0
        super().__init__(teacher_names, teacher_dirs, **params)

    def train_minibatch(self, *args, **kwargs):
        raise NotImplementedError()

    def _setup_loss(self, logits):
        pass

    def _setup_train(self, *args, **kwargs):
        pass

    def _setup_logits(self, *args):
        return {
            'logits': None,
            'init_state': tf.constant(0.0),
            'final_state': tf.constant(0.0)
        }

    def _predict_from_logits(self):
        ens = self._get_ensamble()
        self.logits = tf.log(ens)
        return super()._predict_from_logits()
