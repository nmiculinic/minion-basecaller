import numpy as np
import util
import socket
import tensorflow as tf
from util import dense2d_to_sparse, decode_example, decode_sparse
import input_readers
import multiprocessing
from threading import Thread
import time
from tensorflow.python.client import timeline
import os
import string
import random
import shutil
import warpctc_tensorflow
from tflearn.summaries import add_gradients_summary, add_activations_summary

repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))


class Model():
    def __init__(self, g, num_blocks, batch_size, max_reach, model_fn, block_size=None, block_size_x=None, block_size_y=None, log_dir=None, run_id=None, overwrite=False, reuse=False, queue_cap=None, shrink_factor=1, in_data="EVENTS"):
        """
            Args:
                max_reach: int, size of contextual window for convolutions etc.
                model_fn: function accepting (batch_size, 2*max_reach + block_size, 3) -> (block_size, batch_size, out_classes). Notice shift to time major as well as reduction in time dimension.
        """

        if in_data not in ["RAW", "EVENTS"]:
            raise ValueError("in_data must be one of two types")
        self.in_data = in_data
        self.data_in_dim = 1 if in_data == "RAW" else 3

        if overwrite and reuse:
            raise ValueError("Cannot overwrite and reuse logdit and checkpoints")

        self.__handle_logdir(log_dir, run_id, overwrite, reuse)
        self.block_size_y = block_size_y or block_size
        self.block_size_x = block_size_x or block_size
        self.shrink_factor = shrink_factor
        if self.block_size_x % shrink_factor != 0:
            raise ValueError("shrink factor need to divide block_size_x")
        self.num_blocks = num_blocks
        self.batch_size = batch_size
        self.batch_size_var = tf.placeholder_with_default(tf.convert_to_tensor(batch_size, dtype=tf.int32), [])
        self.max_reach = max_reach
        self.queue_cap = queue_cap or 5 * self.self.batch_size

        with g.as_default():
            net = self.__create_train_input_objects()
            with tf.name_scope("model"):
                data = model_fn(
                    net,
                    self.X_batch_len,
                    max_reach=self.max_reach,
                    block_size=self.block_size_x,
                    out_classes=9,
                    batch_size=self.batch_size_var
                )

            for k, v in data.items():
                self.__dict__[k] = v

            print("logits: ", self.logits.get_shape())
            with tf.control_dependencies([
                tf.assert_equal(tf.shape(self.logits), [self.block_size_x // self.shrink_factor, self.batch_size_var, 9])
            ]):
                self.logits = tf.identity(self.logits)

            with tf.name_scope("loss"):
                loss = warpctc_tensorflow.ctc(
                    self.logits,
                    self.Y_batch.values,
                    self.Y_batch_len,
                    tf.div(self.X_batch_len, self.shrink_factor),
                    blank_label=8
                )

                loss = tf.reduce_mean(loss)
                self.loss = loss

            with tf.name_scope("prediction"):
                predicted, prdicted_logprob = tf.nn.ctc_beam_search_decoder(self.logits, tf.div(self.X_batch_len, self.shrink_factor), merge_repeated=True, top_paths=1)
                self.pred = tf.cast(predicted[0], tf.int32)

            optimizer = tf.train.AdamOptimizer()
            self.grads = optimizer.compute_gradients(loss)
            self.train_op = optimizer.apply_gradients(self.grads)

            self.grad_summ = tf.summary.merge(add_gradients_summary(self.grads))
            self.activation_summ = tf.summary.merge(
                add_activations_summary(g.get_collection("activations")))

        self.g = g
        self.trace_level = tf.RunOptions.NO_TRACE
        self.saver = tf.train.Saver(
            keep_checkpoint_every_n_hours=1,
            var_list=tf.trainable_variables()
        )

        self.batch_time = 0
        self.dequeue_time = 0

        self.bbt = 0
        self.bbt_clock = time.clock()

    def __handle_logdir(self, log_dir, run_id, overwrite, reuse):
        log_dir = log_dir or os.path.join(repo_root, 'log', socket.gethostname())
        run_id = run_id or ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))

        self.run_id = run_id
        self.log_dir = os.path.join(log_dir, self.run_id)

        if os.path.exists(self.log_dir):
            if overwrite:
                shutil.rmtree(self.log_dir)
            elif not reuse:
                raise ValueError("path " + self.log_dir + " exists")

        os.makedirs(self.log_dir, mode=0o744, exist_ok=reuse or overwrite)
        print("Logdir = ", self.log_dir)

    def __create_train_input_objects(self):
        with tf.variable_scope("input"):
            input_vars = [
                tf.get_variable("X", initializer=tf.zeros_initializer([self.batch_size, self.block_size_x * self.num_blocks, self.data_in_dim], tf.float32), trainable=False),
                tf.get_variable("X_len", initializer=tf.zeros_initializer([self.batch_size], tf.int32), trainable=False),
                tf.get_variable("Y", initializer=tf.zeros_initializer([self.batch_size, self.block_size_y * self.num_blocks], tf.uint8), trainable=False),
                tf.get_variable("Y_len", initializer=tf.zeros_initializer([self.batch_size, self.num_blocks], tf.int32), trainable=False),
            ]
            names = [x.name[6:-2] for x in input_vars]  # TODO, hacky
            shapes = [x.get_shape()[1:] for x in input_vars]
            types = [x.dtype.base_dtype for x in input_vars]

            with tf.name_scope("queue_handling"):
                self.queue = tf.FIFOQueue(self.queue_cap, types, shapes=shapes)
                self.queue_size = tf.summary.scalar("queue_filled", self.queue.size())
                self.close_queue = self.queue.close(True, "close_queue")

                self.dequeue_op = self.queue.dequeue_many(self.batch_size)
                for name, x, qx in zip(names, input_vars, self.dequeue_op):
                    self.__dict__[name] = x
                    feed = tf.placeholder_with_default(qx, shape=x.get_shape(), name=name + "_feed")
                    self.__dict__[name + "_feed"] = feed
                    self.__dict__[name + "_assign"] = tf.assign(x, feed, name=name + "_assign")
                    self.__dict__[name + "_enqueue_val"] = tf.placeholder(x.dtype.base_dtype, shape=[None, *x.get_shape()[1:]], name=name + "_enqueue_val")

                self.enqueue_op = self.queue.enqueue_many([self.__dict__[name + "_enqueue_val"] for name in names])
                self.load_queue = tf.group(*[v for k, v in self.__dict__.items() if '_assign' in k])

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
                net.set_shape([None, 2 * self.max_reach + self.block_size_x, self.data_in_dim])
                print(net.get_shape())
                self.X_batch = net

            with tf.name_scope("Y_batch"):
                self.Y_batch_len = tf.squeeze(tf.slice(self.Y_len, [0, self.block_idx], [self.batch_size_var, 1]), [1])
                # TODO: migrate to batch_size_var
                self.Y_batch = dense2d_to_sparse(tf.slice(self.Y, [0, begin_y], [self.batch_size, self.block_size_y]), self.Y_batch_len, dtype=tf.int32)
        return net

    def queue_feeder_proc(self, fun, args, proc=False):
        """ Proc = True is GIL workaround """
        def thread_fn():
            if proc:
                q = multiprocessing.Queue()
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
                    self.sess.run(self.enqueue_op, feed_dict=feed)
                except tf.errors.CancelledError:
                    break
            if proc:
                p.terminate()
        return Thread(target=thread_fn, daemon=True)

    def __rnn_roll(self, add_fetch=[], add_feed={}, timeline_suffix=""):
        if 'sess' not in self.__dict__:
            raise ValueError("session not initialized")

        fetch = [self.final_state]
        fetch.extend(add_fetch)
        run_metadata = tf.RunMetadata()
        sol = []
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

            if self.trace_level > tf.RunOptions.NO_TRACE:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open(os.path.join(self.log_dir, 'timeline.' + timeline_suffix + '.json'), 'w')
                trace_file.write(trace.generate_chrome_trace_format())
            sol.append(vals)
        return sol

    def train_minibatch(self):
        tt = time.clock()
        self.sess.run(self.load_queue)
        self.dequeue_time = 0.9 * self.dequeue_time + 0.1 * (time.clock() - tt)

        self.bbt = 0.8 * self.bbt + 0.2 * (time.clock() - self.bbt_clock)
        tt = time.clock()
        self.__rnn_roll(add_fetch=[self.train_op], timeline_suffix="ctc_loss")
        self.batch_time = 0.8 * self.batch_time + 0.2 * (time.clock() - tt)
        self.bbt_clock = time.clock()

    def summarize(self, iter_step, full=False, write_example=True):
        state, queue_size_sum, y_len = self.sess.run([self.init_state, self.queue_size, self.Y_len])
        print("avg time per batch %.3f, avg_y_len = %.3f, load_time_op %.3f, between batch time %.3f" % (self.batch_time, np.mean(y_len), self.dequeue_time, self.bbt))

        fetches = [self.loss, self.pred]
        if full:
            fetches.extend([self.grad_summ, self.activation_summ])
        vals = self.__rnn_roll(fetches)

        loss_val = np.sum(ff[0] for ff in vals)
        out_net = [decode_sparse(ff[1])[0] for ff in vals]
        summaries = vals[0][2:]

        print("[%s] %4d %6.3f (output -> up, target -> down)" % (self.run_id, iter_step, loss_val))
        if write_example:
            target = self.decode_target(0, pad=self.block_size_y)
            for a, b in zip(out_net, target):
                print(a)
                print(b)
                print('----')
        for summary in summaries:
            self.train_writer.add_summary(summary, global_step=iter_step)
        self.train_writer.add_summary(queue_size_sum, global_step=iter_step)
        self.train_writer.add_summary(tf.Summary(value=[
            tf.Summary.Value(tag="loss", simple_value=loss_val.item()),
            tf.Summary.Value(tag="input/time_per_batch", simple_value=self.batch_time),
            tf.Summary.Value(tag="input/dequeue_time", simple_value=self.dequeue_time),
            tf.Summary.Value(tag="input/between_batch_time", simple_value=self.bbt),
        ]), global_step=iter_step)

    def decode_target(self, idx, pad=None):
        yy, yy_len = self.sess.run([self.Y, self.Y_len])
        return decode_example(yy[idx], yy_len[idx], self.num_blocks, self.block_size_y, pad=pad)

    def eval_x(self, X, X_len):
        batch_size = X.shape[0]

        feed = {
            self.batch_size_var: batch_size,
            self.X: X,
            self.X_len: X_len
        }

        vals = self.__rnn_roll([self.pred], feed, timeline_suffix="eval_x")
        return np.array([decode_sparse(ff) for ff in vals]).T

    def save(self, iter_step):
        self.saver.save(
            self.sess,
            os.path.join(self.log_dir, 'model.ckpt'),
            global_step=iter_step
        )

    def init_session(self, restore=False, checkpoint=None, proc=True, num_workers=4):
        self.sess = tf.Session(
            graph=self.g,
            config=tf.ConfigProto(log_device_placement=False)
        )
        if restore:
            checkpoint = checkpoint or tf.train.latest_checkpoint(self.log_dir)
            if checkpoint is None:
                raise ValueError("No checkpoints found")
            iter_step = int(checkpoint.split('-')[-1])
            self.saver.restore(self.sess, checkpoint)
            print("Restored to checkpoint", checkpoint)
        else:
            self.sess.run(tf.global_variables_initializer())
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

        self.g.finalize()
        self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'train'), graph=self.g)

        if self.in_data == "EVENTS":
            data_thread_fn = lambda:\
                self.queue_feeder_proc(input_readers.get_feed_yield2, [self.block_size_x, self.num_blocks, 10], proc=proc)
        elif self.in_data == "RAW":
            data_thread_fn = lambda:\
                self.queue_feeder_proc(input_readers.get_raw_feed_yield, [self.block_size_x, self.block_size_y, self.num_blocks, 10], proc=proc)
        else:
            raise ValueError("WTF " + self.in_data)

        self.feed_threads = [data_thread_fn() for _ in range(num_workers)]
        for feed_thread in self.feed_threads:
            feed_thread.start()

        return iter_step if restore else 0

    def close_session(self):
        self.coord.request_stop()
        self.sess.run(self.close_queue)
        self.train_writer.flush()
        for feed_thread in self.feed_threads:
                feed_thread.join()
        self.coord.join(self.threads)
        self.sess.close()
