import numpy as np
import util
import socket
import tensorflow as tf
from util import dense2d_to_sparse, decode_example, decode, decode_sparse
import input_readers
import multiprocessing
from threading import Thread
import time
from tensorflow.python.client import timeline


class Model():
    def __init__(self, g, block_size, num_blocks, batch_size, max_reach, model_fn):
        """
            Args:
                max_reach: int, size of contextual window for convolutions etc.
                model_fn: function accepting (batch_size, 2*max_reach + block_size, 3) -> (block_size, batch_size, out_classes). Notice shift to time major as well as reduction in time dimension.
        """
        self.block_size = block_size
        self.num_blocks = num_blocks
        with g.as_default():
            with tf.variable_scope("input"):
                input_vars = [
                    tf.get_variable("X", initializer=tf.zeros_initializer([batch_size, block_size * num_blocks, 3], tf.float32), trainable=False),
                    tf.get_variable("X_len", initializer=tf.zeros_initializer([batch_size], tf.int32), trainable=False),
                    tf.get_variable("Y", initializer=tf.zeros_initializer([batch_size, block_size * num_blocks], tf.uint8), trainable=False),
                    tf.get_variable("Y_len", initializer=tf.zeros_initializer([batch_size, num_blocks], tf.int32), trainable=False),
                ]
                names = [x.name[6:-2] for x in input_vars]
                shapes = [x.get_shape()[1:] for x in input_vars]
                types = [x.dtype.base_dtype for x in input_vars]
                queue = tf.RandomShuffleQueue(5 * batch_size, 2 * batch_size, types, shapes=shapes)

                ops = {}
                self.dequeue_op = queue.dequeue_many(batch_size)
                for name, x, qx in zip(names, input_vars, self.dequeue_op):
                    ops[name] = x
                    feed = tf.placeholder_with_default(qx, shape=x.get_shape(), name=name + "_feed")
                    ops[name + "_feed"] = feed
                    ops[name + "_assign"] = tf.assign(x, feed, name=name + "_assign")
                    ops[name + "_enqueue_val"] = tf.placeholder(x.dtype.base_dtype, shape=[None, *x.get_shape()[1:]], name=name + "_enqueue_val")

                enqueue_op = queue.enqueue_many([ops[name + "_enqueue_val"] for name in names])
                ops['load_queue'] = tf.group(*[v for k, v in ops.items() if '_assign' in k])

                self.enqueue_op = enqueue_op

                block_idx = tf.placeholder(dtype=tf.int32, shape=[], name="block_idx")
                self.block_idx = block_idx
                begin = block_idx * block_size

                X = ops['X']
                with tf.control_dependencies([
                    tf.assert_less_equal(begin + block_size, tf.shape(X)[1], message="Cannot request that many elements from X"),
                    tf.assert_non_negative(begin, message="Beginning slice must be >=0"),
                ]):
                    max_len = tf.shape(X)[1]
                    left = tf.maximum(0, begin - max_reach)
                    right = tf.minimum(max_len, begin + block_size + max_reach)
                    X_len = tf.clip_by_value(ops['X_len'] - block_idx * block_size, 0, block_size)

                net = tf.slice(X, [0, left, 0], [-1, right - left, -1])
                # net = tf.Print(net, [left, right], message="LR")
                # net = tf.Print(net, [block_idx, tf.shape(net)], message="before padding")
                padding = [
                    [0, 0],
                    [tf.maximum(0, max_reach - begin), tf.maximum(0, begin + block_size + max_reach - max_len)],
                    [0, 0]
                ]
                padding = tf.convert_to_tensor(padding)
                # net = tf.Print(net, [padding], message="padding")
                net = tf.pad(net, padding)
                # net = tf.Print(net, [tf.shape(net)], message="after padding")
                net.set_shape([batch_size, 2 * max_reach + block_size, 3])

            data = model_fn(
                net,
                X_len,
                max_reach=max_reach,
                block_size=block_size,
                out_classes=9,
                batch_size=batch_size
            )

            for k, v in data.items():
                ops[k] = v

            logits = ops['logits']
            print("logits: ", logits.get_shape())

            with tf.name_scope("loss"):
                Y_len = tf.squeeze(tf.slice(ops['Y_len'], [0, block_idx], [batch_size, 1]), [1])
                Y = dense2d_to_sparse(tf.slice(ops['Y'], [0, begin], [batch_size, block_size]), Y_len, dtype=tf.int32)

                loss = tf.nn.ctc_loss(inputs=logits, labels=Y, sequence_length=X_len, time_major=True)
                loss = tf.reduce_mean(loss)
                self.loss = loss

            predicted, prdicted_logprob = tf.nn.ctc_beam_search_decoder(logits, X_len, merge_repeated=True, top_paths=1)
            pred = tf.sparse_tensor_to_dense(tf.cast(predicted[0], tf.int32))
            pred = predicted[0]

            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(loss)
            grads = optimizer.compute_gradients(loss)

            self.grads = grads
            self.train_op = train_op
            self.pred = pred

        self.ops = ops
        for k, v in self.ops.items():
            self.__dict__[k] = v

        self.g = g
        self.trace_level = tf.RunOptions.NO_TRACE
        self.batch_time = 0

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
                feed = {self.ops[k]: v for k, v in feed.items()}
                self.sess.run(self.enqueue_op, feed_dict=feed)
            if proc:
                p.terminate()
        return Thread(target=thread_fn, daemon=True)

    def train_minibatch(self):
        if 'sess' not in self.__dict__:
            raise ValueError("session not initialized")

        self.sess.run(self.load_queue)
        tt = time.clock()
        state = self.sess.run(self.init_state)
        for blk in range(self.num_blocks):
            run_metadata = tf.RunMetadata()
            loss_val, _, state = self.sess.run([self.loss, self.train_op, self.final_state], feed_dict={
                self.block_idx: blk,
                self.init_state: state
            }, options=tf.RunOptions(trace_level=self.trace_level),
            run_metadata=run_metadata)

            if self.trace_level > tf.RunOptions.NO_TRACE:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open('timeline.ctf_loss.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format())
        self.batch_time = 0.8 * self.batch_time + 0.2 * (time.clock() - tt)

    def summarize(self, iter_step):
        print("avg time per batch %.3f" % self.batch_time)
        state = self.sess.run(self.init_state)
        out_net = []
        loss = 0
        for blk in range(self.num_blocks):
            run_metadata = tf.RunMetadata()

            loss_val, ff, state = self.sess.run([self.loss, self.pred, self.final_state], feed_dict={
                self.block_idx: blk,
                self.init_state: state
            }, options=tf.RunOptions(trace_level=self.trace_level),
            run_metadata=run_metadata)
            out_net.append(decode_sparse(ff, pad=self.block_size)[0])
            loss += loss_val

            if self.trace_level > tf.RunOptions.NO_TRACE:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open('timeline.ctf_decode.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format())

        print("%4d %6.3f (output -> up, target -> down)" % (iter_step, np.sum(loss_val)))
        target = self.decode_target(0, pad=self.block_size)
        for a, b in zip(out_net, target):
            print(a)
            print(b)
            print('----')

        t0 = time.clock()
        self.sess.run(self.load_queue)
        print("loading_time %.3f" % (time.clock() - t0))

    def init_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

        self.feed_threads = [self.queue_feeder_proc(input_readers.get_feed_yield2, [self.block_size, self.num_blocks, 10], proc=True) for _ in range(3)]
        for feed_thread in self.feed_threads:
            feed_thread.start()

    def decode_target(self, idx, pad=None):
        yy, yy_len = self.sess.run([self.Y, self.Y_len])
        return decode_example(yy[idx], yy_len[idx], self.num_blocks, self.block_size, pad=pad)

    def close_session(self):
        self.coord.request_stop()
        for feed_thread in self.feed_threads:
                feed_thread.join()
        self.coord.join(self.threads)
        self.sess.close()
