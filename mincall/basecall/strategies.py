import tensorflow as tf
import uuid
import numpy as np
import concurrent.futures
import asyncio
from threading import Thread
import os
from mincall.common import TOTAL_BASE_PAIRS
from keras import models

from mincall.external.tensorflow_serving.apis import predict_pb2
from mincall.external.tensorflow_serving.apis import prediction_service_pb2
from grpc.beta import implementations


class BeamSearchStrategy:
    def beam_search(self, logits) -> concurrent.futures.Future:
        raise NotImplemented


class BeamSearchSess(BeamSearchStrategy):
    def __init__(self, sess: tf.Session, surrogate_base_pair, beam_width):
        self.sess = sess

        if surrogate_base_pair:
            self.logits_ph = tf.placeholder(
                tf.float32, shape=(1, None, 2 * TOTAL_BASE_PAIRS + 1)
            )
        else:
            self.logits_ph = tf.placeholder(
                tf.float32, shape=(1, None, TOTAL_BASE_PAIRS + 1)
            )
        self.seq_len_ph = tf.placeholder_with_default(
            [tf.shape(self.logits_ph)[1]], shape=(1,)
        )  # TODO: Write this sanely

        with tf.name_scope("logits_to_bases"):
            if beam_width > 0:
                self.predict = tf.nn.ctc_beam_search_decoder(
                    inputs=tf.transpose(self.logits_ph, [1, 0, 2]),
                    sequence_length=self.seq_len_ph,
                    merge_repeated=surrogate_base_pair,
                    top_paths=1,
                    beam_width=beam_width,
                )
            elif beam_width == 0:
                self.predict = tf.nn.ctc_greedy_decoder(
                    inputs=tf.transpose(self.logits_ph, [1, 0, 2]),
                    sequence_length=self.seq_len_ph,
                    merge_repeated=surrogate_base_pair,
                )
            else:
                raise ValueError(f"Beam width cannot be <0, got {beam_width}")
            self.predict_values = self.predict[0][0].values

    def beam_search(self, logits: np.ndarray, loop=None):
        assert len(
            logits.shape
        ) == 2, f"Logits should be rank 2, got shape {logits.shape}"
        f = concurrent.futures.Future()
        f.set_result(
            self.sess.run(
                self.predict_values,
                feed_dict={
                    self.logits_ph: logits[np.newaxis, :, :],
                }
            )
        )
        return f


class BeamSearchQueue:
    def __init__(
        self, sess: tf.Session, coord: tf.train.Coordinator, surrogate_base_pair
    ):
        self.sess = sess
        self.coord = coord
        self.futures = {}
        self.tf_inq = tf.FIFOQueue(
            capacity=10,
            dtypes=[tf.string, tf.float32],
        )

        self.tf_outq = tf.FIFOQueue(
            capacity=10,
            dtypes=[tf.string, tf.int64],
        )

        self.inq_name = tf.placeholder(tf.string)
        self.inq_logits = tf.placeholder(tf.float32)
        self.inq_enqueue = self.tf_inq.enqueue([self.inq_name, self.inq_logits])
        self.inq_close = self.tf_inq.close()

        with tf.name_scope("logits_to_bases"):
            name, logits = self.tf_inq.dequeue()
            self.predict = tf.nn.ctc_beam_search_decoder(
                inputs=tf.transpose(logits, [1, 0, 2]),
                sequence_length=[tf.shape(logits)[1]],
                merge_repeated=surrogate_base_pair,
                top_paths=1,
                beam_width=50
            )
            enq_op = self.tf_outq.enqueue([
                name,
                self.predict[0][0].values,
            ])
        qr = tf.train.QueueRunner(self.tf_outq, [enq_op] * os.cpu_count())
        tf.train.add_queue_runner(qr)
        self.out_dequeue = self.tf_outq.dequeue()
        self.t = Thread(target=self._start, daemon=True)
        self.t.start()

    def _start(self):
        try:
            while True:
                name, ind = self.sess.run(self.out_dequeue,)
                name = name.decode("ASCII")
                f = self.futures[name]
                f.set_result(ind)
                del self.futures[name]
        except tf.errors.OutOfRangeError:
            # Means the underlying queue is closed and we can safely exit
            return
        except Exception as ex:
            self.coord.request_stop(ex)
            raise

    async def beam_search(self, logits, loop=None):
        f = concurrent.futures.Future()
        name = uuid.uuid4().hex
        self.futures[name] = f
        self.sess.run(
            self.inq_enqueue,
            feed_dict={
                self.inq_name: name,
                self.inq_logits: logits[np.newaxis, :, :],
            },
        )
        return await asyncio.wrap_future(f, loop=loop)

    def stop(self):
        self.sess.run(self.inq_close)
        self.t.join(timeout=10)
        if self.t.is_alive():
            raise ValueError("Thread still alive")


class BeamSearchTFServing(BeamSearchStrategy):
    def __init__(
        self,
        host="localhost",
        port=9001,
        name="default",
        signature_name=tf.saved_model.signature_constants.
        DEFAULT_SERVING_SIGNATURE_DEF_KEY
    ):
        self.channel = implementations.insecure_channel(host, int(port))
        print(dir(self.channel))
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(
            self.channel
        )

        # Send request
        request = predict_pb2.PredictRequest()
        request.model_spec.name = name
        request.model_spec.signature_name = signature_name
        self.req = request

    def beam_search(self, logits):
        assert len(
            logits.shape
        ) == 2, f"Logits should be rank 2, got shape {logits.shape}"
        f = concurrent.futures.Future()

        request = predict_pb2.PredictRequest()
        request.CopyFrom(self.req)
        request.inputs['logits'].CopyFrom(
            tf.make_tensor_proto(logits[np.newaxis, :, :]),
        )

        result = self.stub.Predict(request, 120.0)  # 120 secs timeout
        f.set_result(np.array(result.outputs['path'].int64_val))
        return f


class Signal2LogitsSess:
    def __init__(self, sess: tf.Session, model: models.Model):
        self.sess = sess
        with tf.name_scope("signal_to_logits"):
            self.signal_batch = tf.placeholder(
                tf.float32, shape=(None, None, 1), name="signal"
            )
            self.logits = model(
                self.signal_batch
            )  # [batch size, max_time, channels]

    def signal2logit_fn(self, signal: np.ndarray) -> concurrent.futures.Future:
        assert len(
            signal.shape
        ) == 1, f"Signal should be rank 1, shape: {signal.shape}"
        f = concurrent.futures.Future()
        logits = self.sess.run(
            self.logits,
            feed_dict={
                self.signal_batch: signal[np.newaxis, :, np.newaxis],
            }
        )
        logits = np.squeeze(logits, axis=0)
        f.set_result(logits)
        return f


class Signal2LogitQueue:
    """Never Been tested, use at your own risk!
    """

    def __init__(
        self,
        sess: tf.Session,
        coord: tf.train.Coordinator,
        model,
        max_batch_size: int = 10
    ):
        self.sess = sess
        self.coord = coord

        self.futures = {}
        self.tf_inq = tf.PaddingFIFOQueue(
            capacity=10,
            dtypes=[tf.string, tf.float32, tf.int32],
            shapes=[[], [None, 1], []],
        )

        self.tf_outq = tf.FIFOQueue(
            capacity=10,
            dtypes=[tf.string, tf.int64, tf.int32],
        )

        self.inq_name = tf.placeholder(tf.string, shape=[])
        self.inq_signal = tf.placeholder(tf.float32, shape=(None, 1))
        self.inq_length = tf.placeholder(tf.int32, shape=[])
        self.inq_enqueue = self.tf_inq.enqueue([
            self.inq_name, self.inq_signal, self.inq_length
        ])
        self.inq_close = self.tf_inq.close()

        with tf.name_scope("signal2logits"):
            name, signal, signal_len = self.tf_inq.dequeue_up_to(max_batch_size)
            logits = model(signal)
            enq_op = self.tf_outq.enqueue([name, logits, signal_len])
        qr = tf.train.QueueRunner(self.tf_outq, [enq_op] * os.cpu_count())
        tf.train.add_queue_runner(qr)
        self.out_dequeue = self.tf_outq.dequeue()
        self.t = Thread(target=self._start, daemon=True)
        self.t.start()

    def _start(self):
        try:
            while True:
                for name, logits, signal_len in zip(
                    *self.sess.run(self.out_dequeue,)
                ):
                    name = name.decode("ASCII")
                    f = self.futures[name]
                    f.set_result(logits[:signal_len])
                    del self.futures[name]
        except tf.errors.OutOfRangeError:
            # Means the underlying queue is closed and we can safely exit
            return
        except Exception as ex:
            self.coord.request_stop(ex)
            raise

    async def logits(self, signal: np.ndarray, loop=None):
        f = concurrent.futures.Future()
        name = uuid.uuid4().hex
        self.futures[name] = f
        self.sess.run(
            self.inq_enqueue,
            feed_dict={
                self.inq_name: name,
                self.inq_logits: logits[np.newaxis, :, :],
            },
        )
        return await asyncio.wrap_future(f, loop=loop)

    def stop(self):
        self.sess.run(self.inq_close)
        self.t.join(timeout=10)
        if self.t.is_alive():
            raise ValueError("Thread still alive")
