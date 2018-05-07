from keras import backend as K
from keras import models
import logging
import tensorflow as tf


class Model():
    def __init__(self, cfg, labels, input_signal, signal_len, model: models.Model, trace=False):
        self.logger = logging.getLogger(__name__)
        learning_phase = K.learning_phase()


        self.logits = tf.transpose(
            model(input_signal), [1, 0, 2])  # [max_time, batch_size, class_num]
        self.logger.info(f"Logits shape: {self.logits.shape}")

        ratio = 1
        seq_len = tf.cast(
            tf.floor_div(signal_len + ratio - 1, ratio), tf.int32)  # Round up

        if trace:
            self.logits = tf.Print(
                self.logits, [
                    self.logits,
                    tf.shape(self.logits),
                    tf.shape(input_signal), labels.indices, labels.values,
                    labels.dense_shape
                ],
                message="varios debug out")
            seq_len = tf.Print(
                seq_len, [tf.shape(seq_len), seq_len], message="seq len")

        self.losses = tf.nn.ctc_loss(
            labels=labels,
            inputs=self.logits,
            sequence_length=seq_len,
            preprocess_collapse_repeated=False,
            ctc_merge_repeated=True,
            time_major=True,
        )
        self.ctc_loss = tf.reduce_mean(self.losses)

        tf.add_to_collection('losses', self.ctc_loss)
        tf.summary.scalar('loss', self.ctc_loss)

        self.total_loss = tf.add_n(
            tf.get_collection('losses'), name='total_loss')
        self.train_step = tf.train.AdamOptimizer().minimize(self.total_loss)

        self.predict = tf.nn.ctc_beam_search_decoder(
            inputs=self.logits,
            sequence_length=seq_len,
            merge_repeated=False,
            top_paths=1,
            beam_width=50)[0][0]
