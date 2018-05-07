from keras import backend as K
from keras.layers import Conv1D
import logging
import tensorflow as tf

class Model():
    def __init__(self, cfg, labels, input_signal, signal_len):
        self.logger = logging.getLogger(__name__)
        learning_phase = K.learning_phase()

        net = input_signal
        net = Conv1D(5, 3, input_shape=(cfg.batch_size, cfg.seq_length, 1), padding="same")(net)

        net = net # Tensor of shape [batch_size, max_time, class_num]
        self.logits = tf.transpose(net, [1, 0, 2])  # [max_time, batch_size, class_num]
        self.logger.info(f"Logits shape: {self.logits.shape}")

        ratio = 1
        self.logits = tf.Print(self.logits, [self.logits, tf.shape(self.logits), tf.shape(input_signal), labels.indices, labels.values, labels.dense_shape], message="varios debug out")
        seq_len = tf.cast(tf.floor_div(signal_len + ratio - 1, ratio), tf.int32)  # Round up
        seq_len = tf.Print(seq_len, [tf.shape(seq_len), seq_len], message="seq len")
        self.loss = tf.reduce_mean(tf.nn.ctc_loss(
            labels=labels,
            inputs=self.logits,
            sequence_length=seq_len,
            preprocess_collapse_repeated=False,
            ctc_merge_repeated=True,
            time_major=True,
        ))

        tf.add_to_collection('losses', self.loss)
        tf.summary.scalar('loss', self.loss)

        self.total_loss = tf.add_n(tf.get_collection('losses'),name = 'total_loss')
        self.train_step = tf.train.AdadeltaOptimizer().minimize(self.total_loss)

        self.predict = tf.nn.ctc_beam_search_decoder(
            inputs=self.logits,
            sequence_length=seq_len,
            merge_repeated=False,
            top_paths=1,
            beam_width=50
        )[0][0]
