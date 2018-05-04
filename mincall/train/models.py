from keras import backend as K
from keras.layers import Conv1D
import logging
import tensorflow as tf

class Model():
    def __init__(self, cfg, labels, input_signal, signal_len):
        self.logger = logging.getLogger(__name__)
        learning_phase = K.learning_phase()

        net = input_signal
        net = Conv1D(5, 3, input_shape=(cfg.batch_size, cfg.seq_length, 1))(net)

        self.logits = net # Tensor of shape [batch_size, max_time, class_num]
        self.logger.info(f"Logits shape: {self.logits.shape}")

        ratio = 1
        self.loss = tf.reduce_mean(tf.nn.ctc_loss(
            labels,
            self.logits,
            tf.cast(tf.floor_div(signal_len + ratio - 1, ratio), tf.int32),  # Round up
            ctc_merge_repeated=True,
            time_major=False,
        ))

        tf.add_to_collection('losses', self.loss)
        tf.summary.scalar('loss', self.loss)

        self.total_loss = tf.add_n(tf.get_collection('losses'),name = 'total_loss')
        self.train_step = tf.train.AdadeltaOptimizer().minimize(self.total_loss)
