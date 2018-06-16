from keras import models, layers, regularizers, constraints, backend as K
from keras.engine.topology import Layer
from typing import *
import logging
import tensorflow as tf

import numpy as np


class ConstMultiplierLayer(Layer):
    def __init__(self, **kwargs):
        super(ConstMultiplierLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.k = self.add_weight(
            name='k',
            shape=(),
            initializer='ones',
            dtype='float32',
            trainable=True,
            constraint=constraints.MinMaxNorm(
                min_value=0.0, max_value=1.0, axis=[]
            ),
        )
        super(ConstMultiplierLayer, self).build(input_shape)

    def call(self, x):
        return K.tf.multiply(self.k, x)

    def compute_output_shape(self, input_shape):
        return input_shape


custom_layers = {ConstMultiplierLayer.__name__: ConstMultiplierLayer}


def dummy_model(n_classes: int, hparams: str = None):
    input = layers.Input(shape=(None, 1))
    net = input
    for _ in range(5):
        net = layers.BatchNormalization()(net)
        net = layers.Conv1D(
            10,
            3,
            padding="same",
            dilation_rate=2,
            bias_regularizer=regularizers.l1(0.1)
        )(net)
        net = layers.Activation('relu')(net)

    net = layers.Conv1D(n_classes, 3, padding="same")(net)
    return models.Model(inputs=[input], outputs=[net]), 1


def big_01(n_classes: int, hparams: str):
    input = layers.Input(shape=(None, 1))
    net = layers.BatchNormalization()(input)
    net = layers.Conv1D(
        256, 3, padding="same", bias_regularizer=regularizers.l1(0.1)
    )(net)

    for _ in range(2):
        x = net
        net = layers.Conv1D(256, 5, padding='same')(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = layers.Conv1D(256, 5, padding='same')(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = ConstMultiplierLayer()(net)
        net = layers.add([x, net])
        net = layers.MaxPool1D(padding='same', pool_size=2)(net)

    net = layers.Conv1D(n_classes, 3, padding="same")(net)
    net = layers.BatchNormalization()(net)
    return models.Model(inputs=[input], outputs=[net]), 2 * 2


def m270(n_classes: int, hparams: str):
    input = layers.Input(shape=(None, 1))
    net = input
    block_channels = [32, 64]

    for block_channel in block_channels:
        net = layers.Conv1D(
            block_channel,
            3,
            padding="same",
            bias_regularizer=regularizers.l1(0.1)
        )(net)
        for _ in range(20):
            x = net
            net = layers.Conv1D(block_channel, 3, padding='same')(net)
            net = layers.BatchNormalization()(net)
            net = layers.Activation('relu')(net)
            net = layers.Conv1D(block_channel, 3, padding='same')(net)
            net = layers.BatchNormalization()(net)
            net = layers.Activation('relu')(net)
            net = ConstMultiplierLayer()(net)
            net = layers.add([x, net])
        net = layers.MaxPool1D(padding='same', pool_size=2)(net)

    net = layers.Conv1D(n_classes, 3, padding="same")(net)
    net = layers.BatchNormalization()(net)
    return models.Model(inputs=[input], outputs=[net]), 2**len(block_channels)


def chiron_like(n_classes: int, hparams: str):
    input = layers.Input(shape=(None, 1))
    net = input  # (batch_size, sequence_len, channels)
    out_chan = 256
    for i in range(3):
        with tf.name_scope(f"block_{i}"):
            net = layers.BatchNormalization()(net)
            with tf.variable_scope('branch1'):
                b1 = layers.Conv1D(
                    filters=out_chan,
                    kernel_size=1,
                    activation='linear',
                    padding='same',
                    use_bias=False
                )(net)
            with tf.variable_scope('branch2'):
                b2 = net
                b2 = layers.Conv1D(
                    filters=out_chan,
                    kernel_size=1,
                    activation='relu',
                    padding='same',
                    use_bias=False
                )(b2)
                b2 = layers.Conv1D(
                    filters=out_chan,
                    kernel_size=3,
                    activation='relu',
                    padding='same',
                    use_bias=False
                )(b2)
                b2 = layers.Conv1D(
                    filters=out_chan,
                    kernel_size=1,
                    activation='linear',
                    padding='same',
                    use_bias=False
                )(b2)
            with tf.variable_scope('plus'):
                net = layers.Add()([b1, b2])
                net = layers.Activation('relu')(net)

    hidden_num = [100, 100, n_classes]
    # RNN layers:
    if tf.test.is_built_with_cuda():
        logging.getLogger(__name__).info(f"Using CuDNNLSTM optimized cell")
        cell = layers.CuDNNLSTM
    else:
        cell = layers.LSTM

    for h in hidden_num:
        net = cell(h, return_sequences=True)(net)
    return models.Model(inputs=[input], outputs=[net]), 1


all_models: Dict[str, Callable[[str], models.Model]] = {
    'dummy': dummy_model,
    'big_01': big_01,
    'm270': m270,
    'chiron': chiron_like,
}
