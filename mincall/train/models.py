from keras import models, layers, regularizers, backend as K
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
        )
        super(ConstMultiplierLayer, self).build(input_shape)

    def call(self, x):
        return K.tf.multiply(self.k, x)

    def compute_output_shape(self, input_shape):
        return input_shape

custom_layers = {
    ConstMultiplierLayer.__name__: ConstMultiplierLayer
}

def dummy_model(hparams:str=None):
    input = layers.Input(shape=(None, 1))
    net = input
    for _ in range(5):
        net = layers.BatchNormalization()(net)
        net = layers.Conv1D(
            10,
            3,
            padding="same",
            dilation_rate=2,
            bias_regularizer=regularizers.l1(0.1))(net)
        net = layers.Activation('relu')(net)

    net = layers.Conv1D(5, 3, padding="same")(net)
    return models.Model(inputs=[input], outputs=[net])


def big_01(hparams: str):
    input = layers.Input(shape=(None, 1))
    net = input
    # net = layers.Conv1D(
    #     256,
    #     3,
    #     padding="same",
    #     bias_regularizer=regularizers.l1(0.1))(net)
    # for _ in range(5):
    #     x = net
    #     net = layers.BatchNormalization()(net)
    #     net = layers.Conv1D(
    #         256,
    #         3,
    #         padding="same",
    #         bias_regularizer=regularizers.l1(0.1))(net)
    #     net = layers.Activation('relu')(net)

    x = net
    net = layers.add([
        x,
        ConstMultiplierLayer()(net),
    ])

    net = layers.Conv1D(5, 3, padding="same")(net)
    return models.Model(inputs=[input], outputs=[net])


all_models: Dict[str, Callable[[str], models.Model]] = {
    'dummy': dummy_model,
    'big_01': big_01,
}