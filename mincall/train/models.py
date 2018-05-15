from keras import models, layers, regularizers, backend as K
from typing import *
import logging
import tensorflow as tf


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
        # net * K.variable(value=1.0, dtype='float32'),
        layers.multiply([K.variable(value=1.0, dtype='float32'), net]),
    ])

    net = layers.Conv1D(5, 3, padding="same")(net)
    return models.Model(inputs=[input], outputs=[net])


all_models: Dict[str, Callable[[str], models.Model]] = {
    'dummy': dummy_model,
    'big_01': big_01,
}