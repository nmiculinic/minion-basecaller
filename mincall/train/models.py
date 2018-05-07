from keras import models, layers
import logging
import tensorflow as tf


def dummy_model():
    input = layers.Input(shape=(None, 1))
    net = input
    for _ in range(5):
        net = layers.BatchNormalization()(net)
        net = layers.Conv1D(10, 3, padding="same", dilation_rate=2)(net)
        net = layers.Activation('relu')(net)

    net = layers.Conv1D(5, 3, padding="same")(net)

    m = models.Model(inputs=[input], outputs=[net])
    ratio = 1
    m.build(input_shape=(None, 1))
    return m, ratio
