from keras import models, layers, regularizers, constraints, backend as K
from keras.engine.topology import Layer
from typing import *
from mincall.common import named_tuple_helper
import logging
import tensorflow as tf
import voluptuous

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


class DummyCfg(NamedTuple):
    num_layers: int

    @classmethod
    def scheme(cls, data):
        return named_tuple_helper(cls, {}, data)


def dummy_model(n_classes: int, hparams: Dict):
    cfg: DummyCfg = DummyCfg.scheme(hparams)
    input = layers.Input(shape=(None, 1))
    net = input
    for _ in range(cfg.num_layers):
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


class Big01Cfg(NamedTuple):
    num_blocks: int
    block_elem: int
    block_init_channels: int = 32
    receptive_width: int = 5

    @classmethod
    def scheme(cls, data):
        return named_tuple_helper(cls, {}, data)


def big_01(n_classes: int, hparams: Dict):
    cfg: Big01Cfg = Big01Cfg.scheme(hparams)
    input = layers.Input(shape=(None, 1))
    net = input
    for i in range(cfg.num_blocks):
        channels = 2**i * cfg.block_init_channels
        net = layers.Conv1D(
            channels,
            cfg.receptive_width,
            padding="same",
            bias_regularizer=regularizers.l1(0.1)
        )(net)
        with tf.name_scope(f"block_{i}"):
            for _ in range(cfg.block_elem):
                x = net
                net = layers.Conv1D(
                    channels, cfg.receptive_width, padding='same'
                )(net)
                net = layers.BatchNormalization()(net)
                net = layers.Activation('relu')(net)
                net = layers.Conv1D(
                    channels, cfg.receptive_width, padding='same'
                )(net)
                net = layers.BatchNormalization()(net)
                net = layers.Activation('relu')(net)
                net = ConstMultiplierLayer()(net)
                net = layers.add([x, net])
        net = layers.MaxPool1D(padding='same', pool_size=2)(net)

    net = layers.Conv1D(n_classes, cfg.receptive_width, padding="same")(net)
    return models.Model(inputs=[input], outputs=[net]), 2**cfg.num_blocks


all_models: Dict[str, Callable[[str], models.Model]] = {
    'dummy': dummy_model,
    'big_01': big_01,
}

hparam_cfg: Dict[str, NamedTuple] = {
    'big_01': Big01Cfg,
    'dummy': DummyCfg,
}
