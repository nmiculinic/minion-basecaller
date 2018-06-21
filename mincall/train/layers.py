from keras import models, layers, regularizers, constraints, backend as K
from keras.engine.topology import Layer

__all__ = ["ConstMultiplierLayer", "custom_layers"]


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
