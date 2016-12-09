from keras import backend as K
from keras import regularizers, constraints
from keras.engine.topology import Layer
import tensorflow as tf
import numpy as np

class MapwiseConnected(Layer):
    def __init__(self, weights=None, 
                    dim_ordering='default', 
                    W_regularizer='l1', 
                    W_constraint='nonneg', **kwargs):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        self.W_regularizer = regularizers.get(W_regularizer) 
        self.W_constraint = constraints.get(W_constraint)
        self.initial_weights = weights
        super(MapwiseConnected, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 4
        if self.dim_ordering == 'th':
            self.W_shape = (1, input_shape[1])
        elif self.dim_ordering == 'tf':
            self.W_shape = (1, input_shape[3])
        self.W = K.variable(np.ones(self.W_shape), name='{}_W'.format(self.name))
        self.trainable_weights = [self.W]

        self.W_regularizer.set_param(self.W)
        self.regularizers.append(self.W_regularizer)

        self.constraints = {}
        self.constraints[self.W] = self.W_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True
    
    def get_output_shape_for(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        return tf.mul(x, self.W)

    def get_config(self):
        config = {
                    'dim_ordering': self.dim_ordering,
                    'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                    'W_constraint': self.W_constraint.get_config() if self.W_constraint else None}
        base_config = super(MapwiseConnected, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
