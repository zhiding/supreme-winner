from __future__ import absolute_import
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.utils.generic_utils import get_from_module

def mse(y_true, y_pred):
    return K.mean(K.square(y_pred[:,9]-y_true[:,9]), axis=-1)

def cc_mse(y_true, y_pred):
    wmse = tf.mul(mse(y_true, y_pred), 0.01)
    cc   = K.categorical_crossentropy(y_pred, y_true)
    return tf.add(cc, wmse)
