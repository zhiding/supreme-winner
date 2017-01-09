from __future__ import print_function
import logging
import h5py

from constants import IMAGE_SHAPE
from layers import MapwiseConnected

from keras import backend as K
from keras.layers import Input, Activation, Flatten, Dense, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

def vgg16_bn(input_tensor = None, nb_classes   = 11,
            include_top  = True, weight_path  = None):
    # set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    input_shape = IMAGE_SHAPE

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # block 1
    x = Convolution2D(64, 3, 3, border_mode='same', name='block1_conv1')(img_input)
    x = Activation(activation='relu', name='block1_relu1')(x)
    x = Convolution2D(64, 3, 3, border_mode='same', name='block1_conv2')(x)
    x = Activation(activation='relu', name='block1_relu2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # block 2
    x = Convolution2D(128, 3, 3, border_mode='same', name='block2_conv1')(x)
    x = Activation(activation='relu', name='block2_relu1')(x)
    x = Convolution2D(128, 3, 3, border_mode='same', name='block2_conv2')(x)
    x = Activation(activation='relu', name='block2_relu2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # block 3
    x = Convolution2D(256, 3, 3, border_mode='same', name='block3_conv1')(x)
    x = Activation(activation='relu', name='block3_relu1')(x)
    x = Convolution2D(256, 3, 3, border_mode='same', name='block3_conv2')(x)
    x = Activation(activation='relu', name='block3_relu2')(x)
    x = Convolution2D(256, 3, 3, border_mode='same', name='block3_conv3')(x)
    x = Activation(activation='relu', name='block3_relu3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # block 4
    x = Convolution2D(512, 3, 3, border_mode='same', name='block4_conv1')(x)
    x = BatchNormalization(name='block4_bn1')(x)
    x = Activation(activation='relu', name='block4_relu1')(x)
    x = Convolution2D(512, 3, 3, border_mode='same', name='block4_conv2')(x)
    x = BatchNormalization(name='block4_bn2')(x)
    x = Activation(activation='relu', name='block4_relu2')(x)
    x = Convolution2D(512, 3, 3, border_mode='same', name='block4_conv3')(x)
    x = BatchNormalization(name='block4_bn3')(x)
    x = Activation(activation='relu', name='block4_relu3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # block 5
    x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv1')(x)
    x = BatchNormalization(name='block5_bn1')(x)
    x = Activation(activation='relu', name='block5_relu1')(x)
    x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv2')(x)
    x = BatchNormalization(name='block5_bn2')(x)
    x = Activation(activation='relu', name='block5_relu2')(x)
    x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv3')(x)
    x = BatchNormalization(name='block5_bn3')(x)
    x = Activation(activation='relu', name='block5_relu3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dropout(0.5, name='drop1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dropout(0.5, name='drop2')(x)
        x = Dense(nb_classes, activation='softmax', name='predictions')(x)

    model = Model(img_input, x)

    if weight_path:
        logging.info('######## LOAD CONV LAYER WEIGHT FROM EXISTED MODEL')
        try:
            model.load_weights(weight_path, by_name=True)
        except:
            load_model_weight(model, weight_path)
    return model

def load_model_weight(model, weight_path):
    f = h5py.File(weight_path)
    g = f
    for layer in model.layers:
        layer_name = layer.name
        layer_classname = layer.__class__.__name__
        if 'model_weights' in f.keys():
            g = f['model_weights']
        if layer_classname == 'Convolution2D':
            h = g[layer_name]
            for k in h.keys():
                if layer_name + '_W' in k:
                    W_weight = k
                if layer_name + '_b' in k:
                    b_weight = k
            weights = [ h[W_weight], h[b_weight] ]
            layer.set_weights(weights)
        elif layer_classname == 'BatchNormalization':
            if layer_name not in g.keys():
                continue;
            h = g[layer_name]
            for k in h.keys():
                if layer_name + '_gamma' in k:
                    gamma_weight = k
                if layer_name + '_beta' in k:
                    beta_weight = k
                if layer_name + '_running_mean' in k:
                    mean_weight = k
                if layer_name + '_running_std' in k:
                    std_weight = k
            weights = [ h[gamma_weight], h[beta_weight], h[mean_weight], h[std_weight] ]
            layer.set_weights(weights)
    f.close()

def load_weights_except(model, weights_path):
    f = h5py.File(weights_path)
    for layer in model.layers[1:]:
        if layer.__class__.__name__ in ['Activation', 'MaxPooling2D', 'MapwiseConnected', 'Flatten']:
            continue
        if layer.__class__.__name__ == 'BatchNormalization':
            g = f['model_weights'][layer.name]
            weights = [g['{}_{}:0'.format(layer.name, p)] for p in ['gamma', 'beta', 'running_mean', 'running_std']]
            layer.set_weights(weights)
            continue
        g = f['model_weights'][layer.name]
        weights = [g['{}_{}:0'.format(layer.name, p)] for p in ['W', 'b']]
        layer.set_weights(weights)
    f.close()
    print('Layers weights loaded.')
    return 0 

def load_weights_include(model, weights_path):
    f = h5py.File(weights_path)
    for layer in model.layers[1:]:
        style = layer.__class__.__name__
        if style in ['Activation', 'MaxPooling2D', 'Flatten']:
            continue
        else:
            mw = f['model_weights'][layer.name]
            if style == 'MapwiseConnected':
                w = [ g['{}_W:0'.format(layer.name)] ]
            else:
                w = [ g['{}_{}:0'.format(layer.name, p)] for p in ['W', 'b'] ]
            layer.set_weights(weights)
    f.close()
    print('`load_weights_include` completed.')
    return 0
