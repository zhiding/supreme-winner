from __future__ import print_function
import h5py

from constants import IMAGE_SHAPE

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, Flatten, Dense
from keras.layers.normalization import BatchNormalization


# default model is only add batch normalization layer after conv layer in conv 5
# bn_layer: ['all', 'last', 'last2nd', None]
def vgg16_batchnorm(input_tensor = None, 
                    nb_class     = 11,
                    include_top  = True,
                    bn_layer     = None,
                    init_weights = None):
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
    if bn_layer == 'all':
        x = BatchNormalization(name='block1_bn1')(x)
    x = Activation(activation='relu', name='block1_relu1')(x)
    x = Convolution2D(64, 3, 3, border_mode='same', name='block1_conv2')(x)
    if bn_layer == 'all':
        x = BatchNormalization(name='block1_bn2')(x)
    x = Activation(activation='relu', name='block1_relu2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # block 2
    x = Convolution2D(128, 3, 3, border_mode='same', name='block2_conv1')(x)
    if bn_layer == 'all':
        x = BatchNormalization(name='block2_bn1')(x)
    x = Activation(activation='relu', name='block2_relu1')(x)
    x = Convolution2D(128, 3, 3, border_mode='same', name='block2_conv2')(x)
    if bn_layer == 'all':
        x = BatchNormalization(name='block2_bn2')(x)
    x = Activation(activation='relu', name='block2_relu2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # block 3
    x = Convolution2D(256, 3, 3, border_mode='same', name='block3_conv1')(x)
    if bn_layer == 'all':
        x = BatchNormalization(name='block3_bn1')(x)
    x = Activation(activation='relu', name='block3_relu1')(x)
    x = Convolution2D(256, 3, 3, border_mode='same', name='block3_conv2')(x)
    if bn_layer == 'all':
        x = BatchNormalization(name='block3_bn2')(x)
    x = Activation(activation='relu', name='block3_relu2')(x)
    x = Convolution2D(256, 3, 3, border_mode='same', name='block3_conv3')(x)
    if bn_layer == 'all':
        x = BatchNormalization(name='block3_bn3')(x)
    x = Activation(activation='relu', name='block3_relu3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # block 4
    x = Convolution2D(512, 3, 3, border_mode='same', name='block4_conv1')(x)
    if bn_layer in ['all', 'last2nd']:
        x = BatchNormalization(name='block4_bn1')(x)
    x = Activation(activation='relu', name='block4_relu1')(x)
    x = Convolution2D(512, 3, 3, border_mode='same', name='block4_conv2')(x)
    if bn_layer in ['all', 'last2nd']:
        x = BatchNormalization(name='block4_bn2')(x)
    x = Activation(activation='relu', name='block4_relu2')(x)
    x = Convolution2D(512, 3, 3, border_mode='same', name='block4_conv3')(x)
    if bn_layer in ['all', 'last2nd']:
        x = BatchNormalization(name='block4_bn3')(x)
    x = Activation(activation='relu', name='block4_relu3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # block 5
    x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv1')(x)
    if bn_layer is not None:
        x = BatchNormalization(name='block5_bn1')(x)
    x = Activation(activation='relu', name='block5_relu1')(x)
    x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv2')(x)
    if bn_layer is not None:
        x = BatchNormalization(name='block5_bn2')(x)
    x = Activation(activation='relu', name='block5_relu2')(x)
    x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv3')(x)
    if bn_layer is not None:
        x = BatchNormalization(name='block5_bn3')(x)
    x = Activation(activation='relu', name='block5_relu3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # classification block
        x = Flatten(name='flatten')(x)
        x = Dense(nb_class*4, activation='relu', name='fc1')(x)
        x = Dense(nb_class*4, activation='relu', name='fc2')(x)
        x = Dense(nb_class, activation='softmax', name='predictions')(x)

    model = Model(img_input, x)

    if init_weights is not None:
        load_conv_weights(model, init_weights)

    return model

def load_conv_weights(model, weights_path):
    f = h5py.File(weights_path)
    for layer in model.layers:
        if layer.__class__.__name__ == 'Convolution2D':
            g = f[layer.name]
            weights = [g[layer.name + '_{}_1:0'.format(p)] for p in ['W', 'b']]
            layer.set_weights(weights)
    f.close()
    print('Conv layer weights loaded.')
    return 0
