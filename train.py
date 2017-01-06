from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator, array_to_img, load_img, img_to_array
from keras.applications import vgg16
from keras import backend as K
import keras.backend.tensorflow_backend as tf_bck
from keras.optimizers import Adadelta, Adam, RMSprop
from keras.callbacks import Callback, History, ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.visualize_util import plot

import os, h5py
import numpy as np

from constants import *
from models import load_conv_weights, vgg16_batchnorm

def generate_config(custom_cfg):
    base_cfg = {
        'dataset': 'paris',
        'train_img_path': PARIS_PATH + os.sep + 'paris_train_mean_img.npy',
        'train_lab_path': PARIS_PATH + os.sep + 'paris_train_mean_lab.npy',
        'valid_img_path': PARIS_PATH + os.sep + 'paris_val_mean_img.npy',
        'valid_lab_path': PARIS_PATH + os.sep + 'paris_val_mean_lab.npy',
        'batch_sz': 64,
        'nb_epoch': 50,
        'lr': 0.0001/2,
        'init_weights': VGG_WEIGHTS_NOTOP,
        'nb_class': 11
        }
    cfg = dict(list(base_cfg.items()) + list(custom_cfg.items()))
    if custom_cfg['vgg_version'] is None:
        cfg['method'] = 'finetune'
        cfg['frozen'] = 'block5_conv1'
    elif custom_cfg['vgg_version'] == 'all':
        cfg['method'] = 'all_bn'
        cfg['frozen'] = None
    elif custom_cfg['vgg_version'] == 'last':
        cfg['method'] = 'last_bn'
        cfg['frozen'] = 'block5_conv1'
    elif custom_cfg['vgg_version'] == 'last2nd':
        cfg['method'] = 'part_bn'
        cfg['frozen'] = 'block4_conv1'
    return cfg

def train(cfg):
    with tf_bck.tf.device(cfg['gpu']):
        tf_bck.set_session(tf_bck.tf.Session(
                config=tf_bck.tf.ConfigProto(
                        allow_soft_placement=True)))
        prepare_train(cfg)

def load_model(cfg):
    model = vgg16_batchnorm(nb_class=cfg['nb_class'],
                            bn_layer=cfg['vgg_version'])
    if cfg['load_weights'] is not None:
        print(cfg['load_weights'])
        load_conv_weights(model, cfg['load_weights'])
    return model

def prepare_train(cfg):
    # load data from npys
    x_train = np.load(cfg['train_img_path'])
    y_train = np.load(cfg['train_lab_path'])
    y_train = np_utils.to_categorical(y_train, cfg['nb_class'])
    x_val = np.load(cfg['valid_img_path'])
    y_val = np.load(cfg['valid_lab_path'])
    y_val = np_utils.to_categorical(y_val, cfg['nb_class'])

    model = load_model(cfg)
    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(lr=cfg['lr']), 
                  metrics=['accuracy'])
    
    # set frozen layers
    layers_name = [layer.name for layer in model.layers]
    if cfg['frozen'] is not None:
        frozen = layers_name.index(cfg['frozen'])
        for layer in model.layers[:frozen]:
            layer.trainable = False

    # set checkpoint
    filepath = cfg['dataset'] + '_' + cfg['method'] \
             + '_{epoch:02d}_{val_acc:.4f}.h5' 
    checkpoint = ModelCheckpoint(filepath=filepath, 
                                 monitor='val_acc', 
                                 save_best_only=True, 
                                 mode='max')
    history = TrainHistory()

    # begin training
    model.fit(x_train, y_train, 
              batch_size=cfg['batch_sz'], 
              nb_epoch=cfg['nb_epoch'], 
              callbacks=[history, checkpoint],
              validation_data=(x_val, y_val))

class TrainHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []
    
    def on_batch_begin(self, batch, logs={}):
        self.log_loss = open('./logs/{}_{}_loss_per_batch.txt'.format(
                                cfg['method'], cfg['lr']), 'a')
        self.log_acc = open('./logs/{}_{}_acc_per_batch.txt'.format(
                                cfg['method'], cfg['lr']), 'a')
        
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.log_loss.write('{}\n'.format(logs.get('loss')))
        self.log_loss.close()
        self.accs.append(logs.get('acc'))
        self.log_acc.write('{}\n'.format(logs.get('acc')))
        self.log_acc.close()
        

if __name__ == '__main__':
    ft_cfg = {
            'gpu': '/gpu:3',
            'vgg_version': None,
            #'load_weights': 'model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5' 
            'load_weights': 'paris_finetune_13_0.7203.h5'
            }
    
    all_bn_cfg = {
            'gpu': '/gpu:3',
            'vgg_version': 'all',
            'load_weights': 'paris_all_bn_14_21_0.7972.h5'
            }
    
    last2nd_bn_cfg = {
            'gpu': '/gpu:2',
            'vgg_version': 'last2nd',
            'load_weights': 'paris_part_bn_24_0.8156.h5'
            }
    
    last_bn_cfg = {
            'gpu': '/gpu:3',
            'vgg_version': 'last',
            'load_weights': 'paris_last_bn_07_0.7708.h5'
            }
    cfg = generate_config(ft_cfg)
    train(cfg) 
