from __future__ import absolute_import
from __future__ import print_function

import argparse
import h5py
import logging
import numpy as np
import os

from keras import backend as K
from keras.applications import vgg16
from keras.backend import tensorflow_backend as tf_bck
from keras.callbacks import Callback, History, ModelCheckpoint, EarlyStopping
from keras.layers import Input, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, load_img, img_to_array
from keras.utils import np_utils
from keras.utils.visualize_util import plot

from config import config
from constants import *
from vgg16_bn_model import vgg16_bn


def training():
    global args
    args = parse_args()
    with tf_bck.tf.device('/gpu:{}'.format(args.gpu)):
        tf_bck.set_session(tf_bck.tf.Session(config=tf_bck.tf.ConfigProto(allow_soft_placement=True)))
        x_train = np.load(config.TRAIN.IMAGE)
        y_train = np.load(config.TRAIN.LABEL)
        y_train = np_utils.to_categorical(y_train, config.NB_CLASSES)
        x_val = np.load(config.VAL.IMAGE)
        y_val = np.load(config.VAL.LABEL)
        y_val = np_utils.to_categorical(y_val, config.NB_CLASSES)
        model = vgg16_bn(weight_path=args.pretrained)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=config.LR), metrics=['accuracy'])
        
        layer_name = [ layer.name for layer in model.layers ]
        frozen = layer_name.index(args.frozen)
        for layer in model.layers[:frozen]:
            layer.trainable = False
       
        checkpoint = ModelCheckpoint(filepath='./checkpoints/'+args.checkpoint+'-{epoch:02d}-{loss:.4f}-{val_acc:.4f}.h5',
                                     monitor='val_acc', mode='max')
        history = TrainHistory()
    
        model.fit(x_train, y_train, batch_size=config.BATCH_SIZE, nb_epoch=config.EPOCH, \
                  callbacks=[ history, checkpoint ], validation_data=(x_val, y_val))

class TrainHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []
        self.nb_batch = 0
        self.avg_loss = 0.0
        self.avg_acc = 0.0
    
    def on_batch_begin(self, batch, logs={}):
        self.nb_batch += 1
        if (self.nb_batch > 100):
            self.log_loss = open('./logs/{}-loss.txt'.format(args.log), 'a')
            self.log_acc = open('./logs/{}-acc.txt'.format(args.log), 'a')
        
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accs.append(logs.get('acc'))
        if (self.nb_batch > 100):
            self.avg_loss += (self.losses[-1]-self.losses[-100])
            self.avg_acc += (self.accs[-1]-self.accs[-100])
            self.log_loss.write('{}\n'.format(self.avg_loss/100))
            self.log_loss.close()
            self.log_acc.write('{}\n'.format(self.avg_acc/100))
            self.log_acc.close()
        else:
            self.avg_loss += logs.get('loss')
            self.avg_acc += logs.get('acc')

def parse_args():
    parser = argparse.ArgumentParser(description='Train VGG-16 Network')
    parser.add_argument('--pretrained', dest='pretrained', help='pretrained model prefix', 
                        default=config.PRETRAINED, type=str)
    parser.add_argument('--gpu', dest='gpu', help='gpu device id',
                        default=0, type=str)
    parser.add_argument('--lr', dest='lr', help='learning rate',
                        default=config.LR, type=float)
    parser.add_argument('--log', dest='log', help='log file prefix',
                        default='', type=str)
    parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint file preifx',
                        default=config.CHECKPOINT, type=str)
    parser.add_argument('--resume', dest='resume', help='resume training',
                        default=False, type=bool)
    parser.add_argument('--frozen', dest='frozen', help='frozen layer set lr=0.', 
                        default='block3_conv1', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    training() 
