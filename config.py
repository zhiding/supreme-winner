import numpy as np
from easydict import EasyDict as edict

config = edict()

# TRAINING CONFIG
config.BATCH_SIZE = 64
config.EPOCH = 50
config.GPU_ID = 0
config.LR = 1e-4
config.SGD = 'Adam'
config.CHECKPOINT = './paris-finetune'

config.TRAIN = edict()
config.VAL = edict()

# DATASET CONFIG
config.NB_CLASSES = 11
config.TRAIN.IMAGE = '/home/zhiding/iconip/data/paris_6k/paris_train_mean_img.npy'
config.TRAIN.LABEL = '/home/zhiding/iconip/data/paris_6k/paris_train_mean_lab.npy'
config.VAL.IMAGE = '/home/zhiding/iconip/data/paris_6k/paris_val_mean_img.npy'
config.VAL.LABEL = '/home/zhiding/iconip/data/paris_6k/paris_val_mean_lab.npy'

config.PRETRAINED = './model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
