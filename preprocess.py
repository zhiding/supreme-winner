
import os
from glob import glob
import numpy as np
import random
from shutil import copyfile, rmtree

from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator

from constants import *

# return from `create` func.
nb_train     = 2932 
nb_val       = 1963
# ls -lR | grep ^- | wc -l
nb_train_aug = 8782

def augment(src, dest, mul=3):
    if not os.path.exists(dest):
        os.mkdir(dest)
    dirs = glob('{}/*'.format(src))
    for d in dirs:
        classname = d.split(os.sep)[-1]
        print(classname)
        augment_by_class(src, dest, classname, mul)
    return 0

def augment_by_class(src, dest, classname, mul=3):
    files = glob('{}/{}/*.jpg'.format(src, classname))
    src_dir = os.path.join(src, classname)
    save_dir = os.path.join(dest, classname)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    temp_dir = os.path.join(src_dir, 'temp')
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    for f in files:
        fname = f.split(os.sep)[-1]
        copyfile(f, '{}/{}'.format(temp_dir, fname))

    batch_sz = 2
    nb_batch = len(files) * mul / batch_sz
    data_generator = ImageDataGenerator(
                rotation_range     = 20,
                width_shift_range  = 0.1,
                height_shift_range = 0.1)
    i = 0
    for batch in data_generator.flow_from_directory(
                directory = src_dir,
                target_size = (IMAGE_WIDTH, IMAGE_HEIGHT),
                batch_size = batch_sz,
                seed = SEED,
                save_to_dir = save_dir,
                save_prefix = classname,
                save_format = 'jpg'):
        i += 1
        if (i >= nb_batch):
            break;
    rmtree(temp_dir)
    return 0

def create(dataset=None, prop=0.6, quiet=True):
    if dataset == 'paris':
        path  = PARIS_PATH
        data  = PARIS_DATA
        label = PARIS_LABEL
        corrupt = True
    elif dataset == 'oxford':
        path = OXFORD_PATH
        data = OXFORD_DATA
        label = OXFORD_LABEL
        corrupt = False
    else:
        raise Exception('Choose dataset between {`paris`, `oxford`}.')

    # Remove existed train/valid folders and create 
    train = os.path.join(path, 'train')
    valid = os.path.join(path, 'valid')
    if os.path.exists(train):
        rmtree(train)
    if os.path.exists(valid):
        rmtree(valid)
    os.mkdir(train)
    os.mkdir(valid)

    dirs = glob('{}/*'.format(data))
    nb_train = 0
    nb_valid = 0
    for d in dirs:
        dname = d.split(os.sep)[-1]
        if dname not in label:
            continue
        files = glob('{}/*.jpg'.format(d))
        random.shuffle(files)
        bound = int(len(files) * prop)
        for i in range(len(files)):
            fname = files[i].split(os.sep)[-1]
            if corrupt and (fname in PARIS_CORRUPT):
                continue
            src = os.path.join(data, dname)
            if (i < bound):
                dest = os.path.join(train, dname)
                nb_train += 1
            else:
                dest = os.path.join(valid, dname)
                nb_valid += 1
            move_file(fname, src, dest, quiet)
    print('{} samples in `train` and {} samples in `valid`'.format(nb_train, nb_valid))
    return 0 

def make_content(dataset):
    if dataset == 'paris':
        data_path = PARIS_DATA
    elif dataset == 'oxford':
        data_path = OXFORD_DATA
    file_list = []
    dirs = glob('{}/*'.format(data_path))
    for d in dirs:
        dname = d.split(os.sep)[-1]
        files = glob('{}/*.jpg'.format(d))
        for corrupt in PARIS_CORRUPT:
            filename = '{}/{}'.format(d, corrupt)
            if filename in files:
                files.remove(filename)
        file_list = file_list + files
    return file_list

def move_file(filename, src, dest, quiet=True):
    if not os.path.exists(dest):
        os.mkdir(dest)
    copyfile('{}/{}'.format(src, filename),
             '{}/{}'.format(dest, filename))
    if not quiet:
        print('Move file `{}` from `{}` to `{}`...'.format(filename, src, dest))
    return 0

def preprocess_input(x):
    # RGB ==> BGR
    x = x[:, :, :, ::-1]
    # substract mean 
    x[:, :, :, 0] -= 103.939
    x[:, :, :, 1] -= 116.779
    x[:, :, :, 2] -= 123.68
    return x

def read(path, nb_images, save=False, 
         save_prefix=None, mean=False):
    if 'paris' in path:
        data  = PARIS_DATA
        label = PARIS_LABEL
    elif 'oxford' in path:
        data  = OXFORD_DATA
        label = OXFORD_LABEL
    dirs = glob('{}/*'.format(path))
    x = np.zeros(((nb_images,) + IMAGE_SHAPE), dtype=np.float32) 
    y = np.zeros((nb_images, 1), dtype=np.uint8)
    pos = 0
    for d in dirs:
        dname = d.split(os.sep)[-1]
        files = glob('{}/*.jpg'.format(d))
        for f in files:
            img = load_img(f, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
            img = img_to_array(img)
            x[pos] = img
            y[pos] = label.index(dname) 
            pos += 1
    if mean:
        x = preprocess_input(x)
    if save:
        if save_prefix is None:
            raise Exception('`save_prefix` should be given '
                            'when setting `save=True`')
        else:
            np.save('{}/{}_img.npy'.format(data, save_prefix), x)
            np.save('{}/{}_lab.npy'.format(data, save_prefix), y)
    return 0

if __name__ == '__main__':
    # create('paris')
    # augment('./data/paris_6k/train', './data/paris_6k/train_augment')
    # read(PARIS_TRAIN_AUG, 8782, save=True,
    #            save_prefix='paris_train_mean', mean=True)
    # read(PARIS_VALID, 1963, save=True,
    #            save_prefix='paris_val_mean', mean=True)
    print make_content('paris')
