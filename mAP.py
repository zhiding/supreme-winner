from PIL import Image
from glob import glob
import numpy as np
import os

import keras.backend as K
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array

from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score

from constants import summary
from constants import IMAGE_SHAPE, IMAGE_WIDTH, IMAGE_HEIGHT
from preprocess import make_content, preprocess_input
from models import vgg16_batchnorm

dataset = 'paris'
env = summary[dataset]

def mean_average_precision(dataset, model, layer_name, fwd_path=None, pre_path=None, crop=False):
    mean_ap = 0.
    layer = model.get_layer(layer_name)
    nb_channel = layer.output_shape[-1]
    env = summary[dataset]
    
    if pre_path is not None:
        images = np.load(pre_path)
    else:
        images = preprocess_all(dataset, 
                    save_path='./paris_pre_img.npy') 
    if fwd_path is not None:
        feature_vec = np.load(fwd_path)
    else:
        feature_vec = forward_all(nb_channel, images, 
                    save_path='./paris_feat_vec.npy') 
    content = make_content(dataset)
    for label in env['labels']:
        for num in range(1, 6):
            query_file = '{}/{}_{}_query.txt'.format(
                        env['groundtruth'], label, num)
            query_img = preprocess_query(dataset, query_file, crop)
            query_vec = forward_query(query_img)
            mean_ap += average_precision(dataset, content, label, num, query_vec, feature_vec) ## TODO CHECK IT
    mean_ap /= len(env['labels']) * 5
    return mean_ap

def average_precision(dataset, content, label, num, query_vec, feature_vec):
    ap = 0.
    query = '{}/{}_{}_query.txt'.format(env['groundtruth'], label, num)
    good  = '{}/{}_{}_good.txt'.format(env['groundtruth'], label, num)
    ok    = '{}/{}_{}_ok.txt'.format(env['groundtruth'], label, num)
    match_files = read_txt([good, ok])
    nb_samples = feature_vec.shape[0]
    y_true = np.zeros(nb_samples, dtype='uint8')
    for i in range(nb_samples):
        f = content[i].split(os.sep)[-1]
        if f in match_files:
            y_true[i] = 1
    y_score = 1.0 - cdist(query_vec, feature_vec, 'cosine')[0]
    ap = average_precision_score(y_true, y_score)
    print(query.split(os.sep)[-1], ap)
    return ap

def preprocess_query(dataset, query_file, crop=False):
    data_path = summary[dataset]['data']
    with open(query_file, 'r') as q:
        args = q.readline()
        args = args.split(' ')
        lab_dir = args[0].split('_')[1]
        query_path = '{}/{}/{}.jpg'.format(data_path, lab_dir, args[0])
        region = (float(arg) for arg in args[1:])
        query_img = preprocess(query_path, crop, region)
    return query_img

def forward_query(query_img):
    return forward(query_img)

def preprocess_all(dataset, save_path=None):
    content = make_content(dataset)
    nb_samples = len(content)
    images = np.zeros((nb_samples, )+IMAGE_SHAPE, dtype='float32')
    for i in range(nb_samples):
        images[i] = preprocess(content[i])
    if save_path is not None:
        np.save(save_path, images)
    return images

def forward_all(nb_channel, input_tensor, save_path=None, batch_sz=32):
    nb_samples = input_tensor.shape[0]
    nb_batch = nb_samples / batch_sz
    feature_vec = np.zeros((nb_samples, nb_channel), dtype='float32')
    current = 0
    for i in range(nb_batch):
        current = i * batch_sz
        input_batch = input_tensor[current:current+batch_sz]
        featvec_batch = forward(input_batch)
        feature_vec[current:current+batch_sz] = featvec_batch
    if nb_samples % batch_sz != 0:
        current = nb_batch * batch_sz
        feature_vec[current:] = forward(input_tensor[current:])
    if save_path is not None:
        np.save(save_path, feature_vec)
    return feature_vec

def forward(input_tensor):
    ops = K.function(
                [model.layers[0].input, K.learning_phase()],
                [model.get_layer('fc2').output])
    output_tensor = ops([input_tensor, 0])[0]
    return output_tensor

def preprocess(img_path, crop=False, region=None):
    img = Image.open(img_path)
    if crop:
        img = img.crop(region)
    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def read_txt(txtfiles):
    files = []
    for txtfile in txtfiles:
        with open(txtfile, 'r') as txt:
            for f in txt:
                files.append(f.split('\n')[0] + '.jpg')
    return files

    
if __name__ == '__main__':

    model = vgg16_batchnorm(nb_class=11, bn_layer=None)
    model.load_weights(
            './checkpoints/paris_finetune_weights_23_09_0.7733.h5')
    # mean = mean_average_precision('paris', model, 'fc2', \
    #                 'paris_forward_finetune.npy', \
    #                 'paris_all_imgs.npy', \
    #                 crop=True)
    mean = mean_average_precision('paris', model, \
                'fc2', pre_path='paris_pre_img.npy')
    print mean
