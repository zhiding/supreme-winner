from PIL import Image
from glob import glob
import numpy as np
import os

import keras.backend as K
from keras.preprocessing.image import load_img, img_to_array

from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score

from constants import *
from preprocess import make_content, preprocess_input
from models import vgg16_batchnorm

data_path = PARIS_DATA
gt_path = PARIS_GTRUTH
weights_path = './checkpoints/paris_finetune_weights_23_09_0.7733.h5'
label = PARIS_LABEL

model = vgg16_batchnorm()
model.load_weights(weights_path)

file_list = make_content('paris')
nb_samples = len(file_list)

def preprocess_query(cls_name, num):
    
    query_file = '{}/{}_{}_query.txt'.format(gt_path, cls_name, num)
    with open(query_file, 'r') as q:
        args = q.readline()
        args = args.split(' ')
        cls_name = args[0].split('_')[1]
        query_path = '{}/{}/{}.jpg'.format(data_path, cls_name, args[0])
        query_img = Image.open(query_path)
        region = (float(arg) for arg in args[1:])
        #query_img = query_img.crop(region)
        query_img = query_img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        query_img = img_to_array(query_img)
        query_img = np.expand_dims(query_img, axis=0)
        query_img = preprocess_input(query_img)
    return query_img

def preprocess(img_path):
    img = Image.open(img_path)
    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def preprocess_all(dataset, save_path=None):
    imgs = np.zeros((nb_samples, )+IMAGE_SHAPE, dtype='float32')
    for i in range(nb_samples):
        imgs[i] = preprocess(file_list[i])
    if save_path is not None:
        np.save(save_path, imgs)
    return 0

def forward(input_tensor):
    ops = K.function(
                [model.layers[0].input, K.learning_phase()],
                [model.get_layer('fc2').output])
    output_tensor = ops([input_tensor, 0])[0]
    return output_tensor

def forward_all(input_tensor, save_path=None, batch_sz=32, nb_channel=44):
    nb_samples = input_tensor.shape[0]
    nb_batch = nb_samples / batch_sz
    fc2_vec = np.zeros((nb_samples, nb_channel), dtype='float32')
    current = 0
    for i in range(nb_batch):
        current = i * batch_sz
        input_batch = input_tensor[current:current+batch_sz]
        fc2_batch = forward(input_batch)
        fc2_vec[current:current+batch_sz] = fc2_batch
    if nb_samples % batch_sz != 0:
        current = nb_batch * batch_sz
        fc2_vec[current:] = forward(input_tensor[current:])
    if save_path is not None:
        np.save(save_path, fc2_vec)
    return 0

def read_txt(txtfile):
    files = []
    with open(txtfile, 'r') as txt:
        for line in txt:
            files.append(line.split('\n')[0]+'.jpg')
    return files

def avg_precision(lab, num, forward_vec=None):
    gt_files = []
    for word in ['query', 'good', 'ok']:
        gt_files.append('{}/{}_{}_{}.txt'.format(gt_path, lab, num, word))
    true_images = []
    true_images += read_txt(gt_files[1])
    true_images += read_txt(gt_files[2])
    y_true = np.zeros(nb_samples, dtype='uint8')
    for i in range(nb_samples):
        img = file_list[i].split(os.sep)[-1]
        if img in true_images:
            y_true[i] = 1
    
    query_img = preprocess_query(lab, num)
    query_vec = forward(query_img)
    y_score = 1.0 - cdist(query_vec, forward_vec, 'cosine')[0]
    ap = average_precision_score(y_true, y_score)
    print(gt_files[0], ap)
    return ap

def mean_avg_precision(dataset):
    result = 0.
    forward_vec = np.load('./paris_forward_finetune.npy')
    for lab in label:
        if lab == 'general':
            continue
        for i in range(1,6):
            query_file = '{}/{}_{}_query.txt'.format(gt_path, lab, i)
            result += avg_precision(lab, i, forward_vec)
    result /= (len(label)*5)
    print(result)
    return result
    
if __name__ == '__main__':
    # avg_precision('defense', '1')
    # preprocess_all(dataset='paris', save_path='./paris_all_imgs.npy')
    # input_tensor = np.load('./paris_all_imgs.npy')
    # forward_all(input_tensor, save_path='./paris_predictions_finetune.npy')
    # print avg_precision('defense', 2, forward_vec)
    # preprocess_query('defense', 2)
    # avg_precision('triomphe', 2, forward_vec)
    mean_avg_precision('paris')
