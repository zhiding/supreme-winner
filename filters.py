from PIL import Image
import numpy as np
from operator import add

import keras.backend as K
from keras.preprocessing.image import load_img, img_to_array

from config import img_shape
from paris_6k_read import preprocess_input
from finetune_models import vgg16_finetune

data_path = './data/paris_6k/data'
weights_path = './paris6k_batchnorm_weights_21_0.7008.h5'

model = vgg16_finetune(include_top=True)

nb_filters = 512

# Note:
#   model weights should be loaded before calling
    
def forward(model, from_layer, to_layer, input_tensor):
    if from_layer == 'input':
        op_fwd = K.function(
                    [model.layers[0].input, K.learning_phase()],
                    [model.get_layer(to_layer).output])
    else:
        op_fwd = K.function(
                    [model.get_layer(from_layer).input, K.learning_phase()],
                    [model.get_layer(to_layer).output])
    fwd = op_fwd([input_tensor, 0])[0]
    
    return fwd

def critical_batch(model, input_tensor, cls, alpha=4.0):
    pred_before = forward(model, 'input', 'predictions', input_tensor)
    pool = forward(model, 'input', 'block5_pool', input_tensor)
    conv_filter = np.zeros(nb_filters)
    for i in range(nb_filters):
        pool[:,:,:,i] *= alpha
        pred_after = forward(model, 'flatten', 'predictions', pool)
        pool[:,:,:,i] /= alpha
        for k in range(input_tensor.shape[0]):
            if pred_after[k, cls] > pred_before[k, cls]:
                conv_filter[i] += 1
    return conv_filter

def critical(model, input_tensor, cls, batch_size=32, alpha=4.0):
    nb_batches = input_tensor.shape[0] / batch_size + 1
    conv_filter = np.zeros(nb_filters)
    for k in range(nb_batches):
        start = batch_size * k
        if (batch_size*(k+1) > input_tensor.shape[0]):
            end = input_tensor.shape[0]
        else:
            end = batch_size * (k+1)
        conv_filter_batch = critical_batch(model, input_tensor[start:end],
                        cls, alpha)
        conv_filter = map(add, conv_filter, conv_filter_batch)
        print('Epoch {}: completed'.format(k))
    return conv_filter

