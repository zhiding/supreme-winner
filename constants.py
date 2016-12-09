import os

WORK_PATH = os.getcwd()
workplace = os.getcwd()

CHECKPOINTS = os.path.join(WORK_PATH, 'checkpoints')

VGG_WEIGHTS_NOTOP = os.path.join(WORK_PATH, 
    'model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

IMAGE_WIDTH   = 224
IMAGE_HEIGHT  = 224
IMAGE_CHANNEL = 3
IMAGE_SHAPE = (
            IMAGE_WIDTH, 
            IMAGE_HEIGHT,
            IMAGE_CHANNEL)

DATA_PATH   = os.path.join(WORK_PATH, 'data')
data_folder = os.path.join(workplace, 'data')
PARIS_PATH  = os.path.join(DATA_PATH, 'paris_6k')
paris_folder  = os.path.join(data_folder, 'paris_6k') 
oxford_folder = os.path.join(data_folder, 'oxford_5k')

PARIS_CORRUPT = [ 
    'paris_louvre_000136.jpg',
    'paris_louvre_000146.jpg',
    'paris_moulinrouge_000422.jpg',
    'paris_museedorsay_001059.jpg',
    'paris_notredame_000188.jpg',
    'paris_pantheon_000284.jpg',
    'paris_pantheon_000960.jpg',
    'paris_pantheon_000974.jpg',
    'paris_pompidou_000195.jpg',
    'paris_pompidou_000196.jpg',
    'paris_pompidou_000201.jpg',
    'paris_pompidou_000467.jpg',
    'paris_pompidou_000640.jpg',
    'paris_sacrecoeur_000299.jpg',
    'paris_sacrecoeur_000330.jpg',
    'paris_sacrecoeur_000353.jpg',
    'paris_triomphe_000662.jpg',
    'paris_triomphe_000833.jpg',
    'paris_triomphe_000863.jpg',
    'paris_triomphe_000867.jpg']
             
PARIS_DATA    = os.path.join(PARIS_PATH, 'data')
PARIS_GTRUTH  = os.path.join(PARIS_PATH, 'groundtruth')
PARIS_TRAIN   = os.path.join(PARIS_PATH, 'train')
PARIS_VALID   = os.path.join(PARIS_PATH, 'valid')
PARIS_TRAIN_AUG = os.path.join(PARIS_PATH, 'train_augment')


PARIS_LABEL = [
            'defense',
            'eiffel',
            'invalides',
            'louvre',
            'moulinrouge',
            'museedorsay',
            'notredame',
            'pantheon',
            'pompidou',
            'sacrecoeur',
            'triomphe']

paris_labels = [
            'defense',
            'eiffel',
            'invalides',
            'louvre',
            'moulinrouge',
            'museedorsay',
            'notredame',
            'pantheon',
            'pompidou',
            'sacrecoeur',
            'triomphe']

OXFORD_LABEL = [
            'all_souls',
            'ashmolean',
            'balliol',
            'bodleian',
            'christ_church',
            'cornmarket',
            'hertford',
            'keble',
            'magdalen',
            'pitt_rivers',
            'radcliffe_camera']

oxford_labels = [
            'all_souls',
            'ashmolean',
            'balliol',
            'bodleian',
            'christ_church',
            'cornmarket',
            'hertford',
            'keble',
            'magdalen',
            'pitt_rivers',
            'radcliffe_camera']

SEED = 12761

summary = {
    'paris': {
        'labels': paris_labels,
        'groundtruth': '{}/groundtruth'.format(paris_folder),
        
        
        },
    'oxford': {
        'labels': oxford_labels,

        }
    }
