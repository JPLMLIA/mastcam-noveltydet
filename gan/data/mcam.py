import logging
import numpy as np
import os.path
import random
from glob import glob

RANDOM_SEED = 42
RNG = np.random.RandomState(42)
random.seed(RANDOM_SEED)

logger = logging.getLogger(__name__)

def get_train(label):
    """Get training dataset for Mastcam"""
    trainx, names = _load_6f_images(path='/home/jovyan/data/train_typical')
    # Augment data with horizontal flips
    train_data_aug = np.ndarray([trainx.shape[0]*2, trainx.shape[1], trainx.shape[2], trainx.shape[3]])
    for i in range(trainx.shape[0]):
        hflip = np.fliplr(trainx[i])
        train_data_aug[trainx.shape[0]+i] = hflip
        train_names.append(names[i])
    trainx = train_data_aug
    # Rescale data to [-1, 1]
    trainx = _rescale_images(trainx)
    trainy = np.zeros(trainx.shape[0])
    return trainx, trainy

def get_test(label):
    """Get testing dataset for Mastcam"""
    base_novel_url = '/home/jovyan/data/test_novel'
    if label == 'all':
        path = os.path.join(base_novel_url, 'all')
    elif label == 'bedrock':
        path = os.path.join(base_novel_url, 'bedrock')
    elif label == 'drt':
        path = os.path.join(base_novel_url, 'drt')
    elif label == 'meteorite':
        path = os.path.join(base_novel_url, 'meteorite')
    elif label == 'veins':
        path = os.path.join(base_novel_url, 'veins')
    elif label == 'dump-pile':
        path = os.path.join(base_novel_url, 'dump-pile')
    elif label == 'drill-hole':
        path = os.path.join(base_novel_url, 'drill-hole')
    elif label == 'float':
        path = os.path.join(base_novel_url, 'float')
    elif label == 'broken-rock':
        path = os.path.join(base_novel_url, 'broken-rock')
    elif label == 'none': # during training
        path = ''

    testx_typ, names_typ = _load_6f_images(path='/home/jovyan/data/test_typical', shuffle=False)
    testx_nov, names_nov = _load_6f_images(path=path, shuffle=False)
    if label == 'none':
        testx = testx_typ
        names = np.array(names_typ)
        testy = np.zeros(testx_typ.shape[0])
    else:
        testx = np.concatenate([testx_typ, testx_nov])
        names = np.concatenate([names_typ, names_nov])
        testy = np.concatenate([np.zeros(testx_typ.shape[0]), np.ones(testx_nov.shape[0])])
    
    # Rescale data to [-1, 1]
    testx = _rescale_images(testx)
    
    return testx, testy, names

def get_shape_input():
    """Get shape of the dataset for Mcam"""
    return (None, 64, 64, 6)

def get_shape_input_flatten():
    """Get shape of the flatten dataset for Mcam"""
    return (None, 64*64*6)

def get_shape_label():
    """Get shape of the labels in Mcam dataset"""
    return (None,)

def num_classes():
    """Get number of classes in Mcam dataset"""
    return 2

def _mad(x):
    mad = np.median(np.abs(x-np.median(x)))
    if mad == 0:
        return 0.000001
    else:
        return mad

def _rescale_images(images):
    # Shift to -127, 127
    images = np.interp(images, (0, 255), (-127, 127))
    # Scale to [-1, 1]
    images = images / 127.
    return images

# These sequences were found to be worthy of ignoring for various reasonss
def _check_blacklist(filename):
    blacklist_seqs = ['mcam01052', 'mcam06606']
    for b in blacklist_seqs:
        if b in filename:
            return True
    return False

def _check_stripe(image):
    left_band = image[:,:10,:]
    right_band = image[:,-10:,:]
    left_black = len(np.where(left_band.flatten()<10)[0])
    right_black = len(np.where(right_band.flatten()<10)[0])
    if (left_black / float(left_band.flatten().shape[0])) > 0.10 \
         or (right_black / float(right_band.flatten().shape[0])) > 0.10:
         return True
    else:
        return False

def _load_6f_images(path, sample=False, shuffle=True, remove_striped=True):
    fns = glob(os.path.join(path, '*.npy'))
    if shuffle:
        random.shuffle(fns)
    if sample:
        fns = fns[:2000]
    num_images = len(fns)
    dataset = []
    names = []
    for fn in fns:
        if _check_blacklist(fn):
            continue
        im = np.load(fn).astype(np.float32)
        if im.shape != (64,64,6):
            continue
        elif _check_stripe(im) and remove_striped:
            continue
        else:
            dataset.append(im)
            names.append(fn.split('/')[-1][:-4])
    dataset = np.array(dataset)
    print('Loaded dataset with shape:', dataset.shape)
    return dataset, names


