import numpy as np
import os.path
import random
random.seed(1) # for reproduceability
from glob import glob
from sklearn.preprocessing import StandardScaler

def load_data(path, img_type):
    if img_type == 'encodings':
        names = glob(os.path.join(path, img_type, '*.npy'))
        data = np.ndarray([len(names), 16, 16, 3])
        for i, f in enumerate(names):
            data[i] = np.load(f)
        #data = datasets.standardize_encoded_images(data)
        print('Loaded dataset with shape:', data.shape)
    else:
        data, names = load_6f_images(path=os.path.join(path, img_type), shuffle=False, remove_striped=False)
        #data = datasets.standardize_6f_images(data)
    return data, names

# These sequences were found to be worthy of ignoring for various reasonss
def check_blacklist(filename):
    blacklist_seqs = ['mcam01052', 'mcam06606']
    for b in blacklist_seqs:
        if b in filename:
            return True
    return False

def check_stripe(image):
    left_band = image[:,:10,:]
    right_band = image[:,-10:,:]
    left_black = len(np.where(left_band.flatten()<10)[0])
    right_black = len(np.where(right_band.flatten()<10)[0])
    if (left_black / float(left_band.flatten().shape[0])) > 0.10 \
         or (right_black / float(right_band.flatten().shape[0])) > 0.10:
         return True
    else:
        return False

def load_6f_images(path, sample=False, shuffle=True, remove_striped=True):
    fns = glob(os.path.join(path, '*.npy'))
    if shuffle:
        random.shuffle(fns)
    if sample:
    	fns = fns[:2000]
    num_images = len(fns)
    dataset = []
    names = []
    for fn in fns:
        if check_blacklist(fn):
            continue
        im = np.load(fn).astype(np.float32)
        if im.shape != (64,64,6):
            continue
        elif check_stripe(im) and remove_striped:
            continue
        else:
            dataset.append(im)
            names.append(fn.split('/')[-1][:-4])
    dataset = np.array(dataset)
    print('Loaded dataset with shape:', dataset.shape)
    return dataset, names

def standardize_6f_images(images):
    scaler = StandardScaler()
    images = np.reshape(images, (images.shape[0], 64*64*6))
    # Subtract mean
    images = images - np.mean(images, axis=0)
    # Scale to -1, 1
    images = scaler.fit_transform(images)
    images = np.reshape(images, (images.shape[0], 64, 64, 6))
    return images

def standardize_encoded_images(images):
    scaler = StandardScaler()
    images = np.reshape(images, (images.shape[0], 16*16*3))
    # Subtract mean
    images = images - np.mean(images, axis=0)
    # Scale to -1, 1
    images = scaler.fit_transform(images)
    images = np.reshape(images, (images.shape[0], 16, 16, 3))
    return images