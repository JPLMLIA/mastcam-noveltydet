import matplotlib.pyplot as plt
import argparse
import numpy as np
from glob import glob
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import os.path

from skimage.measure import compare_ssim

import sys
sys.path.insert(0, '/Users/hannahrae/src/mcam_novelty')
import datasets

parser = argparse.ArgumentParser()
parser.add_argument('--novel_test', help='Path to test images of novel examples with subdirectories for input and reconstructions')
parser.add_argument('--typical_test', help='Path to test images of typical examples with subdirectories for input and reconstructions')
parser.add_argument('--score', choices=['ssim','l2','outliers'], help='Which anomaly scoring function to use')
parser.add_argument('--novel_subclass', 
                    choices=['all', 'bedrock', 'broken-rock', 'drill-hole', 'drt', 'dump-pile', 'float', 'meteorite', 'veins'],
                    help='Whether to test a specific subclass of novel images or all novel images')
args = parser.parse_args()

def l2(batch1, batch2):
    l2_norm = np.ndarray(batch1.shape[0])
    for i in range(batch1.shape[0]):
        flat_diff = batch1[i].flatten() - batch2[i].flatten()
        l2_norm[i] = np.linalg.norm(flat_diff, ord=2)
    return l2_norm

def ssim(batch1, batch2):
    ssim_error = np.ndarray(batch1.shape[0])
    for i in range(batch1.shape[0]):
        ssim_error[i] = compare_ssim(batch1[i], batch2[i], multichannel=True, data_range=255.0, gaussian_weights=True)
    return ssim_error

def outliers(batch1, batch2):
    num_outliers = np.ndarray(batch1.shape[0])
    for i in range(batch1.shape[0]):
        errormap = np.square(batch1[i]-batch2[i])
        mu = np.mean(errormap)
        std = np.std(errormap)
        outlier_inds = np.where(errormap.flatten() > mu)[0]
        num_outliers[i] = len(outlier_inds)
    return num_outliers

# Compute either SSIM or l2 for all the points
test = {}
if args.score == 'ssim':
    test_typical_inputs, test_typical_names = datasets.load_data(path=args.typical_test, img_type='inputs')
    test_typical_recons, _ = datasets.load_data(path=args.typical_test, img_type='reconstructions')
    test['typical'] = ssim(test_typical_inputs, test_typical_recons)

    novel_inputs, novel_names = datasets.load_data(path=args.novel_test, img_type='inputs')
    novel_recons, _ = datasets.load_data(path=args.novel_test, img_type='reconstructions')
    test['novel'] = ssim(novel_inputs, novel_recons)

    if args.novel_subclass != 'all':
        test['novel'] = datasets.novel_subclass_only(test['novel'], novel_names, args.novel_subclass)
        print test['novel'].shape

elif args.score == 'l2':
    test_typical_inputs, test_typical_names = datasets.load_data(path=args.typical_test, img_type='inputs')
    test_typical_recons, _ = datasets.load_data(path=args.typical_test, img_type='reconstructions')
    test['typical'] = l2(test_typical_inputs, test_typical_recons)

    novel_inputs, novel_names = datasets.load_data(path=args.novel_test, img_type='inputs')
    novel_recons, _ = datasets.load_data(path=args.novel_test, img_type='reconstructions')
    test['novel'] = l2(novel_inputs, novel_recons)

    if args.novel_subclass != 'all':
        test['novel'] = datasets.novel_subclass_only(test['novel'], novel_names, args.novel_subclass)
        print test['novel'].shape

elif args.score == 'outliers':
    test_typical_inputs, test_typical_names = datasets.load_data(path=args.typical_test, img_type='inputs')
    test_typical_recons, _ = datasets.load_data(path=args.typical_test, img_type='reconstructions')
    test['typical'] = outliers(test_typical_inputs, test_typical_recons)

    novel_inputs, novel_names = datasets.load_data(path=args.novel_test, img_type='inputs')
    novel_recons, _ = datasets.load_data(path=args.novel_test, img_type='reconstructions')
    test['novel'] = outliers(novel_inputs, novel_recons)

    if args.novel_subclass != 'all':
        test['novel'] = datasets.novel_subclass_only(test['novel'], novel_names, args.novel_subclass)
        print test['novel'].shape

elif args.score == 'both':
    test_typical_inputs, test_typical_names = datasets.load_data(path=args.typical_test, img_type='inputs')
    test_typical_recons, _ = datasets.load_data(path=args.typical_test, img_type='reconstructions')
    test['typical'] = l2(test_typical_inputs, test_typical_recons) + ssim(test_typical_inputs, test_typical_recons)

    novel_inputs, novel_names = datasets.load_data(path=args.novel_test, img_type='inputs')
    novel_recons, _ = datasets.load_data(path=args.novel_test, img_type='reconstructions')
    test['novel'] = l2(novel_inputs, novel_recons) + ssim(novel_inputs, novel_recons)

    if args.novel_subclass != 'all':
        test['novel'] = datasets.novel_subclass_only(test['novel'], novel_names, args.novel_subclass)
        print test['novel'].shape
else:
    print("Error: Unsupported score type %s." % args.score)

test_scores = np.concatenate([test['typical'], test['novel']])   
y_test = np.concatenate([np.full(shape=test['typical'].shape[0], fill_value=0), np.full(shape=test['novel'].shape[0], fill_value=1)])


np.savetxt('/Users/hannahrae/data/mcam_novelty/experiments/datasetv3.3/error/error-%s-%s.txt' % (args.score, args.novel_subclass), 
            zip(test_scores,
                test_scores,
                y_test),
            fmt='%s')
