import tensorflow as tf
import numpy as np
import cv2
import argparse
import math
import os.path
from os import mkdir

import datasets
import model

parser = argparse.ArgumentParser()
parser.add_argument('--results_path', help='Path to store results, including name of new directory to be made')
parser.add_argument('--model_ckpt', help='Name for checkpoint files of trained model')
parser.add_argument('--data_dir', help='Directory containing images to encode')
args = parser.parse_args()

def cae_encode():
    # Load the autoencoder graph
    ae = model.autoencoder(input_shape=[None, 64, 64, 6])
    # Load the data
    data, names = datasets.load_6f_images(args.data_dir)
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Restore weights/tensors from disk
        saver.restore(sess, args.model_ckpt)
        print("Model restored.")
        # Evaluate the encoded tensor z
        encoded = sess.run(ae['z'], feed_dict={ae['x']: data})
        print('Encoded data shape:', encoded.shape)

    # Save the encoded data
    mkdir(args.results_path)
    for i in range(data.shape[0]):
        # Save the encoded map as an RGB image
        cv2.imwrite(os.path.join(args.results_path, '%s.jpg' % names[i]), encoded[i])
        # Save the encoded map as a numpy file
        np.save(os.path.join(args.results_path, '%s.npy' % names[i]), encoded[i])

if __name__ == '__main__':
    cae_encode()