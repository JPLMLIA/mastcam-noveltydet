import tensorflow as tf
import numpy as np
import cv2
import argparse
import math
import os.path
from os import mkdir

import datasets
import model

NUM_BATCHES = 5

parser = argparse.ArgumentParser()
parser.add_argument('--results_path', help='Path to store results, including name of new directory to be made')
parser.add_argument('--model_ckpt', help='Name for checkpoint files of trained model')
parser.add_argument('--data_dir', help='Directory containing test dataset images')
args = parser.parse_args()

def get_error_map(inp, recon):
    return np.square(np.subtract(inp, recon))

def test():
    # Load the autoencoder graph
    ae = model.autoencoder(input_shape=[None, 64, 64, 6])
    # Load the data
    data, names = datasets.load_6f_images(args.data_dir, shuffle=True, sample=False)
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Restore weights/tensors from disk
        saver.restore(sess, args.model_ckpt)
        print("Model restored.")

        # Evaluate test data in batches
        batch_size = data.shape[0] // NUM_BATCHES
        recon = np.ndarray(data.shape)
        encoded = np.ndarray([data.shape[0], 16, 16, 3])
        for i in range(NUM_BATCHES):
            if i == NUM_BATCHES-1:
                batch_x = data[i*batch_size:]
                recon[i*batch_size:] = sess.run(ae['y'], feed_dict={ae['x']: batch_x})
                encoded[i*batch_size:] = sess.run(ae['z'], feed_dict={ae['x']: batch_x})
            else:
                batch_x = data[i*batch_size:i*batch_size+batch_size]
                recon[i*batch_size:i*batch_size+batch_size] = sess.run(ae['y'], feed_dict={ae['x']: batch_x})
                encoded[i*batch_size:i*batch_size+batch_size] = sess.run(ae['z'], feed_dict={ae['x']: batch_x})

    # Save the various stages through the network
    mkdir(args.results_path)
    mkdir(os.path.join(args.results_path, 'reconstructions'))
    mkdir(os.path.join(args.results_path, 'error_maps'))
    mkdir(os.path.join(args.results_path, 'inputs'))
    mkdir(os.path.join(args.results_path, 'encodings'))
    for i in range(data.shape[0]):
        # Save the reconstructed images
        np.save(os.path.join(args.results_path, 'reconstructions', '%s.npy' % names[i]), recon[i])
        # Save the encoded maps
        np.save(os.path.join(args.results_path, 'encodings', '%s.npy' % names[i]), encoded[i])
        # Save the input images
        np.save(os.path.join(args.results_path, 'inputs', '%s.npy' % names[i]), data[i])
        # Save the error maps between input and reconstructed images
        np.save(os.path.join(args.results_path, 'error_maps', '%s.npy' % names[i]), get_error_map(data[i], recon[i]))

if __name__ == '__main__':
    test()
