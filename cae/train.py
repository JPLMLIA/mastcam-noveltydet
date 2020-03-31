import tensorflow as tf
import numpy as np
import math
import argparse
import os.path
import os

import datasets
import model

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training')
parser.add_argument('--convergence_delta', type=float, default=0.003, help='Change between previous loss and current loss at which convergence is determined')
parser.add_argument('--hybrid_lambda', type=float, default=0, help='Weight for MSE loss in hybrid loss function')
parser.add_argument('--loss_fn', choices=['mse', 'ssim', 'hybrid', 'bxent'], help='Loss function to minimize in training')
parser.add_argument('--gpu', choices=['0', '1', '2', '3'], help='Which GPU to run on')
parser.add_argument('--summaries_dir', default='/scratch/hkerner/tf-summaries', help='Where to save training summaries')
parser.add_argument('--train_dir', help='Path to training images')
parser.add_argument('--val_dir', help='Path to validation images')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def train():
    # load mastcam data
    train_data, train_names = datasets.load_6f_images(path=args.train_dir)
    val_data, val_names = datasets.load_6f_images(path=args.val_dir)

    ae = model.autoencoder(input_shape=[None, 64, 64, 6], loss=args.loss_fn, hybrid_lambda=args.hybrid_lambda)

    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1.0).minimize(ae['cost'])

    # We create a session to use the graph
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(os.path.join(args.summaries_dir, 'train'),
                                          sess.graph)
    val_writer = tf.summary.FileWriter(os.path.join(args.summaries_dir, 'validation'))
    sess.run(tf.global_variables_initializer())

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Fit all training data
    num_batches = train_data.shape[0] / args.batch_size
    print("num batches = %d" % num_batches)
    step = 1
    epochs = 1
    prev_loss = 10000000
    while True:
        for batch_i in range(num_batches):
            idx = batch_i*args.batch_size
            batch_xs = train_data[idx:idx+args.batch_size]
            sess.run(optimizer, feed_dict={ae['x']: batch_xs, ae['keep_prob']: 0.6})
            train_loss = sess.run(ae['merged'], feed_dict={ae['x']: batch_xs, ae['keep_prob']: 0.6})
            train_writer.add_summary(train_loss, step)
            step += 1
        val_loss = sess.run(ae['merged'], feed_dict={ae['x']: val_data, ae['keep_prob']: 1.0})
        val_writer.add_summary(val_loss, step)
        loss = sess.run(ae['cost'], feed_dict={ae['x']: val_data, ae['keep_prob']: 1.0})
        print('Validation loss at Epoch %d (delta=%f)' % (epochs, prev_loss - loss), loss)
        if prev_loss - loss < args.convergence_delta:
            break
        prev_loss = loss
        epochs += 1

    # Save the model for future training or testing
    name = 'udr_12-8-3_3-3-3_nodrop_loss=%s_lambda=%f_epochs=%d_data=%s' % (args.loss_fn, args.hybrid_lambda, epochs, args.train_dir.split('/')[-2])
    save_path = saver.save(sess, "/scratch/hkerner/saved_sessions/%s.ckpt" % name)
    print("Model saved in path: %s" % save_path)

    val_writer.close()
    train_writer.close()

# %%
if __name__ == '__main__':
    train()
