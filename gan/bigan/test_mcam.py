import time
import numpy as np
import tensorflow as tf
import logging
import importlib
import sys
import bigan.mcam_utilities as network
import data.mcam as data
from utils.evaluations import do_prc, do_roc
from sklearn.metrics import roc_curve, auc

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

RANDOM_SEED = 13
FREQ_PRINT = 20 # print frequency image tensorboard [20]
CKPT = './bigan_results/train_logs/mcam/fm/0.1/demo/42/model.ckpt'

def get_getter(ema):  # to update neural net with moving avg variables, suitable for ss learning cf Saliman
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var

    return ema_getter

def test(weight, method, degree, random_seed, label):
    """ Runs the Bigan on the Mastcam dataset

    Note:
        Saves summaries on tensorboard. To display them, please use cmd line
        tensorboard --logdir=model.training_logdir() --port=number
    Args:
        nb_epochs (int): number of epochs
        weight (float, optional): weight for the anomaly score composition
        method (str, optional): 'fm' for ``Feature Matching`` or "cross-e"
                                     for ``cross entropy``, "efm" etc.
        anomalous_label (int): int in range 0 to 10, is the class/digit
                                which is considered outlier
    """
    # Placeholders
    input_pl = tf.placeholder(tf.float32, shape=data.get_shape_input(), name="input")
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_pl")

    # Test Data
    testx, testy, testnames = data.get_test('all')

    # Parameters
    starting_lr = network.learning_rate
    batch_size = network.batch_size
    latent_dim = network.latent_dim
    ema_decay = 0.999

    rng = np.random.RandomState(RANDOM_SEED)
    nr_batches_test = int(testx.shape[0] / batch_size)

    gen = network.decoder
    enc = network.encoder
    dis = network.discriminator

    with tf.variable_scope('encoder_model'):
        z_gen = enc(input_pl, is_training=is_training_pl)

    with tf.variable_scope('generator_model'):
        z = tf.random_normal([batch_size, latent_dim])
        x_gen = gen(z, is_training=is_training_pl)
        reconstruct = gen(z_gen, is_training=is_training_pl, reuse=True)

    with tf.variable_scope('discriminator_model'):
        l_encoder, inter_layer_inp = dis(z_gen, input_pl, is_training=is_training_pl) 
        l_generator, inter_layer_rct = dis(z, x_gen, is_training=is_training_pl, reuse=True)
        #l_generator, inter_layer_rct = dis(z_gen, reconstruct, is_training=is_training_pl, reuse=True)

    with tf.name_scope('loss_functions'):
        # discriminator
        loss_dis_enc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.random_uniform(shape=tf.shape(l_encoder), minval=0.9, maxval=1.0),logits=l_encoder))
        #loss_dis_enc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant(0.9, shape=tf.shape(l_encoder)),logits=l_encoder))
        loss_dis_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_generator),logits=l_generator))
        # loss_dis_enc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_encoder),logits=l_encoder))
        # loss_dis_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_generator),logits=l_generator))
        loss_discriminator = loss_dis_gen + loss_dis_enc

        # generator
        #loss_reconstruction = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=input_pl, logits=reconstruct))
        loss_reconstruction = tf.reduce_mean(tf.losses.mean_squared_error(labels=input_pl, predictions=reconstruct))
        #loss_reconstruction = tf.norm(tf.contrib.layers.flatten(input_pl-reconstruct), ord=2)
        loss_features = tf.norm(tf.contrib.layers.flatten(inter_layer_inp-inter_layer_rct), ord=2)
        # loss_dis_gen_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_generator),logits=l_generator))
        # loss_dis_gen_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(input_pl),logits=input_pl))
        # loss_dis = -tf.log(loss_dis_gen_fake) + tf.log(1-loss_dis_gen_real)
        loss_dis = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_generator),logits=l_generator))
        loss_generator = 1*loss_dis + 0.4*loss_reconstruction + 0*loss_features

        # encoder
        # test adding loss in encoder instead?
        loss_encoder = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_encoder),logits=l_encoder)) + 0.4*loss_reconstruction

    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        dvars = [var for var in tvars if 'discriminator_model' in var.name]
        gvars = [var for var in tvars if 'generator_model' in var.name]
        evars = [var for var in tvars if 'encoder_model' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
        update_ops_enc = [x for x in update_ops if ('encoder_model' in x.name)]
        update_ops_dis = [x for x in update_ops if ('discriminator_model' in x.name)]

        optimizer_dis = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='dis_optimizer')
        #optimizer_dis = tf.train.GradientDescentOptimizer(learning_rate=0.0001, name='dis_optimizer')
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='gen_optimizer')
        optimizer_enc = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='enc_optimizer')

        with tf.control_dependencies(update_ops_gen):
            gen_op = optimizer_gen.minimize(loss_generator, var_list=gvars)
        with tf.control_dependencies(update_ops_enc):
            enc_op = optimizer_enc.minimize(loss_encoder, var_list=evars)
        with tf.control_dependencies(update_ops_dis):
            dis_op = optimizer_dis.minimize(loss_discriminator, var_list=dvars)

        # Exponential Moving Average for estimation
        dis_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_dis = dis_ema.apply(dvars)

        with tf.control_dependencies([dis_op]):
            train_dis_op = tf.group(maintain_averages_op_dis)

        gen_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_gen = gen_ema.apply(gvars)

        with tf.control_dependencies([gen_op]):
            train_gen_op = tf.group(maintain_averages_op_gen)

        enc_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_enc = enc_ema.apply(evars)

        with tf.control_dependencies([enc_op]):
            train_enc_op = tf.group(maintain_averages_op_enc)

    with tf.variable_scope('encoder_model'):
        z_gen_ema = enc(input_pl, is_training=is_training_pl,
                        getter=get_getter(enc_ema), reuse=True)

    with tf.variable_scope('generator_model'):
        reconstruct_ema = gen(z_gen_ema, is_training=is_training_pl,
                              getter=get_getter(gen_ema), reuse=True)

    with tf.variable_scope('discriminator_model'):
        l_encoder_ema, inter_layer_inp_ema = dis(z_gen_ema,
                                                 input_pl,
                                                 is_training=is_training_pl,
                                                 getter=get_getter(dis_ema),
                                                 reuse=True)
        l_generator_ema, inter_layer_rct_ema = dis(z_gen_ema,
                                                   reconstruct_ema,
                                                   is_training=is_training_pl,
                                                   getter=get_getter(dis_ema),
                                                   reuse=True)
    with tf.name_scope('Testing'):
        with tf.variable_scope('Reconstruction_loss'):
            delta = input_pl - reconstruct_ema
            delta_flat = tf.contrib.layers.flatten(delta)
            #gen_score = tf.reduce_mean(tf.losses.mean_squared_error(input_pl, reconstruct_ema))
            gen_score = tf.norm(delta_flat, ord=degree, axis=1,
                              keep_dims=False, name='epsilon')

        with tf.variable_scope('Discriminator_loss'):
            if method == "cross-e":
                dis_score = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(l_generator_ema),logits=l_generator_ema)

            elif method == "fm":
                fm = inter_layer_inp_ema - inter_layer_rct_ema
                fm = tf.contrib.layers.flatten(fm)
                dis_score = tf.norm(fm, ord=degree, axis=1,
                                 keep_dims=False, name='d_loss')

            dis_score = tf.squeeze(dis_score)

        with tf.variable_scope('Score'):
            list_scores = (1 - weight) * gen_score + weight * dis_score


    saver = tf.train.Saver()

    with tf.Session() as sess:

        # Restore weights/tensors from disk
        saver.restore(sess, CKPT)
        print("Model restored.")

        inds = rng.permutation(testx.shape[0])
        testx = testx[inds]  # shuffling  dataset
        testy = testy[inds] # shuffling  dataset
        testnames = testnames[inds]
        scores = []
        inference_time = []

        test_encodings = np.ndarray([testx.shape[0], network.latent_dim])
        test_reconstructions = np.ndarray(testx.shape)

        # Create scores
        for t in range(nr_batches_test):

            # construct randomly permuted minibatches
            ran_from = t * batch_size
            ran_to = (t + 1) * batch_size
            begin_val_batch = time.time()

            feed_dict = {input_pl: testx[ran_from:ran_to],
                         is_training_pl:False}

            scores += sess.run(list_scores,
                                         feed_dict=feed_dict).tolist()

            # store z_gen_ema (encoding)
            test_encodings[ran_from:ran_to] = sess.run(z_gen_ema,
                                                        feed_dict=feed_dict).tolist()
            # store reconstruct_ema (reconstruction)
            test_reconstructions[ran_from:ran_to] = sess.run(reconstruct_ema,
                                                        feed_dict=feed_dict).tolist()

            inference_time.append(time.time() - begin_val_batch)

        print('Testing : mean inference time is %.4f' % (
            np.mean(inference_time)))

        ran_from = nr_batches_test * batch_size
        ran_to = (nr_batches_test + 1) * batch_size
        size = testx[ran_from:ran_to].shape[0]
        fill = np.ones([batch_size - size, 64, 64, 6])

        batch = np.concatenate([testx[ran_from:ran_to], fill], axis=0)
        feed_dict = {input_pl: batch,
                     is_training_pl: False}

        batch_score = sess.run(list_scores,
                           feed_dict=feed_dict).tolist()

        scores += batch_score[:size]

        roc_auc = do_roc(scores, testy, testnames,
               file_name=r'bigan/mcam/{}/{}/{}'.format(method, weight,
                                                     label),
               directory=r'results/bigan/mcam/{}/{}/'.format(method,
                                                           weight))

        os.mkdir('results/bigan/mcam/{}/{}/{}'.format(method,
                                                           weight, label))
        os.mkdir(os.path.join('results/bigan/mcam/{}/{}/{}'.format(method,
                                                           weight, label), 'reconstructions'))
        os.mkdir(os.path.join('results/bigan/mcam/{}/{}/{}'.format(method,
                                                           weight, label), 'encodings'))
        os.mkdir(os.path.join('results/bigan/mcam/{}/{}/{}'.format(method,
                                                           weight, label), 'inputs'))
        os.mkdir(os.path.join('results/bigan/mcam/{}/{}/{}'.format(method,
                                                           weight, label), 'error_maps'))
        for i in range(testx.shape[0]):
            # Save the reconstructed images
            np.save(os.path.join('results/bigan/mcam/{}/{}/{}'.format(method,
                                                           weight, label), 'reconstructions', '%s.npy' % testnames[i]), test_reconstructions[i])
            # Save the encoded maps
            np.save(os.path.join('results/bigan/mcam/{}/{}/{}'.format(method,
                                                           weight, label), 'encodings', '%s.npy' % testnames[i]), test_encodings[i])
            # Save the input images
            np.save(os.path.join('results/bigan/mcam/{}/{}/{}'.format(method,
                                                           weight, label), 'inputs', '%s.npy' % testnames[i]), testx[i])
            # Save the error maps between input and reconstructed images
            np.save(os.path.join('results/bigan/mcam/{}/{}/{}'.format(method,
                                                           weight, label), 'error_maps', '%s.npy' % testnames[i]), np.square(np.subtract(testx[i], test_reconstructions[i])))

        print("Testing | ROC AUC = {:.4f}".format(roc_auc))

def run(nb_epochs, weight, method, degree, label, random_seed=42):
    """ Runs the training process"""
    with tf.Graph().as_default():
        # Set the graph level seed
        tf.set_random_seed(random_seed)
        test(weight, method, degree, random_seed, label)
