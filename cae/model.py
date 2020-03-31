import tensorflow as tf
import math
import numpy as np

from skimage.measure import compare_ssim

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def autoencoder(input_shape,
                n_filters=[1, 12, 8, 3],
                filter_sizes=[5, 5, 5, 6],
                loss='hybrid',
                hybrid_lambda=0):

    # input to the network
    x = tf.placeholder(tf.float32, input_shape, name='x')
    
    # fraction of neurons to keep in dropout layer
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # ensure 2-d is converted to square tensor.
    if len(x.get_shape()) == 2:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(
            x, [-1, x_dim, x_dim, n_filters[0]])
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')
    current_input = x_tensor

    input_image = current_input

    # Build the encoder
    encoder = []
    shapes = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        if layer_i == 0:
            stride=1
        else:
            stride=2
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        W = tf.clip_by_norm(W, clip_norm=4.0)
        b = tf.Variable(tf.ones([n_output]))
        encoder.append(W)
        output = lrelu(
            tf.add(tf.nn.conv2d(
                current_input, W, strides=[1, stride, stride, 1], padding='SAME'), b))

        current_input = output

    # store the latent representation
    z = current_input
    print(z.shape)
    encoder.reverse()
    shapes.reverse()

    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        if layer_i == 2:
            stride=1
        else:
            stride=2
        W = encoder[layer_i] # should be clipped already
        b = tf.Variable(tf.ones([W.get_shape().as_list()[2]]))
        output = lrelu(tf.add(
            tf.nn.conv2d_transpose(
                current_input, W,
                tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                strides=[1, stride, stride, 1], padding='SAME'), b))
        current_input = output

    # now have the reconstruction through the network
    y = current_input

    input_max = tf.reduce_max(x_tensor)
    input_min = tf.reduce_min(x_tensor)
    output_max = tf.reduce_max(y)
    output_min = tf.reduce_min(y)
    if loss == 'hybrid':
        cost = -tf.reduce_mean(tf.image.ssim(x_tensor, y, max_val=255.0)) + hybrid_lambda*tf.reduce_mean(tf.losses.mean_squared_error(x_tensor, y))
    elif loss == 'ssim':
        cost = -tf.reduce_mean(tf.image.ssim(x_tensor, y, max_val=255.0))
    elif loss == 'mse':
        cost = tf.reduce_mean(tf.losses.mean_squared_error(x_tensor, y))
    elif loss == 'bxent':
        cost = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(x_tensor, y))
    tf.summary.scalar('loss', cost)

    # Merge all the summaries
    merged = tf.summary.merge_all()

    return {'x': x, 'z': z, 'y': y, 'cost': cost, 'keep_prob': keep_prob, 'merged': merged, 'output_max': output_max, 'output_min': output_min, 'input_max': input_max, 'input_min': input_min}
