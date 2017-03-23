# Copyright (c) 2017 Byungsoo Kim. All Rights Reserved.
# 
# Byungsoo Kim, ETH Zurich
# kimby@inf.ethz.ch, http://byungsoo.me
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def _variable_summaries(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
        x: Tensor
    Returns:
        nothing
    """
    with tf.name_scope('summaries'):
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        x_name = x.op.name # re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        mean = tf.reduce_mean(x)
        tf.summary.scalar('mean/' + x_name, mean)
        tf.summary.scalar('stddev/' + x_name, tf.sqrt(tf.reduce_sum(tf.square(x - mean))))
        tf.summary.scalar('max/' + x_name, tf.reduce_max(x))
        tf.summary.scalar('min/' + x_name, tf.reduce_min(x))
        tf.summary.scalar('sparsity/' + x_name, tf.nn.zero_fraction(x))
        tf.summary.histogram(x_name, x)


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable

    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        # We instantiate all variables using tf.get_variable() instead of
        # tf.Variable() in order to share variables across multiple GPU training runs.
        # If we only ran this model on a single GPU, we could simplify this function
        # by replacing all instances of tf.get_variable() with tf.Variable().
        var = tf.get_variable(name, shape, initializer=initializer)
    return var
    

def _weight_variable(name, shape):
    """weight and summary initialization
    truncate the values more than 2 stddev and re-pick
    """
    W = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=0.1))
    # _variable_summaries(W)
    return W


def _batch_normalization(name, x, d_next, phase_train, is_conv=True):
    """batch_norm/1_scale, 2_offset, 3_batch"""
    if is_conv:
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
    else:
        batch_mean, batch_var = tf.nn.moments(x, [0])
    scale = _variable_on_cpu(name+'/1_scale', [d_next], tf.ones_initializer()) # gamma
    offset = _variable_on_cpu(name+'/2_offset', [d_next], tf.zeros_initializer()) # beta
    
    ema = tf.train.ExponentialMovingAverage(decay=0.5, name='ema')

    def _mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train,
                        _mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    
    n = tf.nn.batch_normalization(x, mean, var, offset, scale, 1e-3, name=name+'/3_batch')
    # _variable_summaries(scale)
    # _variable_summaries(offset)
    # _variable_summaries(n)
    return n


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name


    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                        decay=self.momentum, 
                        updates_collections=None,
                        epsilon=self.epsilon,
                        scale=True,
                        is_training=train,
                        scope=self.name)


def _conv2d(layer_name, x, k, s, d_next, phase_train, use_relu=True, reuse=False):
    """down, flat-convolution layer"""
    with tf.variable_scope(layer_name) as scope:
        if reuse:
            scope.reuse_variables()

        d_prev = x.get_shape()[3].value
        W = _weight_variable('1_filter_weights', [k, k, d_prev, d_next])
        conv = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME', name='2_conv_feature')
        # _variable_summaries(conv)
        # batch = _batch_normalization('3_batch_norm', conv, d_next, phase_train)
        batch = batch_norm(name='3_batch')
        if use_relu:
            # relu = tf.nn.relu(batch, name='4_relu')
            relu = tf.nn.relu(batch(conv, train=phase_train), name='4_relu')
            # _variable_summaries(relu)
            return relu
        else:
            # return batch
            return batch(conv, train=phase_train)


def _up_conv2d(layer_name, x, k, s, d_next, phase_train, use_relu=True):
    """up-convolution layer"""
    with tf.variable_scope(layer_name):
        d_prev = x.get_shape()[3].value
        W = _weight_variable('1_filter_weights', [k, k, d_prev, d_next])
        o = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, d_next]
        conv = tf.nn.conv2d_transpose(x, W, output_shape=o, strides=[1, s, s, 1], padding='SAME', name='2_upconv_feature')
        # _variable_summaries(conv)
        # batch = _batch_normalization('3_batch_norm', conv, d_next, phase_train)
        batch = batch_norm(name='3_batch')
        if use_relu:
            # relu = tf.nn.relu(batch, name='4_relu')
            relu = tf.nn.relu(batch(conv, train=phase_train), name='4_relu')
            # _variable_summaries(relu)
            return relu
        else:
            # return batch
            return batch(conv, train=phase_train)
        

def model2(x, phase_train):
    h_conv01 = _conv2d('01_flat', x,        3, 1, 64, phase_train)
    h_conv02 = _conv2d('02_flat', h_conv01, 3, 1, 64, phase_train)
    h_conv03 = _conv2d('03_flat', h_conv02, 3, 1, 64, phase_train)
    h_conv04 = _conv2d('04_flat', h_conv03, 3, 1, 64, phase_train)
    h_conv05 = _conv2d('05_flat', h_conv04, 3, 1, 64, phase_train)
    h_conv06 = _conv2d('06_flat', h_conv05, 3, 1, 64, phase_train)
    h_conv07 = _conv2d('07_flat', h_conv06, 3, 1, 64, phase_train)
    h_conv08 = _conv2d('08_flat', h_conv07, 3, 1, 64, phase_train)
    h_conv09 = _conv2d('09_flat', h_conv08, 3, 1, 64, phase_train)
    h_conv10 = _conv2d('10_flat', h_conv09, 3, 1, 64, phase_train)

    # down 1
    h_conv01_d1 = _conv2d('01-1_down', h_conv01, 3, 2, 64, phase_train, use_relu=False)
    h_conv02_d1 = _conv2d('02_flat', h_conv01_d1, 3, 1, 64, phase_train, True, True)
    h_conv03_d1 = _conv2d('03_flat', h_conv02_d1, 3, 1, 64, phase_train, True, True)
    h_conv04_d1 = _conv2d('04_flat', h_conv03_d1, 3, 1, 64, phase_train, True, True)
    h_conv05_d1 = _conv2d('05_flat', h_conv04_d1, 3, 1, 64, phase_train, True, True)
    h_conv06_d1 = _conv2d('06_flat', h_conv05_d1, 3, 1, 64, phase_train, True, True)
    h_conv07_d1 = _conv2d('07_flat', h_conv06_d1, 3, 1, 64, phase_train, True, True)
    h_conv08_d1 = _conv2d('08_flat', h_conv07_d1, 3, 1, 64, phase_train, True, True)
    h_conv09_d1 = _conv2d('09_flat', h_conv08_d1, 3, 1, 64, phase_train, True, True)
    h_conv10_d1 = _conv2d('10_flat', h_conv09_d1, 3, 1, 64, phase_train, True, True)
    h_conv10_u1 = _up_conv2d('10-1_up', h_conv10_d1, 3, 2, 64, phase_train, use_relu=False)

    # add 1
    # h_conv10_add = tf.add(h_conv10, h_conv10_u1, name='10-3_add')

    # down 2
    h_conv01_d2 = _conv2d('01-2_down', h_conv01, 3, 4, 64, phase_train, use_relu=False)
    h_conv02_d2 = _conv2d('02_flat', h_conv01_d2, 3, 1, 64, phase_train, True, True)
    h_conv03_d2 = _conv2d('03_flat', h_conv02_d2, 3, 1, 64, phase_train, True, True)
    h_conv04_d2 = _conv2d('04_flat', h_conv03_d2, 3, 1, 64, phase_train, True, True)
    h_conv05_d2 = _conv2d('05_flat', h_conv04_d2, 3, 1, 64, phase_train, True, True)
    h_conv06_d2 = _conv2d('06_flat', h_conv05_d2, 3, 1, 64, phase_train, True, True)
    h_conv07_d2 = _conv2d('07_flat', h_conv06_d2, 3, 1, 64, phase_train, True, True)
    h_conv08_d2 = _conv2d('08_flat', h_conv07_d2, 3, 1, 64, phase_train, True, True)
    h_conv09_d2 = _conv2d('09_flat', h_conv08_d2, 3, 1, 64, phase_train, True, True)
    h_conv10_d2 = _conv2d('10_flat', h_conv09_d2, 3, 1, 64, phase_train, True, True)
    h_conv10_u2 = _up_conv2d('10-2_up', h_conv10_d2, 3, 4, 64, phase_train, use_relu=False)
    
    # add 1 and 2
    h_conv10_add12 = tf.add(h_conv10_u1, h_conv10_u2, name='10-3_add12')
    h_conv10_add = tf.add(h_conv10, h_conv10_add12, name='10-3_add')
    
    h_conv11 = _conv2d('11_flat', h_conv10_add, 3, 1, 64, phase_train)
    h_conv12 = _conv2d('12_flat', h_conv11, 3, 1, 64, phase_train)
    h_conv13 = _conv2d('13_flat', h_conv12, 3, 1, 64, phase_train)
    h_conv14 = _conv2d('14_flat', h_conv13, 3, 1, 64, phase_train)
    h_conv15 = _conv2d('15_flat', h_conv14, 3, 1, 64, phase_train)
    h_conv16 = _conv2d('16_flat', h_conv15, 3, 1, 64, phase_train)
    h_conv17 = _conv2d('17_flat', h_conv16, 3, 1, 64, phase_train)
    h_conv18 = _conv2d('18_flat', h_conv17, 3, 1, 64, phase_train)
    h_conv19 = _conv2d('19_flat', h_conv18, 3, 1, 64, phase_train)
    h_conv20 = _conv2d('20_residual', h_conv19, 3, 1,  1, phase_train, use_relu=False)

    return tf.add(h_conv20, x, name='21_predict')


def model1(x, phase_train):
    with tf.variable_scope("model1") as scope:
        # all flat-convolutional layer: k=3x3, s=1x1, d=64
        h_conv01 = _conv2d('01_flat', x,        3, 1, 64, phase_train)
        h_conv02 = _conv2d('02_flat', h_conv01, 3, 1, 64, phase_train)
        h_conv03 = _conv2d('03_flat', h_conv02, 3, 1, 64, phase_train)
        h_conv04 = _conv2d('04_flat', h_conv03, 3, 1, 64, phase_train)
        h_conv05 = _conv2d('05_flat', h_conv04, 3, 1, 64, phase_train)
        h_conv06 = _conv2d('06_flat', h_conv05, 3, 1, 64, phase_train)
        h_conv07 = _conv2d('07_flat', h_conv06, 3, 1, 64, phase_train)
        h_conv08 = _conv2d('08_flat', h_conv07, 3, 1, 64, phase_train)
        h_conv09 = _conv2d('09_flat', h_conv08, 3, 1, 64, phase_train)
        h_conv10 = _conv2d('10_flat', h_conv09, 3, 1, 64, phase_train)
        h_conv11 = _conv2d('11_flat', h_conv10, 3, 1, 64, phase_train)
        h_conv12 = _conv2d('12_flat', h_conv11, 3, 1, 64, phase_train)
        h_conv13 = _conv2d('13_flat', h_conv12, 3, 1, 64, phase_train)
        h_conv14 = _conv2d('14_flat', h_conv13, 3, 1, 64, phase_train)
        h_conv15 = _conv2d('15_flat', h_conv14, 3, 1, 64, phase_train)
        h_conv16 = _conv2d('16_flat', h_conv15, 3, 1, 64, phase_train)
        h_conv17 = _conv2d('17_flat', h_conv16, 3, 1, 64, phase_train)
        h_conv18 = _conv2d('18_flat', h_conv17, 3, 1, 64, phase_train)
        h_conv19 = _conv2d('19_flat', h_conv18, 3, 1, 64, phase_train)
        h_conv20 = _conv2d('20_residual', h_conv19, 3, 1,  1, phase_train, use_relu=False)
        return tf.add(h_conv20, x, name='21_predict')


def inference(x, phase_train, model=1):
    model_selector = {
        1: model1,
        2: model2
    }
    return model_selector[model](x, phase_train)


def loss(y_hat, y, w=None):
    # y_hat: estimate, y: training set
    if w is None:
        l2_loss = tf.nn.l2_loss(y_hat - y, name='l2_loss')
    else:
        l2_loss = tf.nn.l2_loss(tf.multiply(y_hat - y, w), name='l2_loss')
    return l2_loss