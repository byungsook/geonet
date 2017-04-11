from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


flags = tf.app.flags
FLAGS = flags.FLAGS


class BatchNorm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name='batch_norm'):
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


def _conv2d(name, input_, output_dim, k=5, s=2, stddev=0.02, train=True):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k, k, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, s, s, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        bn = BatchNorm(name='batchnorm')
        return _lrelu(bn(conv, train=train))


def _deconv2d(name, input_, output_dim, output_size, k=5, s=2, stddev=0.02, train=True, last=False):
    with tf.variable_scope(name):
        output_shape = [FLAGS.batch_size, output_size, output_size, output_dim]
        w = tf.get_variable('w', [k, k, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        # try:
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, s, s, 1])
        # # Support for verisons of TensorFlow before 0.7.0
        # except AttributeError:
        # deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
         
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if not last:
            bn = BatchNorm(name='batchnorm')        
            return tf.nn.relu(bn(deconv, train=train))
        else:
            return tf.nn.tanh(deconv)

     
def _lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak*x)


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def model1(x, crop_size, train):
    with tf.variable_scope("down-up") as scope:
        # x: 128x128x1
        h_conv01 = _conv2d('01_down', x,        64,  train=train) # 64x64x64
        h_conv02 = _conv2d('02_down', h_conv01, 128, train=train) # 32x32x128
        h_conv03 = _conv2d('03_down', h_conv02, 256, train=train) # 16x16x256
        h_conv04 = _conv2d('04_down', h_conv03, 512, train=train) # 8x8x512

        o = int(crop_size / 16) # 8
        h_conv05 = _deconv2d('05_up', h_conv04, 256, o*2,  train=train) # 16x16x256
        h_conv06 = _deconv2d('06_up', h_conv05, 128, o*4,  train=train) # 32x32x128
        h_conv07 = _deconv2d('07_up', h_conv06, 64,  o*8,  train=train) # 64x64x64
        h_conv08 = _deconv2d('08_up', h_conv07, 1,   o*16, train=train, last=True) # 128x128x1
    return h_conv08


def inference(x, crop_size, train, model=1):
    model_selector = {
        1: model1
    }
    return model_selector[model](x, crop_size, train)


def loss(y_hat, y, w=None):
    l2_loss = tf.nn.l2_loss(y_hat - y, name='l2_loss')
    return l2_loss