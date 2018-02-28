import numpy as np
import tensorflow as tf
from ops import *
slim = tf.contrib.slim

def VDSR(x_, hidden_num, repeat_num, data_format, name='VDSR', train=True, reuse=False):
    x = x_ # noisy image
    with tf.variable_scope(name, reuse=reuse) as vs:
        for i in range(repeat_num-1):
            x = batch_norm(conv2d(x, hidden_num, data_format, k=3, s=1),
                           train, data_format, act=tf.nn.relu)

        x = batch_norm(conv2d(x, 1, data_format, k=3, s=1), train, data_format) # -noise
        out = x_ + x # clean image
    variables = tf.contrib.framework.get_variables(vs)
    return out, variables
        
def DnCNN(x_, filters, repeat_num, data_format, name='DnCNN', train=True, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as vs:
        x = conv2d(x_, filters, data_format, k=3, s=1, act=tf.nn.relu)
        for i in range(2, repeat_num):
            x = conv2d(x, filters, data_format, k=3, s=1, bias=False)
            x = batch_norm(x, train, data_format, act=tf.nn.relu)
        out = x_ - conv2d(x, 1, data_format, k=3, s=1) # clean image
    variables = tf.contrib.framework.get_variables(vs)
    return out, variables

def main(_):
    b_num = 8
    h = 128
    w = 128
    ch_num = 1
    data_format = 'NCHW'

    x = tf.placeholder(dtype=tf.float32, shape=[b_num, h, w, ch_num])
    if data_format == 'NCHW':
        x = nhwc_to_nchw(x)

    filters = 64
    repeat_num = 20
    y = VDSR(x, filters, repeat_num, data_format)

    # repeat_num = 17
    # y = DnCNN(x, filters, repeat_num, data_format)
    
    show_all_variables()

if __name__ == '__main__':
    tf.app.run()