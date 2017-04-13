# Copyright (c) 2017 Byungsoo Kim. All Rights Reserved.
# 
# Byungsoo Kim, ETH Zurich
# kimby@inf.ethz.ch, http://byungsoo.me
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import os
from datetime import datetime
import time
import threading
import random
import multiprocessing
import signal
import sys

import scipy.io
from skimage import transform

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


# parameters
flags = tf.app.flags
flags.DEFINE_string('data_dir', 'data/sketch',
                    """Path to data directory.""")
flags.DEFINE_integer('image_width', 128,
                     """Image Width.""")
flags.DEFINE_integer('image_height', 128,
                     """Image Height.""")
flags.DEFINE_integer('batch_size', 8,
                     """Number of images to process in a batch.""")
flags.DEFINE_integer('num_threads', 8,
                     """# of threads for batch generation.""")
flags.DEFINE_boolean('transform', True,
                     """whether to transform or not""")
flags.DEFINE_float('range_max', 0.00777,
                   """max range for normalization""")
FLAGS = flags.FLAGS


class BatchManager(object):
    def __init__(self):
        # read all svg files
        self._next_id = 0
        self._data_list = []
        if FLAGS.file_list:
            file_list_path = os.path.join(FLAGS.data_dir, FLAGS.file_list)
            with open(file_list_path, 'r') as f:
                while True:
                    line = f.readline()
                    if not line: break

                    file_path = os.path.join(FLAGS.data_dir, line.rstrip())
                    self._data_list.append(file_path)
        else:
            for root, _, files in os.walk(FLAGS.data_dir):
                for file in files:
                    if not file.lower().endswith('png'):
                        continue

                    file_path = os.path.join(root, file)
                    self._data_list.append(file_path)

        self.num_examples_per_epoch = len(self._data_list)
        self.num_epoch = 1

        num_cpus = multiprocessing.cpu_count()
        FLAGS.num_threads = np.amin([FLAGS.num_threads, FLAGS.batch_size, num_cpus])
        
        image_shape = [FLAGS.image_height, FLAGS.image_width, 1]
        self._q = tf.FIFOQueue(FLAGS.batch_size*10, [tf.float32, tf.float32], shapes=[image_shape, image_shape])

        self._x = tf.placeholder(dtype=tf.float32, shape=image_shape)
        self._y = tf.placeholder(dtype=tf.float32, shape=image_shape)
        self._enqueue = self._q.enqueue([self._x, self._y])


    def batch(self):
        return self._q.dequeue_many(FLAGS.batch_size)

    def start_thread(self, sess):
        # Main thread: create a coordinator.
        self._coord = tf.train.Coordinator()

        # Create a method for loading and enqueuing
        def load_n_enqueue(sess, enqueue, coord, x, y, data_list, FLAGS):
            while not coord.should_stop():
                file_path = random.choice(data_list)
                x_, y_ = preprocess(file_path, FLAGS)
                sess.run(enqueue, feed_dict={x: x_, y: y_})

        # Create threads that enqueue
        self._threads = [threading.Thread(target=load_n_enqueue, 
                                          args=(sess, 
                                                self._enqueue,
                                                self._coord,
                                                self._x,
                                                self._y,
                                                self._data_list,
                                                FLAGS)
                                          ) for i in xrange(FLAGS.num_threads)]

        # define signal handler
        def signal_handler(signum,frame):
			#print "stop training, save checkpoint..."
			#saver.save(sess, "./checkpoints/VDSR_norm_clip_epoch_%03d.ckpt" % epoch ,global_step=global_step)
			sess.run(self._q.close(cancel_pending_enqueues=True))
			self._coord.request_stop()
			self._coord.join(self._threads)
			sys.exit(1)
        signal.signal(signal.SIGINT, signal_handler)

        # Start the threads and wait for all of them to stop.
        for t in self._threads:
            t.start()

    def stop_thread(self):
        self._coord.request_stop()
        self._coord.join(self._threads)


def preprocess(file_path, FLAGS):
    if FLAGS.model == 1:
        x = scipy.io.loadmat(file_path)['result'] / 127.5 - 1.0 # [-1, 1]
    else:
        x = scipy.io.loadmat(file_path)['result'] / 255.0 # [0, 1]

    if FLAGS.transform:
        np.random.seed()

        while True:
            # random flip and rotate
            flip = (np.random.rand() > 0.5)
            if flip:
                x_rotate = np.fliplr(x)
            else:
                x_rotate = x
            r = np.random.rand() * 360.0
            x_rotate = transform.rotate(x_rotate, r, order=3, mode='symmetric')

            # # debug
            # plt.figure()
            # plt.subplot(131)
            # plt.imshow(x, cmap=plt.cm.gray)
            # plt.subplot(132)
            # plt.imshow(x_rotate, cmap=plt.cm.gray)
            # plt.subplot(133)
            # plt.imshow(transform.rotate(x, r, resize=True, mode='symmetric'), cmap=plt.cm.gray)
            # plt.show()

            # random left corner
            image_shape = x.shape
            lc_r = np.random.randint(image_shape[0]-FLAGS.image_height+1)
            lc_c = np.random.randint(image_shape[1]-FLAGS.image_width+1)
            x_crop = x_rotate[lc_r:lc_r+FLAGS.image_height, lc_c:lc_c+FLAGS.image_width]

            # check inside
            hs = int(FLAGS.image_height*0.25)
            ws = int(FLAGS.image_width*0.25)
            he = hs + int(FLAGS.image_height*0.5)
            we = ws + int(FLAGS.image_width*0.5)
            nz = np.nonzero(x_crop[hs:he, ws:we] < 1)
            if len(nz[0]) > 0:
                break
    else:
        assert(x.shape[0] == FLAGS.image_height and x.shape[1] == FLAGS.image_width)
        x_crop = x

    # transform y in the same way
    dir_path, file_name =  os.path.split(file_path)
    y_path = os.path.join(dir_path, 'disp'+file_name)
    if FLAGS.model == 1:
        y = scipy.io.loadmat(y_path)['result'] / FLAGS.range_max # [-1, 1]
    else:
        y = scipy.io.loadmat(y_path)['result'] / FLAGS.range_max * 0.5 + 0.5 # [0, 1]

    # print(batch[batch_id], np.amin(y), np.amax(y), np.average(y))

    if FLAGS.transform:
        if flip:
            y_rotate = np.fliplr(y)
        else:
            y_rotate = y
        y_rotate = transform.rotate(y_rotate, r, order=3, mode='symmetric')
        y_crop = y_rotate[lc_r:lc_r+FLAGS.image_height, lc_c:lc_c+FLAGS.image_width]
    else:
        assert(y.shape[0] == FLAGS.image_height and y.shape[1] == FLAGS.image_width)
        y_crop = y

    # # debug
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow((x_crop + 1.0) * 0.5, cmap=plt.cm.gray, clim=(0.0, 1.0))
    # plt.subplot(122)
    # plt.imshow((y_crop + 1.0) * 0.5, cmap=plt.cm.gray, clim=(0.0, 1.0))
    # plt.show()

    return x_crop[...,np.newaxis], y_crop[...,np.newaxis]


def show_sketches():
    # sketches for training/evaluation: 0-255
    plt.figure()
    x = scipy.io.loadmat('data/sketch/2812_level0.mat')['result'] / 255.0
    print(np.amin(x), np.amax(x), np.average(x))
    plt.subplot(231)
    plt.imshow(x, cmap=plt.cm.gray, clim=(0.0, 1.0))

    x = scipy.io.loadmat('data/sketch/3015_level0.mat')['result'] / 255.0
    print(np.amin(x), np.amax(x), np.average(x))
    plt.subplot(232)
    plt.imshow(x, cmap=plt.cm.gray, clim=(0.0, 1.0))
    
    x = scipy.io.loadmat('data/sketch/9991_level0.mat')['result'] / 255.0
    print(np.amin(x), np.amax(x), np.average(x))
    plt.subplot(233)
    plt.imshow(x, cmap=plt.cm.gray, clim=(0.0, 1.0))    

    x = scipy.io.loadmat('data/sketch/skcAO_fr_level0.mat')['result'] / 255.0
    print(np.amin(x), np.amax(x), np.average(x))
    plt.subplot(234)
    plt.imshow(x, cmap=plt.cm.gray, clim=(0.0, 1.0))

    x = scipy.io.loadmat('data/sketch/skcAO_sm_level0.mat')['result'] / 255.0
    print(np.amin(x), np.amax(x), np.average(x))
    plt.subplot(235)
    plt.imshow(x, cmap=plt.cm.gray, clim=(0.0, 1.0))
    
    x = scipy.io.loadmat('data/sketch/skcAO_sp_level0.mat')['result'] / 255.0
    print(np.amin(x), np.amax(x), np.average(x))
    plt.subplot(236)
    plt.imshow(x, cmap=plt.cm.gray, clim=(0.0, 1.0))    
    plt.show()


def show_displaments():
    # displacements for training
    plt.figure()
    x = scipy.io.loadmat('data/sketch/disp2812_level0.mat')['result'] / FLAGS.range_max * 0.5 + 0.5
    print(np.amin(x), np.amax(x), np.average(x))
    plt.subplot(131)
    plt.imshow(x, cmap=plt.cm.gray, clim=(0.0, 1.0))

    x = scipy.io.loadmat('data/sketch/disp3015_level0.mat')['result'] / FLAGS.range_max * 0.5 + 0.5
    print(np.amin(x), np.amax(x), np.average(x))
    plt.subplot(132)
    plt.imshow(x, cmap=plt.cm.gray, clim=(0.0, 1.0))
    
    x = scipy.io.loadmat('data/sketch/disp9991_level0.mat')['result'] / FLAGS.range_max * 0.5 + 0.5
    print(np.amin(x), np.amax(x), np.average(x))
    plt.subplot(133)
    plt.imshow(x, cmap=plt.cm.gray, clim=(0.0, 1.0))
    plt.show()


if __name__ == '__main__':
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('geonet'):
        working_path = os.path.join(current_path, 'geonet')
        os.chdir(working_path)

    # parameters 
    flags.DEFINE_string('file_list', 'train.txt', """file_list""")
    FLAGS.num_threads = 8


    # # for debug
    show_sketches()
    show_displaments()
    # preprocess('data/sketch/2812_level0.mat', FLAGS)


    batch_manager = BatchManager()
    x, y = batch_manager.batch()

    sess = tf.Session()
    batch_manager.start_thread(sess)

    test_iter = 1
    start_time = time.time()
    for _ in xrange(test_iter):
        x_batch, y_batch = sess.run([x, y])
    duration = time.time() - start_time
    duration = duration / test_iter
    batch_manager.stop_thread()

    print ('%s: %.3f sec/batch' % (datetime.now(), duration))

    plt.figure()
    for i in xrange(FLAGS.batch_size):
        x = np.reshape(x_batch[i,:], [FLAGS.image_height, FLAGS.image_width])
        y = np.reshape(y_batch[i,:], [FLAGS.image_height, FLAGS.image_width])
        x = (x + 1.0) * 0.5
        y = (y + 1.0) * 0.5
        plt.subplot(121)
        plt.imshow(x, cmap=plt.cm.gray, clim=(0.0, 1.0))
        plt.subplot(122)
        plt.imshow(y, cmap=plt.cm.gray, clim=(0.0, 1.0))
        plt.show()

    print('Done')
