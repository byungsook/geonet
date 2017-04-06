# Copyright (c) 2017 Byungsoo Kim. All Rights Reserved.
# 
# Byungsoo Kim, ETH Zurich
# kimby@inf.ethz.ch, http://byungsoo.me
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import io
from random import shuffle
import copy
import multiprocessing.managers
import multiprocessing.pool
from functools import partial
import platform

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import transform
from skimage import feature
from skimage.filters import roberts, sobel, scharr, prewitt
# from skimage.feature import canny
from scipy import stats
import scipy.io

import tensorflow as tf


# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', 'data/faces_low_res/maps/100k/original',
                           """Path to data directory.""")
tf.app.flags.DEFINE_integer('image_width', 128,
                            """Image Width.""")
tf.app.flags.DEFINE_integer('image_height', 128,
                            """Image Height.""")
tf.app.flags.DEFINE_integer('batch_size', 8,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_processors', 8,
                            """# of processors for batch generation.""")
tf.app.flags.DEFINE_boolean('transform', True,
                          """whether to transform or not""")
# tf.app.flags.DEFINE_float('min_scale', 0.125,
#                             """minimum of downscale factor.""")
tf.app.flags.DEFINE_string('noise_level', 'n1',
                            """noise level.""")
tf.app.flags.DEFINE_boolean('weight_on', False,
                          """whether to use weight for sharp features or not""")
tf.app.flags.DEFINE_float('weight_sigma', 0.5,
                          """sigma for weight kernel""")


class MPManager(multiprocessing.managers.SyncManager):
    pass
MPManager.register('np_empty', np.empty, multiprocessing.managers.ArrayProxy)


class Param(object):
    def __init__(self):
        self.image_width = FLAGS.image_width
        self.image_height = FLAGS.image_height
        self.transform = FLAGS.transform
        # self.min_scale = FLAGS.min_scale
        self.noise_level = FLAGS.noise_level
        self.weight_on = FLAGS.weight_on
        self.weight_sigma = FLAGS.weight_sigma
        self.is_train = FLAGS.is_train


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

        if FLAGS.num_processors > FLAGS.batch_size:
            FLAGS.num_processors = FLAGS.batch_size

        if FLAGS.num_processors == 1:
            self.x_batch = np.zeros([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
            self.y_batch = np.zeros([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
            self.w_batch = np.zeros([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
        else:
            self._mpmanager = MPManager()
            self._mpmanager.start()
            self._pool = multiprocessing.pool.Pool(processes=FLAGS.num_processors)

            self.x_batch = self._mpmanager.np_empty([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
            self.y_batch = self._mpmanager.np_empty([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
            self.w_batch = self._mpmanager.np_empty([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
            self._batch = self._mpmanager.list(['' for _ in xrange(FLAGS.batch_size)])
            self._func = partial(train_set, batch=self._batch, x_batch=self.x_batch, y_batch=self.y_batch, w_batch=self.w_batch, FLAGS=Param())


    def __del__(self):
        if FLAGS.num_processors > 1:
            self._pool.terminate() # or close
            self._pool.join()


    def batch(self):
        if FLAGS.num_processors == 1:
            _batch = []
            for i in xrange(FLAGS.batch_size):
                _batch.append(self._data_list[self._next_id])
                train_set(i, _batch, self.x_batch, self.y_batch, self.w_batch, FLAGS)
                self._next_id = (self._next_id + 1) % len(self._data_list)
                if self._next_id == 0:
                    self.num_epoch = self.num_epoch + 1
                    shuffle(self._data_list)
        else:
            for i in xrange(FLAGS.batch_size):
                self._batch[i] = self._data_list[self._next_id]
                self._next_id = (self._next_id + 1) % len(self._data_list)
                if self._next_id == 0:
                    self.num_epoch = self.num_epoch + 1
                    shuffle(self._data_list)

            self._pool.map(self._func, range(FLAGS.batch_size))

        return self.x_batch, self.y_batch, self.w_batch


def train_set(batch_id, batch, x_batch, y_batch, w_batch, FLAGS):
    # x_path = batch[batch_id][:-4] + ('_%s' % FLAGS.noise_level) + batch[batch_id][-4:]
    dir_path, file_path = os.path.split(batch[batch_id])
    x_path = dir_path + ('/../%s/' % FLAGS.noise_level)  + file_path[:-4] + ('_%s' % FLAGS.noise_level) + file_path[-4:]
    if batch[batch_id][-3:] == 'png':
        x_img = Image.open(x_path)
        x = np.array(x_img)[:,:,0].astype(np.float) / 255.0
    else:
        RANGE_MAX = 0.075
        x = scipy.io.loadmat(x_path)['result'] / RANGE_MAX * 0.5 + 0.5 # [0, 1]
        # print(x_path, np.amin(x), np.amax(x), np.average(x))

    if FLAGS.transform:
        np.random.seed()

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

        # down_scale_factor = np.random.rand()*(1 - FLAGS.min_scale) + FLAGS.min_scale # [FLAGS.min_scale, 1)
        # down_scale_factor = FLAGS.min_scale
        # crop_size = int(image_size*down_scale_factor)

        # random left corner
        image_shape = x.shape
        lc_r = np.random.randint(image_shape[0]-FLAGS.image_height+1)
        lc_c = np.random.randint(image_shape[1]-FLAGS.image_width+1)
        x_crop = x_rotate[lc_r:lc_r+FLAGS.image_height, lc_c:lc_c+FLAGS.image_width]
    else:
        assert(x.shape[0] == FLAGS.image_height and x.shape[1] == FLAGS.image_width)
        x_crop = x

    # transform y in the same way
    if batch[batch_id][-3:] == 'png':
        y_img = Image.open(batch[batch_id])
        y = np.array(y_img)[:,:,0].astype(np.float) / 255.0
    else:
        y = scipy.io.loadmat(batch[batch_id])['result'] / RANGE_MAX * 0.5 + 0.5 # [0, 1]
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

    # use clean input with a probability of 10%
    if FLAGS.is_train and np.random.rand() < 0.1:
        x_crop = y_crop

    # edge detection for loss weight
    if FLAGS.weight_on:
        w_crop = scharr(y_crop)
        # w_crop = canny(y_crop) # worse for this case
        # w_crop += 1
        w_crop /= np.amax(w_crop) # [0 1]
        # w_crop = scipy.stat.threshold(w_crop, threshmin=0.5, threshmax=1, newval=0)
        if FLAGS.weight_sigma > 0:
            w_crop = np.exp(-0.5 * ((1.0-w_crop) / FLAGS.weight_sigma)**2)
        w_crop += 1 # [1 2]
        # w_crop *= 1000 # [0 1000]
        # w_crop += 1 # [1 1001]
    else:
        w_crop = np.ones([FLAGS.image_height, FLAGS.image_width])

    # # debug
    # plt.figure()
    # # plt.subplot(121)
    # # plt.imshow(x_crop, cmap=plt.cm.gray, clim=(0.0, 1.0))
    # # plt.subplot(122)
    # # plt.imshow(y_crop, cmap=plt.cm.gray, clim=(0.0, 1.0))
    # plt.subplot(231)
    # plt.imshow(x_crop, cmap=plt.cm.gray)
    # plt.subplot(232)
    # plt.imshow(y_crop, cmap=plt.cm.gray)
    # plt.subplot(233)
    # plt.imshow(w_crop, cmap=plt.cm.gray)
    # w_crop_r = roberts(y_crop)
    # w_crop_s = sobel(y_crop)
    # w_crop_p = prewitt(y_crop)
    # # w_crop_r = canny(y_crop, sigma=0.1)
    # # w_crop_s = canny(y_crop, sigma=0.3)
    # # w_crop_r = canny(y_crop, sigma=0.7)
    # plt.subplot(234)
    # plt.imshow(w_crop_r, cmap=plt.cm.gray)
    # plt.subplot(235)
    # plt.imshow(w_crop_s, cmap=plt.cm.gray)
    # plt.subplot(236)
    # plt.imshow(w_crop_p, cmap=plt.cm.gray)
    # # mng = plt.get_current_fig_manager()
    # # mng.full_screen_toggle()
    # plt.show()
    # # print('x', np.amin(x_crop), np.amax(x_crop))
    # # print('y', np.amin(y_crop), np.amax(y_crop))
    # # print('w', np.amin(w_crop), np.amax(w_crop))

    x_batch[batch_id,:,:,0] = x_crop
    y_batch[batch_id,:,:,0] = y_crop
    w_batch[batch_id,:,:,0] = w_crop


def split_dataset():
    file_list = []
    for root, _, files in os.walk(FLAGS.data_dir):
        for file in files:
            if not file.lower().endswith('png'):
                continue

            file_path = os.path.join(root, file)
            file_list.append(file_path)

    num_files = len(file_list)
    ids = np.random.permutation(num_files)
    train_id = int(num_files * 0.9)
    with open(os.path.join(FLAGS.data_dir,'train.txt'), 'w') as f: 
        for id in ids[:train_id]:
            f.write(file_list[id] + '\n')
    with open(os.path.join(FLAGS.data_dir,'test.txt'), 'w') as f: 
        for id in ids[train_id:]:
            f.write(file_list[id] + '\n')


if __name__ == '__main__':
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('geonet'):
        working_path = os.path.join(current_path, 'geonet')
        os.chdir(working_path)

    # parameters 
    tf.app.flags.DEFINE_string('file_list', 'train_mat.txt', """file_list""")
    tf.app.flags.DEFINE_boolean('is_train', True, """is train""")
    FLAGS.num_processors = 1
    # # eval
    # FLAGS.file_list = 'test_mat.txt'
    # FLAGS.is_train = False
    # FLAGS.image_height = 1024
    # FLAGS.image_width = 1024
    # FLAGS.transform = False
    # FLAGS.batch_size = 1
    
    # split_dataset()

    batch_manager = BatchManager()
    x_batch, y_batch, w_batch = batch_manager.batch()

    plt.figure()
    for i in xrange(FLAGS.batch_size):
        plt.subplot(121)
        plt.imshow(np.reshape(x_batch[i,:], [FLAGS.image_height, FLAGS.image_width]), cmap=plt.cm.gray, clim=(0.0, 1.0))
        plt.subplot(122)
        plt.imshow(np.reshape(y_batch[i,:], [FLAGS.image_height, FLAGS.image_width]), cmap=plt.cm.gray, clim=(0.0, 1.0))
        # plt.subplot(133)
        # plt.imshow(np.reshape(w_batch[i,:], [FLAGS.image_height, FLAGS.image_width]), cmap=plt.cm.gray, clim=(0.0, 1.0))
        plt.show()

    print('Done')
