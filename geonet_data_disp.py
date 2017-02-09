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

import tensorflow as tf


# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', 'data/displacement',
                           """Path to data directory.""")
tf.app.flags.DEFINE_integer('image_width', 256,
                            """Image Width.""")
tf.app.flags.DEFINE_integer('image_height', 256,
                            """Image Height.""")
tf.app.flags.DEFINE_integer('batch_size', 4,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_processors', 8,
                            """# of processors for batch generation.""")
tf.app.flags.DEFINE_integer('noise_level', 5,
                            """noise level.""")


class MPManager(multiprocessing.managers.SyncManager):
    pass
MPManager.register('np_empty', np.empty, multiprocessing.managers.ArrayProxy)


class Param(object):
    def __init__(self):
        self.image_width = FLAGS.image_width
        self.image_height = FLAGS.image_height
        self.noise_level = FLAGS.noise_level


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
        else:
            self._mpmanager = MPManager()
            self._mpmanager.start()
            self._pool = multiprocessing.pool.Pool(processes=FLAGS.num_processors)

            self.x_batch = self._mpmanager.np_empty([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
            self.y_batch = self._mpmanager.np_empty([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
            self._batch = self._mpmanager.list(['' for _ in xrange(FLAGS.batch_size)])
            self._func = partial(train_set, batch=self._batch, x_batch=self.x_batch, y_batch=self.y_batch, FLAGS=Param())


    def __del__(self):
        if FLAGS.num_processors > 1:
            self._pool.terminate() # or close
            self._pool.join()


    def batch(self):
        if FLAGS.num_processors == 1:
            _batch = []
            for i in xrange(FLAGS.batch_size):
                _batch.append(self._data_list[self._next_id])
                train_set(i, _batch, self.x_batch, self.y_batch, FLAGS)
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

        return self.x_batch, self.y_batch


def train_set(batch_id, batch, x_batch, y_batch, FLAGS):
    x_img_path = batch[batch_id][:-4] + ('_n%d' % FLAGS.noise_level) + batch[batch_id][-4:]
    x_img = Image.open(x_img_path)
    x = np.array(x_img)[:,:,0].astype(np.float) / 255.0

    # downscale factor
    np.random.seed()
    down_scale_factor = np.random.rand()*0.75 + 0.25 # [0.25, 1)
    crop_size = int(1024*down_scale_factor)
    
    # random left corner
    lc = np.random.randint(1024-crop_size, size=2)
    x_crop = x[lc[0]:lc[0]+crop_size, lc[1]:lc[1]+crop_size]
    
    # random flip and rotate
    flip = (np.random.rand() > 0.5)
    if flip:
        x_crop = np.fliplr(x_crop)
    r = np.random.randint(low=-180, high=180)
    x_crop = transform.rotate(x_crop, r, mode='symmetric')

    # resize
    x_crop = transform.resize(x_crop, (FLAGS.image_height, FLAGS.image_width), mode='symmetric')

    # transform y in the same way
    y_img = Image.open(batch[batch_id])
    y = np.array(y_img)[:,:,0].astype(np.float) / 255.0

    y_crop = y[lc[0]:lc[0]+crop_size, lc[1]:lc[1]+crop_size]
    if flip:
        y_crop = np.fliplr(y_crop)
    y_crop = transform.rotate(y_crop, r, mode='symmetric')
    y_crop = transform.resize(y_crop, (FLAGS.image_height, FLAGS.image_width), mode='symmetric')

    # # debug
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(x_crop, cmap=plt.cm.gray)
    # plt.subplot(122)
    # plt.imshow(y_crop, cmap=plt.cm.gray)
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    # plt.show()
    # print('x', np.amin(x_crop), np.amax(x_crop))
    # print('y', np.amin(y_crop), np.amax(y_crop))

    x_batch[batch_id,:,:,0] = x_crop
    y_batch[batch_id,:,:,0] = y_crop


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
    tf.app.flags.DEFINE_string('file_list', 'train.txt', """file_list""")
    FLAGS.num_processors = 1
    
    # split_dataset()

    batch_manager = BatchManager()
    x_batch, y_batch = batch_manager.batch()

    plt.figure()        
    for i in xrange(FLAGS.batch_size):
        plt.subplot(121)
        plt.imshow(np.reshape(x_batch[i,:], [FLAGS.image_height, FLAGS.image_width]), cmap=plt.cm.gray)
        plt.subplot(122)
        plt.imshow(np.reshape(y_batch[i,:], [FLAGS.image_height, FLAGS.image_width]), cmap=plt.cm.gray)
        plt.show()

    print('Done')
