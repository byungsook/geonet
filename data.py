import os
from glob import glob
import threading
import multiprocessing
import signal
import sys
from datetime import datetime
import platform

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import scipy.io
import skimage.transform
from tqdm import tqdm

from ops import *


class BatchManager(object):
    def __init__(self, config):
        self.root = config.data_path
        self.rng = np.random.RandomState(config.random_seed)

        org_path = os.path.join(config.data_path, 'original')
        noise_path = os.path.join(config.data_path, 'noise', 'n{}'.format(config.noise_level))

        self.paths = {}
        for data_type in ['train', 'test']:
            list_path = os.path.join(config.data_path, '{}.txt'.format(data_type))
            
            paths = []
            with open(list_path, 'r') as f:
                while True:
                    line = f.readline()
                    if not line: break
                    file_name = line.rstrip()
                    x = os.path.join(noise_path, file_name+'_n{}.mat'.format(config.noise_level))
                    y = os.path.join(org_path, file_name+'.mat')
                    paths.append({'x': x, 'y': y})
                    if data_type == 'test':
                        paths.append({'x': y, 'y': y})

            self.paths[data_type] = paths

        # print(self.paths)
        assert(len(self.paths['train']) > 0 and len(self.paths['test']) > 0)
        
        self.batch_size = config.batch_size
        self.range_max = config.range_max
        self.patch_size = config.patch_size
        patch_dim = [self.patch_size, self.patch_size, 1]

        self.image_size = config.image_size
        self.img_dim = [1, self.image_size, self.image_size, 1] # for test

        self.capacity = 10000
        self.q = tf.FIFOQueue(self.capacity, [tf.float32, tf.float32], [patch_dim, patch_dim])
        self.x = tf.placeholder(dtype=tf.float32, shape=patch_dim)
        self.y = tf.placeholder(dtype=tf.float32, shape=patch_dim)
        self.enqueue = self.q.enqueue([self.x, self.y])
        self.num_threads = config.num_worker
        # np.amin([config.num_worker, multiprocessing.cpu_count(), self.batch_size])

    def __del__(self):
        try:
            self.stop_thread()
        except AttributeError:
            pass

    def start_thread(self, sess):
        print('%s: start to enque with %d threads' % (datetime.now(), self.num_threads))

        # Main thread: create a coordinator.
        self.sess = sess
        self.coord = tf.train.Coordinator()

        # Create a method for loading and enqueuing
        def load_n_enqueue(sess, enqueue, coord, paths, rng,
                           x, y, img_size, range_max):
            with coord.stop_on_exception():                
                while not coord.should_stop():
                    id = rng.randint(len(paths))
                    x_, y_ = preprocess(paths[id], img_size, range_max, rng)
                    sess.run(enqueue, feed_dict={x: x_, y: y_})

        # Create threads that enqueue
        self.threads = [threading.Thread(target=load_n_enqueue, 
                                          args=(self.sess, 
                                                self.enqueue,
                                                self.coord,
                                                self.paths['train'],
                                                self.rng,
                                                self.x,
                                                self.y,
                                                self.patch_size,
                                                self.range_max)
                                          ) for i in range(self.num_threads)]

        # define signal handler
        def signal_handler(signum, frame):
            #print "stop training, save checkpoint..."
            #saver.save(sess, "./checkpoints/VDSR_norm_clip_epoch_%03d.ckpt" % epoch ,global_step=global_step)
            print('%s: canceled by SIGINT' % datetime.now())
            self.coord.request_stop()
            self.sess.run(self.q.close(cancel_pending_enqueues=True))
            self.coord.join(self.threads)
            sys.exit(1)
        signal.signal(signal.SIGINT, signal_handler)

        # Start the threads and wait for all of them to stop.
        for t in self.threads:
            t.start()

        # g = tf.get_default_graph()
        # g._finalized = False
        # qs = 0
        # qs_prev = 0
        # max_qs = int(self.capacity*0.8)
        # print('%s: enque %d samples first.. (will take more or less 15 mins)' % (datetime.now(), max_qs))
        # while qs < max_qs:
        #     qs = self.sess.run(self.q.size())
        # # print('%s: enque %d samples first..' % (datetime.now(), max_qs))
        # # pbar = tqdm(total=max_qs)
        # # while qs < max_qs:
        # #     qs = self.sess.run(self.q.size())
        # #     if qs > qs_prev:
        # #         diff = qs - qs_prev
        # #         pbar.update(diff)
        # #         qs_prev = qs
        # # pbar.close()
        # print('%s: done' % datetime.now())

    def stop_thread(self):
        # dirty way to bypass graph finilization error
        g = tf.get_default_graph()
        g._finalized = False

        self.coord.request_stop()
        self.sess.run(self.q.close(cancel_pending_enqueues=True))
        self.coord.join(self.threads)

    def batch(self):
        return self.q.dequeue_many(self.batch_size)

    def test_batch(self):
        for i, file_path in enumerate(self.paths['test']):
            x_, y_ = preprocess(file_path, self.image_size, self.range_max, self.rng,
                                transform=False)
            yield x_[np.newaxis,...], y_[np.newaxis,...]


def preprocess(path, img_size, range_max, rng, transform=True):
    x1 = scipy.io.loadmat(path['x'])['y'] / range_max * 0.5 + 0.5 # [0, 1]
    
    if transform:
        # random flip and rotate
        flip = (rng.rand() > 0.5)
        if flip:
            x2 = np.fliplr(x1)
        else:
            x2 = x1
        r = rng.rand() * 360.0
        x3 = skimage.transform.rotate(x2, r, order=3, mode='symmetric')

        # random left corner
        image_shape = x1.shape
        lc_r = rng.randint(image_shape[0]-img_size+1)
        lc_c = rng.randint(image_shape[1]-img_size+1)
        x = x3[lc_r:lc_r+img_size, lc_c:lc_c+img_size]
    else:
        assert(x1.shape[0] == img_size and x1.shape[1] == img_size)
        x = x1

    # transform y in the same way
    y1 = scipy.io.loadmat(path['y'])['y'] / range_max * 0.5 + 0.5 # [0, 1]

    if transform:
        if flip:
            y2 = np.fliplr(y1)
        else:
            y2 = y1
        y3 = skimage.transform.rotate(y2, r, order=3, mode='symmetric')
        y = y3[lc_r:lc_r+img_size, lc_c:lc_c+img_size]
    else:
        assert(y1.shape[0] == img_size and y1.shape[1] == img_size)
        y = y1

    # # debug
    # print(np.amax(x1), np.amin(x1))
    # print(np.amax(x3), np.amin(x3))
    # print(x.shape)
    # plt.figure()
    # plt.subplot(241)
    # plt.imshow(x1, cmap=plt.cm.gray, vmin=0, vmax=1)
    # plt.subplot(242)
    # plt.imshow(x2, cmap=plt.cm.gray, vmin=0, vmax=1)
    # plt.subplot(243)
    # plt.imshow(x3, cmap=plt.cm.gray, vmin=0, vmax=1)
    # plt.subplot(244)
    # plt.imshow(x, cmap=plt.cm.gray, vmin=0, vmax=1)
    # plt.subplot(245)
    # plt.imshow(y1, cmap=plt.cm.gray, vmin=0, vmax=1)
    # plt.subplot(246)
    # plt.imshow(y2, cmap=plt.cm.gray, vmin=0, vmax=1)
    # plt.subplot(247)
    # plt.imshow(y3, cmap=plt.cm.gray, vmin=0, vmax=1)
    # plt.subplot(248)
    # plt.imshow(y, cmap=plt.cm.gray, vmin=0, vmax=1)
    # plt.show()

    return x[...,np.newaxis], y[...,np.newaxis]

def main(config):
    prepare_dirs_and_logger(config)
    batch_manager = BatchManager(config)
    preprocess({'x': 'data/faces_100k_180119/noise/n3/YJ_n3.mat',
                'y': 'data/faces_100k_180119/original/YJ.mat'},
               batch_manager.patch_size,
               batch_manager.range_max,
               batch_manager.rng)

    # thread test
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    sess_config.log_device_placement = False
    sess = tf.Session(config=sess_config)
    batch_manager.start_thread(sess)

    x, y = batch_manager.batch()
    if config.data_format == 'NCHW':
        x = nhwc_to_nchw(x)
    x_, y_ = sess.run([x, y])
    batch_manager.stop_thread()

    if config.data_format == 'NCHW':
        x_ = x_.transpose([0, 2, 3, 1])

    x_ = x_*255
    y_ = y_*255

    save_image(x_, '{}/x_fixed.png'.format(config.model_dir))
    save_image(y_, '{}/y_fixed.png'.format(config.model_dir))

    print('batch manager test done')

if __name__ == "__main__":
    from config import get_config
    from utils import prepare_dirs_and_logger, save_config, save_image

    config, unparsed = get_config()
    main(config)