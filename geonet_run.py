# Copyright (c) 2017 Byungsoo Kim. All Rights Reserved.
# 
# Byungsoo Kim, ETH Zurich
# kimby@inf.ethz.ch, http://byungsoo.me
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import time
import math

from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import scipy.io
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

import geonet_model


# parameters
flags = tf.app.flags
flags.DEFINE_string('result_dir', 'result/face_whole_0.01_32',
                    """Directory where to write event logs """
                    """and checkpoint.""")
flags.DEFINE_string('data_dir', 'data/10FacialModels_whole',
                    """Directory where to write event logs """
                    """and checkpoint.""")
flags.DEFINE_string('file_list', 'test_mat.txt',
                    """file_list""")
flags.DEFINE_string('checkpoint_dir', 'log/face_whole_0.01_32',
                    """If specified, restore this pretrained model.""")
flags.DEFINE_float('moving_avg_decay', 0.9999,
                   """The decay to use for the moving average.""")
flags.DEFINE_integer('crop_size', 1024, # 128
                     """crop size.""")
flags.DEFINE_integer('batch_size', 1, # 16
                     """batch size.""")
flags.DEFINE_string('noise_level', 'n1',
                    """noise level.""")
FLAGS = flags.FLAGS


def run():
    num_files = 0
    file_path_list = []
    gt_path_list = []

    if FLAGS.file_list:
        file_list_path = os.path.join(FLAGS.data_dir, FLAGS.file_list)
        with open(file_list_path, 'r') as f:
            while True:
                line = f.readline()
                if not line: break

                file = line.rstrip()
                file_path = os.path.join(FLAGS.data_dir, file)
                gt_path_list.append(file_path)
                # file_path = file_path[:-4] + ('_%.3f' % FLAGS.noise_level) + file_path[-4:]
                dir_path, file_path = os.path.split(file_path)
                file_path = dir_path + ('/../%s/' % FLAGS.noise_level)  + file_path[:-4] + ('_%s' % FLAGS.noise_level) + file_path[-4:]
                file_path_list.append(file_path)
    else:
        for root, _, files in os.walk(FLAGS.data_dir):
            for file in files:
                if not file.lower().endswith('png'):
                    continue

                file_path = os.path.join(root, file)
                file_path_list.append(file_path)

    num_files = len(file_path_list)


    global_step = tf.Variable(0, name='global_step', trainable=False)
    is_train = False
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    x_ph = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.crop_size, FLAGS.crop_size, 1])
    
    # Build a Graph that computes the logits predictions from the inference model.
    y_hat = geonet_model.inference(x_ph, phase_train)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    # saver = tf.train.Saver()

    # Start evaluation
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Create a saver (restorer).
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and FLAGS.checkpoint_dir:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(FLAGS.checkpoint_dir, ckpt_name))
        print('%s: Pre-trained model restored from %s' % 
            (datetime.now(), ckpt_name))
    else:
        print('cannot find pretrained model')
        assert(False)

    stat_path = os.path.join(FLAGS.result_dir, 'stat.txt')
    f = open(stat_path, 'w')

    for file_id, file_path in enumerate(file_path_list):
        start_time = time.time()
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        print('%s: %d/%d-%s start to process' % (datetime.now(), file_id+1, num_files, file_name))
        f.write('%s: %d/%d-%s start to process\n' % (datetime.now(), file_id+1, num_files, file_name))
        
        # matrix input
        RANGE_MAX = 0.2
        x = scipy.io.loadmat(file_path)['result'] / RANGE_MAX * 0.5 + 0.5 # [0, 1]

        # # image input
        # x_img = Image.open(file_path)
        # x = np.array(x_img)[:,:,0].astype(np.float) / 255.0

        # if there is ground truth
        if len(gt_path_list) > 0:
            # y_img = Image.open(gt_path_list[file_id])
            # y_gt = np.array(y_img)[:,:,0].astype(np.float) / 255.0
            y_gt = scipy.io.loadmat(gt_path_list[file_id])['result'] / RANGE_MAX * 0.5 + 0.5 # [0, 1]
        else:
            y_gt = None


        x_shape = x.shape
        assert(x_shape[0] % FLAGS.crop_size == 0 and x_shape[1] % FLAGS.crop_size == 0)

        x_batch = np.zeros([FLAGS.batch_size, FLAGS.crop_size, FLAGS.crop_size, 1], dtype=np.float)
        y = np.zeros(x_shape, dtype=np.float)
        batch_id = 0
        batch_position = []

        for r in range(x_shape[0]//FLAGS.crop_size):
            for c in range(x_shape[1]//FLAGS.crop_size):
                r0 = r * FLAGS.crop_size
                r1 = r0 + FLAGS.crop_size
                c0 = c * FLAGS.crop_size
                c1 = c0 + FLAGS.crop_size
                x_batch[batch_id,:,:,0] = x[r0:r1,c0:c1]
                batch_position.append([r0,r1,c0,c1])
                batch_id += 1
                batch_id %= FLAGS.batch_size
                if batch_id == 0:
                    y_hat_batch = sess.run(y_hat, feed_dict={phase_train: is_train, x_ph: x_batch})
                    for i in xrange(FLAGS.batch_size):
                        r0 = batch_position[i][0]
                        r1 = batch_position[i][1]
                        c0 = batch_position[i][2]
                        c1 = batch_position[i][3]
                        y[r0:r1,c0:c1] = y_hat_batch[i,:,:,0]
                    del batch_position[:]
        
        if batch_id != 0:
            y_hat_batch = sess.run(y_hat, feed_dict={phase_train: is_train, x_ph: x_batch})
            for i in xrange(batch_id):
                r0 = batch_position[i][0]
                r1 = batch_position[i][1]
                c0 = batch_position[i][2]
                c1 = batch_position[i][3]
                y[r0:r1,c0:c1] = y_hat_batch[i,:,:,0]

        if y_gt is not None:
            l2_loss = np.sum((y - y_gt)**2) * 0.5
            print('%s: %d/%d-%s l2 loss %.3f' % (datetime.now(), file_id+1, num_files, file_name, l2_loss))
            f.write('%s: %d/%d-%s l2 loss %.3f\n' % (datetime.now(), file_id+1, num_files, file_name, l2_loss))

        # # debug
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(x, cmap=plt.cm.gray)
        # plt.subplot(122)
        # plt.imshow(y, cmap=plt.cm.gray)
        # plt.show()

        # save displacement map image result
        output_path = os.path.join(FLAGS.result_dir, file_name + '.png')
        scipy.misc.imsave(output_path, (y*255).astype(np.uint8))

        # save displacement map result
        y = (y - 0.5) * 2.0 * RANGE_MAX
        output_path = os.path.join(FLAGS.result_dir, file_name + '.mat')
        scipy.io.savemat(output_path, dict(y=y))

        duration = time.time() - start_time
        print('%s: %d/%d-%s processed (%.3f sec)' % (datetime.now(), file_id+1, num_files, file_name, duration))
        f.write('%s: %d/%d-%s processed (%.3f sec)\n' % (datetime.now(), file_id+1, num_files, file_name, duration))

    f.close()
    print('Done')


def main(_):
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('geonet'):
        working_path = os.path.join(current_path, 'geonet')
        os.chdir(working_path)
        
    # create eval directory
    if tf.gfile.Exists(FLAGS.result_dir):
        tf.gfile.DeleteRecursively(FLAGS.result_dir)
    tf.gfile.MakeDirs(FLAGS.result_dir)

    # start
    run()


if __name__ == '__main__':
    tf.app.run()