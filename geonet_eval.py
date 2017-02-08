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
import tensorflow as tf

import geonet_model
import geonet_data

# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', 'eval/test',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', 'log/test/geonet.ckpt',
                           """If specified, restore this pretrained model.""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.9999,
                          """The decay to use for the moving average.""")
tf.app.flags.DEFINE_integer('max_images', 8,
                            """max # images to save.""")
tf.app.flags.DEFINE_string('file_list', 'test.txt',
                           """file_list""")
tf.app.flags.DEFINE_integer('num_epoch', 1, # 10
                            """# epoch""")


def evaluate():
    with tf.Graph().as_default() as g:
        batch_manager = geonet_data.BatchManager()
        print('%s: %d files' % (datetime.now(), batch_manager.num_examples_per_epoch))

        global_step = tf.Variable(0, name='global_step', trainable=False)
        is_train = True
        phase_train = tf.placeholder(tf.bool, name='phase_train')

        x = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_height, FLAGS.image_width, 1])
        y = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_height, FLAGS.image_width, 1])
        
        # Build a Graph that computes the logits predictions from the inference model.
        y_hat = geonet_model.inference(x, phase_train)

        # Calculate loss.
        loss = geonet_model.loss(y_hat, y)

        # # Restore the moving average version of the learned variables for eval.
        # variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)
        # variables_to_restore = variable_averages.variables_to_restore()
        # saver = tf.train.Saver(variables_to_restore)
        saver = tf.train.Saver()


        # Build the summary writer
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        loss_ph = tf.placeholder(tf.float32)
        loss_summary = tf.summary.scalar('l2 loss (raw)', loss_ph)

        loss_avg = tf.placeholder(tf.float32)
        loss_avg_summary = tf.summary.scalar('l2 loss (avg)', loss_avg)

        summary_y_writer = tf.summary.FileWriter(FLAGS.eval_dir + '/y', g)
        summary_y_hat_writer = tf.summary.FileWriter(FLAGS.eval_dir + '/y_hat', g)

        x_u8 = tf.placeholder(dtype=tf.uint8, shape=[None, FLAGS.image_height, FLAGS.image_width, 1])
        y_u8 = tf.placeholder(dtype=tf.uint8, shape=[None, FLAGS.image_height, FLAGS.image_width, 1])
        y_hat_u8 = tf.placeholder(dtype=tf.uint8, shape=[None, FLAGS.image_height, FLAGS.image_width, 1])
        x_summary = tf.summary.image('x', x_u8, max_outputs=FLAGS.max_images)
        y_summary = tf.summary.image('y', y_u8, max_outputs=FLAGS.max_images)
        y_hat_summary = tf.summary.image('y_hat', y_hat_u8, max_outputs=FLAGS.max_images)

        # Start evaluation
        with tf.Session() as sess:
            if FLAGS.pretrained_model_checkpoint_path:
                # assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
                saver.restore(sess, FLAGS.pretrained_model_checkpoint_path)
                print('%s: Pre-trained model restored from %s' %
                    (datetime.now(), FLAGS.pretrained_model_checkpoint_path))

            num_eval = batch_manager.num_examples_per_epoch * FLAGS.num_epoch
            num_iter = int(math.ceil(num_eval / FLAGS.batch_size))
            print('total iter: %d' % num_iter)
            total_loss = 0
            for step in range(num_iter):
                start_time = time.time()
                x_batch, y_batch = batch_manager.batch()
                y_hat_, loss_value = sess.run([tf.cast(tf.multiply(y_hat, 255.0), tf.uint8), loss], 
                                                   feed_dict={phase_train: is_train, x: x_batch, y: y_batch})

                new_shape = [FLAGS.max_images, FLAGS.image_height, FLAGS.image_width, 1]
                x_ = x_batch[:FLAGS.max_images,:]
                x_ = np.reshape(x_*255.0, new_shape).astype(np.uint8)
                y_ = y_batch[:FLAGS.max_images,:]
                y_ = np.reshape(y_*255.0, new_shape).astype(np.uint8)

                total_loss += loss_value
                duration = time.time() - start_time
                examples_per_sec = FLAGS.batch_size / float(duration)
                print('%s: epoch %d, step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)' % (
                        datetime.now(), batch_manager.num_epoch, step, loss_value, examples_per_sec, duration))

                loss_summary_str, x_summary_str, y_summary_str, y_hat_summary_str = sess.run(
                    [loss_summary, x_summary, y_summary, y_hat_summary],
                    feed_dict={loss_ph: loss_value, x_u8: x_, y_u8: y_, y_hat_u8: y_hat_})
                summary_writer.add_summary(loss_summary_str, step)

                x_summary_tmp = tf.Summary()
                y_summary_tmp = tf.Summary()
                y_hat_summary_tmp = tf.Summary()
                x_summary_tmp.ParseFromString(x_summary_str)
                y_summary_tmp.ParseFromString(y_summary_str)
                y_hat_summary_tmp.ParseFromString(y_hat_summary_str)
                for i in xrange(FLAGS.max_images):
                    new_tag = '%06d/%03d' % (step, i)
                    x_summary_tmp.value[i].tag = new_tag
                    y_summary_tmp.value[i].tag = new_tag
                    y_hat_summary_tmp.value[i].tag = new_tag

                summary_writer.add_summary(x_summary_tmp, step)
                summary_y_writer.add_summary(y_summary_tmp, step)
                summary_y_hat_writer.add_summary(y_hat_summary_tmp, step)

            # Compute precision
            loss_avg_ = total_loss / num_iter
            print('%s: loss avg = %.3f' % (datetime.now(), loss_avg_))

            loss_avg_summary_str = sess.run(loss_avg_summary, feed_dict={loss_avg: loss_avg_})
            g_step = tf.train.global_step(sess, global_step)
            summary_writer.add_summary(loss_avg_summary_str, g_step)

    print('done')


def main(_):
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('geonet'):
        working_path = os.path.join(current_path, 'geonet')
        os.chdir(working_path)
        
    # create eval directory
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)

    # start evaluation
    evaluate()

if __name__ == '__main__':
    tf.app.run()
