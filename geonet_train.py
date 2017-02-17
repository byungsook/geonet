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

from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf

import geonet_model
import geonet_data

# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('log_dir', 'log/test',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           """If specified, restore this pretrained model """
                           """before beginning any training.
                           e.g. log/test/geonet.ckpt """)
tf.app.flags.DEFINE_integer('max_steps', 10, # 20000
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('decay_steps', 30000,
                            """Decay steps""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.01,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('learning_decay_factor', 0.1,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.9999,
                          """The decay to use for the moving average.""")
tf.app.flags.DEFINE_float('clip_gradients', 0.1,
                          """range for clipping gradients.""")
tf.app.flags.DEFINE_integer('max_images', 1,
                            """max # images to save.""")
tf.app.flags.DEFINE_integer('stat_steps', 10,
                            """statistics steps.""")
tf.app.flags.DEFINE_integer('summary_steps', 100,
                            """summary steps.""")
tf.app.flags.DEFINE_integer('save_steps', 5000,
                            """save steps""")
tf.app.flags.DEFINE_string('file_list', 'train_mat.txt',
                           """file_list""")
tf.app.flags.DEFINE_boolean('is_train', True,
                            """whether it is training or not""")


def train():
    """Train the network for a number of steps."""
    with tf.Graph().as_default():
        batch_manager = geonet_data.BatchManager()
        print('%s: %d files' % (datetime.now(), batch_manager.num_examples_per_epoch))

        phase_train = tf.placeholder(tf.bool, name='phase_train')

        x = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_height, FLAGS.image_width, 1])
        y = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_height, FLAGS.image_width, 1])
        w = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_height, FLAGS.image_width, 1])

        # Build a Graph that computes the logits predictions from the inference model.
        y_hat = geonet_model.inference(x, phase_train)

        # Calculate loss.
        if FLAGS.weight_on:
            loss = geonet_model.loss(y_hat, y, w)
        else:
            loss = geonet_model.loss(y_hat, y)

        ###############################################################################
        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        loss_averages_op = loss_averages.apply([loss])

        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(loss.op.name + ' (raw)', loss)
        tf.summary.scalar(loss.op.name, loss_averages.average(loss))

        # Decay the learning rate exponentially based on the number of steps.
        learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                   global_step,
                                                   FLAGS.decay_steps,
                                                   FLAGS.learning_decay_factor,
                                                   staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)
        # # or use fixed learning rate
        # learning_rate = 1e-3

        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer(learning_rate)
            grads = opt.compute_gradients(loss)
            max_grad = FLAGS.clip_gradients / learning_rate
            grads = [(tf.clip_by_value(grad, -max_grad, max_grad), var) for grad, var in grads]

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # # Add histograms for gradients.
        # for grad, var in grads:
        #     if grad is not None:
        #         tf.histogram_summary(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
            train_op = tf.no_op(name='train')


        ####################################################################
        # Start running operations on the Graph. 
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))

        # Create a saver (restorer).
        saver = tf.train.Saver()
        if FLAGS.pretrained_model_checkpoint_path:
            # assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
            saver.restore(sess, FLAGS.pretrained_model_checkpoint_path)
            print('%s: Pre-trained model restored from %s' %
                (datetime.now(), FLAGS.pretrained_model_checkpoint_path))
        else:
            sess.run(tf.global_variables_initializer(), feed_dict={phase_train: FLAGS.is_train})

        # Build the summary operation.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        summary_y_writer = tf.summary.FileWriter(FLAGS.log_dir + '/y', sess.graph)
        summary_y_hat_writer = tf.summary.FileWriter(FLAGS.log_dir + '/y_hat', sess.graph)
        summary_w_writer = tf.summary.FileWriter(FLAGS.log_dir + '/w', sess.graph)

        x_u8 = tf.placeholder(dtype=tf.uint8, shape=[None, FLAGS.image_height, FLAGS.image_width, 1])
        y_u8 = tf.placeholder(dtype=tf.uint8, shape=[None, FLAGS.image_height, FLAGS.image_width, 1])
        y_hat_u8 = tf.placeholder(dtype=tf.uint8, shape=[None, FLAGS.image_height, FLAGS.image_width, 1])
        x_summary = tf.summary.image('x', x_u8, max_outputs=FLAGS.max_images)
        y_summary = tf.summary.image('y', y_u8, max_outputs=FLAGS.max_images)
        y_hat_summary = tf.summary.image('y_hat', y_hat_u8, max_outputs=FLAGS.max_images)
        w_summary = tf.summary.image('w', w, max_outputs=FLAGS.max_images)

        ####################################################################
        # Start to train.
        print('%s: start to train' % datetime.now())
        start_step = tf.train.global_step(sess, global_step)
        for step in xrange(start_step, FLAGS.max_steps):
            # Train one step.
            start_time = time.time()
            x_batch, y_batch, w_batch = batch_manager.batch()
            if FLAGS.weight_on:
                _, loss_value = sess.run([train_op, loss], feed_dict={phase_train: FLAGS.is_train,
                                                                  x: x_batch, y: y_batch, w: w_batch})
            else:
                _, loss_value = sess.run([train_op, loss], feed_dict={phase_train: FLAGS.is_train,
                                                                  x: x_batch, y: y_batch})
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            # Print statistics periodically.
            if step % FLAGS.stat_steps == 0 or step < 100:
                examples_per_sec = FLAGS.batch_size / float(duration)
                print('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)' % 
                    (datetime.now(), step, loss_value, examples_per_sec, duration))

            # Write the summary periodically.
            if step % FLAGS.summary_steps == 0 or step < 100:
                new_shape = [FLAGS.max_images, FLAGS.image_height, FLAGS.image_width, 1]
                x_ = x_batch[:FLAGS.max_images,:]
                x_ = np.reshape(x_*255.0, new_shape).astype(np.uint8)
                y_ = y_batch[:FLAGS.max_images,:]
                y_ = np.reshape(y_*255.0, new_shape).astype(np.uint8)
                y_hat_ = sess.run(tf.cast(tf.multiply(y_hat, 255.0), tf.uint8),
                    feed_dict={phase_train: FLAGS.is_train, x: x_batch, y: y_batch})

                summary_str, x_summary_str, y_summary_str, y_hat_summary_str, w_summary_str = sess.run(
                    [summary_op, x_summary, y_summary, y_hat_summary, w_summary],
                    feed_dict={phase_train: FLAGS.is_train, x: x_batch, y: y_batch, w: w_batch,
                               x_u8: x_, y_u8: y_, y_hat_u8: y_hat_})
                summary_writer.add_summary(summary_str, step)
                
                x_summary_tmp = tf.Summary()
                y_summary_tmp = tf.Summary()
                y_hat_summary_tmp = tf.Summary()
                x_summary_tmp.ParseFromString(x_summary_str)
                y_summary_tmp.ParseFromString(y_summary_str)
                y_hat_summary_tmp.ParseFromString(y_hat_summary_str)
                for i in xrange(FLAGS.max_images):
                    new_tag = '%06d/%02d' % (step, i)
                    x_summary_tmp.value[i].tag = new_tag
                    y_summary_tmp.value[i].tag = new_tag
                    y_hat_summary_tmp.value[i].tag = new_tag
        
                summary_writer.add_summary(x_summary_tmp, step)
                summary_y_writer.add_summary(y_summary_tmp, step)
                summary_y_hat_writer.add_summary(y_hat_summary_tmp, step)

                if FLAGS.weight_on:
                    w_summary_tmp = tf.Summary()
                    w_summary_tmp.ParseFromString(w_summary_str)
                    for i in xrange(FLAGS.max_images):
                        w_summary_tmp.value[i].tag = new_tag
                    summary_w_writer.add_summary(w_summary_tmp, step)

            # Save the model checkpoint periodically.
            if (step + 1) % FLAGS.save_steps == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.log_dir, 'geonet.ckpt')
                saver.save(sess, checkpoint_path)

        print('done')


def main(_):
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('geonet'):
        working_path = os.path.join(current_path, 'geonet')
        os.chdir(working_path)

    # create log directory
    if FLAGS.log_dir.endswith('log'):
        FLAGS.log_dir = os.path.join(FLAGS.log_dir, datetime.now().isoformat().replace(':', '-'))
    elif tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    train()


if __name__ == '__main__':
    tf.app.run()
