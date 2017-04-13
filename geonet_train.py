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
import pprint

from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf

import geonet_model
import geonet_data

# parameters
flags = tf.app.flags
flags.DEFINE_string('log_dir', 'log/test',
                    """Directory where to write event logs """
                    """and checkpoint.""")
flags.DEFINE_boolean('log_device_placement', False,
                     """Whether to log device placement.""")
flags.DEFINE_string('checkpoint_dir', '',
                    """If specified, restore this pretrained model """
                    """before beginning any training.
                    e.g. log/test""")
flags.DEFINE_integer('model', 1, # [1-2]
                     """type of training model.""")
flags.DEFINE_integer('max_steps', 10, # 50000
                     """Number of batches to run.""")
flags.DEFINE_integer('decay_steps', 30000,
                     """Decay steps""")
flags.DEFINE_float('initial_learning_rate', 0.01,
                   """Initial learning rate.""")
flags.DEFINE_float('learning_decay_factor', 0.1,
                   """Learning rate decay factor.""")
flags.DEFINE_float('moving_avg_decay', 0.9999,
                   """The decay to use for the moving average.""")
flags.DEFINE_float('clip_gradients', 0.1,
                   """range for clipping gradients.""")
flags.DEFINE_integer('max_images', 1,
                     """max # images to save.""")
flags.DEFINE_integer('stat_steps', 10,
                     """statistics steps.""")
flags.DEFINE_integer('summary_steps', 100,
                     """summary steps.""")
flags.DEFINE_integer('save_steps', 5000,
                     """save steps""")
flags.DEFINE_string('file_list', 'train.txt',
                    """file_list""")
flags.DEFINE_boolean('is_train', True,
                     """whether it is training or not""")
# flags.DEFINE_integer('gpu_id', 0,
#                      """gpu id""")
FLAGS = flags.FLAGS


def train():
    """Train the network for a number of steps."""
    with tf.Graph().as_default():
    # with tf.Graph().as_default(), tf.device('/cpu:0'):
        # print flags
        flag_file_path = os.path.join(FLAGS.log_dir, 'flag.txt')
        with open(flag_file_path, 'wt') as out:
            pprint.PrettyPrinter(stream=out).pprint(flags.FLAGS.__flags)

        ###############################################################################
        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        
        # Decay the learning rate exponentially based on the number of steps.
        learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                   global_step,
                                                   FLAGS.decay_steps,
                                                   FLAGS.learning_decay_factor,
                                                   staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)
        # # or use fixed learning rate
        # learning_rate = 1e-3

        #####################################
        # gpu start
        # with tf.device('/gpu:%d' % FLAGS.gpu_id):
        batch_manager = geonet_data.BatchManager()
        print('%s: %d files' % (datetime.now(), batch_manager.num_examples_per_epoch))

        x, y = batch_manager.batch()

        # Build a Graph
        y_hat = geonet_model.inference(x, FLAGS.image_width, FLAGS.is_train, model=FLAGS.model)

        # Calculate loss.
        loss = geonet_model.loss(y_hat, y)

        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        loss_averages_op = loss_averages.apply([loss])

        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(loss.op.name + ' (raw)', loss)
        tf.summary.scalar(loss.op.name, loss_averages.average(loss))

        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer(learning_rate)
            grads = opt.compute_gradients(loss)

        # gradient clipping
        max_grad = FLAGS.clip_gradients / learning_rate
        grads = [(tf.clip_by_value(grad, -max_grad, max_grad), var) for grad, var in grads]
        # gpu end
        #####################################

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # # Add histograms for gradients.
        # for grad, var in grads:
        #     if grad is not None:
        #         tf.histogram_summary(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)


        ####################################################################
        # Start running operations on the Graph.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = FLAGS.log_device_placement
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
            sess.run(tf.global_variables_initializer())

        # Build the summary operation.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        summary_y_writer = tf.summary.FileWriter(FLAGS.log_dir + '/y', sess.graph)
        summary_y_hat_writer = tf.summary.FileWriter(FLAGS.log_dir + '/y_hat', sess.graph)

        x_u8 = tf.placeholder(dtype=tf.uint8, shape=[None, FLAGS.image_height, FLAGS.image_width, 1])
        y_u8 = tf.placeholder(dtype=tf.uint8, shape=[None, FLAGS.image_height, FLAGS.image_width, 1])
        y_hat_u8 = tf.placeholder(dtype=tf.uint8, shape=[None, FLAGS.image_height, FLAGS.image_width, 1])
        x_summary = tf.summary.image('x', x_u8, max_outputs=FLAGS.max_images)
        y_summary = tf.summary.image('y', y_u8, max_outputs=FLAGS.max_images)
        y_hat_summary = tf.summary.image('y_hat', y_hat_u8, max_outputs=FLAGS.max_images)

        ####################################################################
        # Start to train.
        print('%s: start to train' % datetime.now())
        batch_manager.start_thread(sess)
        start_step = tf.train.global_step(sess, global_step)
        for step in xrange(start_step, FLAGS.max_steps):
            # Train one step.
            start_time = time.time()
            _, loss_value, x_batch, y_batch = sess.run([train_op, loss, x, y])
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
                y_ = y_batch[:FLAGS.max_images,:]
                y_hat_ = sess.run(y_hat, feed_dict={x: x_batch, y: y_batch})
                
                if FLAGS.model == 1:
                    x_ = np.reshape((x_+1.0)*127.5, new_shape).astype(np.uint8)
                    y_ = np.reshape((y_+1.0)*127.5, new_shape).astype(np.uint8)
                    y_hat_ = ((y_hat_+1.0)*127.5).astype(np.uint8)
                else:
                    x_ = np.reshape(x_*255.0, new_shape).astype(np.uint8)
                    y_ = np.reshape(y_*255.0, new_shape).astype(np.uint8)
                    y_hat_ = (y_hat_*255.0).astype(np.uint8)
    
                summary_str, x_summary_str, y_summary_str, y_hat_summary_str = sess.run(
                    [summary_op, x_summary, y_summary, y_hat_summary],
                    feed_dict={x: x_batch, y: y_batch, 
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

            # Save the model checkpoint periodically.
            if (step + 1) % FLAGS.save_steps == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.log_dir, 'geonet.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step)

        batch_manager.stop_thread()
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
