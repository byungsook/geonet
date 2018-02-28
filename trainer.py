from __future__ import print_function

import os
import numpy as np
from tqdm import trange
import scipy.io

from models import *
from utils import save_image

class Trainer(object):
    def __init__(self, config, batch_manager):
        tf.set_random_seed(config.random_seed)
        self.config = config
        self.batch_manager = batch_manager
        self.model = config.model
        self.x, self.y = batch_manager.batch()
        self.xt = tf.placeholder(tf.float32, shape=batch_manager.img_dim)
        self.yt = tf.placeholder(tf.float32, shape=batch_manager.img_dim)
        self.dataset = config.dataset

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size

        self.lr = tf.Variable(config.lr, name='lr')
        self.lr_update = tf.assign(self.lr, tf.maximum(self.lr*0.1, config.lr_lower_boundary), name='lr_update')

        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.b_num = config.batch_size
        self.filters = config.filters
        self.repeat_num = config.repeat_num

        self.model_dir = config.model_dir

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format
        if self.data_format == 'NCHW':
            self.x = nhwc_to_nchw(self.x)
            self.y = nhwc_to_nchw(self.y)
            self.xt = nhwc_to_nchw(self.xt)
            self.yt = nhwc_to_nchw(self.yt)

        self.start_step = config.start_step
        self.log_step = config.log_step
        self.test_step = config.test_step
        self.max_step = config.max_step
        self.save_sec = config.save_sec
        self.lr_update_step = config.lr_update_step

        self.step = tf.Variable(self.start_step, name='step', trainable=False)

        self.is_train = config.is_train
        self.build_model()

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=self.save_sec,
                                global_step=self.step,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)
        if self.is_train:
            self.batch_manager.start_thread(self.sess)

    def build_model(self):
        if self.model == 'VDSR':
            self.y_, self.var = VDSR(
                    self.x, self.filters, self.repeat_num, self.data_format)                
            self.yt_, _ = VDSR(
                    self.xt, self.filters, self.repeat_num, self.data_format,
                    train=False, reuse=True)        
        else:
            self.y_, self.var = DnCNN(
                    self.x, self.filters, self.repeat_num, self.data_format)                
            self.yt_, _ = DnCNN(
                    self.xt, self.filters, self.repeat_num, self.data_format,
                    train=False, reuse=True)

        self.y_img = denorm_img(self.y_, self.data_format) # for debug
        self.yt_img = denorm_img(self.yt_, self.data_format)

        show_all_variables()        

        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(self.config.optimizer))

        optimizer = optimizer(self.lr, beta1=self.beta1, beta2=self.beta2)

        # losses
        self.loss = tf.reduce_mean(tf.squared_difference(self.y_, self.y))

        # test loss
        self.tl = tf.reduce_mean(tf.squared_difference(self.yt_, self.yt))
        self.test_loss = tf.placeholder(tf.float32)

        self.optim = optimizer.minimize(self.loss, global_step=self.step, var_list=self.var)
 
        summary = [
            tf.summary.image("x", denorm_img(self.x, self.data_format)),
            tf.summary.image("y_gt", denorm_img(self.y, self.data_format)),
            tf.summary.image("y", self.y_img),

            tf.summary.scalar("loss/loss", self.loss),
           
            tf.summary.scalar("misc/lr", self.lr),
            tf.summary.scalar('misc/q', self.batch_manager.q.size())
        ]

        self.summary_op = tf.summary.merge(summary)

        summary = [
            tf.summary.scalar("loss/test_loss", self.test_loss),
        ]

        self.summary_test = tf.summary.merge(summary)

    def train(self):
        y_list = []
        for _, y in self.batch_manager.test_batch():
            y_list.append(y[0]*255)
        y_list = np.array(y_list)
        save_image(y_list, '{}/y_gt.png'.format(self.model_dir), nrow=4)
        
        for step in trange(self.start_step, self.max_step):
            fetch_dict = {
                "optim": self.optim,
                "loss": self.loss,
            }           

            if step % self.log_step == 0 or step == self.max_step-1:
                fetch_dict.update({
                    "summary": self.summary_op,                    
                })

            if step % self.test_step == self.test_step-1 or step == self.max_step-1:
                tl_ = 0.0
                y_list = []
                for i, (x, y) in enumerate(self.batch_manager.test_batch()):
                    if self.data_format == 'NCHW':
                        x = to_nchw_numpy(x)
                        y = to_nchw_numpy(y)
                    tl, yt_ = self.sess.run([self.tl, self.yt_], {self.xt: x, self.yt: y})
                    ymat = (yt_ - 0.5) * 2.0 * self.batch_manager.range_max
                    yt_path = os.path.join(self.model_dir, '{}.mat'.format(i))
                    scipy.io.savemat(yt_path, dict(y=ymat))

                    ymat_minmax = [np.amin(ymat),np.amax(ymat)]
                    minmax_path = os.path.join(self.model_dir, 'minmax{}.txt'.format(i))
                    with open(minmax_path, 'w') as f:
                        f.write(ymat_minmax[0] + ' ' + ymat_minmax[1])
                    print('test loss: {}, min: {}, max: {}'.format(tl,ymat_minmax[0], ymat_minmax[1]))
                    tl_ += tl
                    
                    yt_ = to_nhwc_numpy(yt_)                
                    y_list.append(yt_[0]*255)

                tl_ /= len(self.batch_manager.paths['test'])
                print('avg test loss:', tl_)
                
                summary_test = self.sess.run(self.summary_test, {self.test_loss: tl_})
                self.summary_writer.add_summary(summary_test, step)
                self.summary_writer.flush()

                y_list = np.array(y_list)
                save_image(y_list, '{}/y_{}.png'.format(self.model_dir, step), nrow=4)

            result = self.sess.run(fetch_dict)

            if step % self.log_step == 0 or step == self.max_step-1:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

                loss = result['loss']
                assert not np.isnan(loss), 'Model diverged with loss = NaN'

                print("\n[{}/{}] Loss: {:.6f}".format(step, self.max_step, loss))

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run(self.lr_update)

        # save last checkpoint..
        save_path = os.path.join(self.model_dir, 'model.ckpt')
        self.saver.save(self.sess, save_path, global_step=self.step)
        self.batch_manager.stop_thread()

    def test(self):
        pass