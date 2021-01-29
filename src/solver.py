# --------------------------------------------------------------------------
# Tensorflow Implementation of Synthetic Fingerprint Generation
# Licensed under The MIT License [see LICENSE for details]
# Written by Young-Mook Kang
# Email: kym343@naver.com
# --------------------------------------------------------------------------
import os
import numpy as np
import tensorflow as tf
import utils as utils


class Solver(object):
    def __init__(self, data, gen_model, flags, session, log_dir=None):
        self.data = data
        self.model = gen_model
        self.flags = flags
        self.batch_size = self.flags.batch_size
        self.sess = session
        self.log_dir = log_dir

        # Initialize saver
        self.saver = tf.compat.v1.train.Saver(max_to_keep=1)
        self._init_gen_variables()

    def _init_gen_variables(self):
        var_list = [var for var in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='pix2pix')]
        self.sess.run(tf.compat.v1.variables_initializer(var_list=var_list))

    def train(self):
        # imgs, clses, segs, irises, coordinates = self.data.train_random_batch_include_iris(batch_size=self.batch_size)
        imgs, segs = self.data.train_random_batch(batch_size=self.batch_size)

        feed = {self.model.img_tfph: imgs,
                self.model.mask_tfph: segs,
                self.model.rate_tfph: 0.5,
                self.model.trainMode: True}

        self.sess.run(self.model.dis_optim, feed_dict=feed)
        self.sess.run(self.model.gen_optim, feed_dict=feed)

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, g_loss, g_adv_loss, g_cond_loss, d_loss, summary = self.sess.run(
            [self.model.gen_optim, self.model.gen_loss, self.model.gen_adv_loss, self.model.cond_loss,
             self.model.dis_loss, self.model.summary_op], feed_dict=feed)

        return g_loss, g_adv_loss, g_cond_loss, d_loss, summary

    def generate_test_imgs(self, batch_size=20):
        print(' [*] Generate test imgs...')

        latent_all = np.zeros((self.data.num_test_imgs, *self.data.output_img_shape), dtype=np.float32)
        GT_all = np.zeros((self.data.num_test_imgs, *self.data.output_img_shape), dtype=np.float32)
        segs_all = np.zeros((self.data.num_test_imgs, *self.data.input_img_shape), dtype=np.float32)
        samples_all = np.zeros((self.data.num_test_imgs, *self.data.output_img_shape), dtype=np.float32)

        for i, index in enumerate(range(0, self.data.num_test_imgs, batch_size)):
            print('[{}/{}] generating process...'.format(i + 1, (self.data.num_test_imgs // batch_size)))

            img_tests,  seg_tests = self.data.direct_batch(batch_size=batch_size, index=index, stage='test')

            feed = {self.model.img_tfph: img_tests,
                    self.model.mask_tfph: seg_tests,
                    self.model.rate_tfph: 0.5,              # rate: 1 - keep_prob
                    self.model.trainMode: False}

            samples, latent, GT = self.sess.run([self.model.g_sample, self.model.crop_pred_latent, self.model.img_GT], feed_dict=feed)

            latent_all[index:index + img_tests.shape[0], :, :, :] = latent
            GT_all[index:index + img_tests.shape[0], :, :, :] = GT
            segs_all[index:index + img_tests.shape[0], :, :, :] = seg_tests
            samples_all[index:index + img_tests.shape[0], :, :, :] = samples

        return latent_all, segs_all, samples_all, GT_all

    def img_sample(self, iter_time, save_dir, batch_size=4):
        # imgs, clses, segs, irises, coordinates = self.data.train_random_batch_include_iris(batch_size=batch_size)
        imgs, segs = self.data.train_random_batch(batch_size=batch_size)

        feed = {self.model.mask_tfph: segs,
                self.model.img_tfph: imgs,
                self.model.rate_tfph: 0.5,                 # rate: 1 - keep_prob
                self.model.trainMode: True}

        samples, latent, GT = self.sess.run([self.model.g_sample, self.model.crop_pred_latent, self.model.img_GT], feed_dict=feed)
        utils.save_imgs(img_stores=[latent, segs, samples, GT], iter_time=iter_time, save_dir=save_dir, is_vertical=True)

    def save_model(self, logger, model_dir, iter_time):
        self.saver.save(self.sess, os.path.join(model_dir, 'model'), global_step=iter_time)
        logger.info('[*] Model saved! Iter: {}'.format(iter_time))

    def load_model(self, logger, model_dir, is_train=False):
        if is_train:
            logger.info(' [*] Reading checkpoint...')
        else:
            print(' [*] Reading checkpoint...')

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(model_dir, ckpt_name))

            meta_graph_path = ckpt.model_checkpoint_path + '.meta'
            iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

            return True, iter_time + 1
        else:
            return False, None, None
