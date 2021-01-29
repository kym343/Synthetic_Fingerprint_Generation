# --------------------------------------------------------------------------
# Tensorflow Implementation of Synthetic Fingerprint Generation
# Licensed under The MIT License [see LICENSE for details]
# Written by Young-Mook Kang
# Email: kym343@naver.com
# --------------------------------------------------------------------------
import os
import logging
import numpy as np
import tensorflow as tf

import utils as utils
import tensorflow_utils as tf_utils


class Pix2pix(object):
    def __init__(self, input_img_shape=(320, 280, 1), session=None, lr=2e-4, total_iters=2e5, is_train=True,
                 log_dir=None, lambda_1=1000., lambda_2=10., lambda_3=10., num_class=3, num_identities=1000, batch_size=4,
                 use_batch_norm=True, name='pix2pix'):
        self.input_img_shape = input_img_shape
        self.output_img_shape = (*self.input_img_shape[0:2], 1)

        self.sess = session
        self.is_train = is_train
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.num_class = num_class
        self.num_identities = num_identities

        self.gen_c = [16, 32, 32, 32, 32, 32, 32, 32, 64, 64,
                      32, 32, 32, 32, 32, 32, 32, 32, 32, 16,
                      16, 16, 16, self.output_img_shape[2]]
        # self.gen_c = [64, 128, 256, 512, 512, 512, 512, 512,
        #               512, 512, 512, 512, 256, 128, 64, self.output_img_shape[2]]
        self.dis_c = [64, 128, 256, 512, 1]
        # self.dis_c = [32, 64, 128, 256, 512, 1]

        self.lr = lr
        self.total_steps = total_iters
        self.start_decay_step = int(self.total_steps * 0.5)
        self.decay_steps = self.total_steps - self.start_decay_step
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.use_batch_norm = use_batch_norm
        self.name = name
        self.tb_lr = None
        self.crop_pred_latent = None
        self.crop_latent_value = tf.reshape(tf.constant([255.], dtype=tf.float32), shape=(1, 1, 1, 1))
        self.crop_generate_value = tf.reshape(tf.constant([204.], dtype=tf.float32), shape=(1, 1, 1, 1))
        self.crop_backgound_value = tf.reshape(tf.constant([102.], dtype=tf.float32), shape=(1, 1, 1, 1))

        self.logger = logging.getLogger(__name__)  # logger
        self.logger.setLevel(logging.INFO)
        utils.init_logger(logger=self.logger, log_dir=self.log_dir, is_train=self.is_train, name=self.name)

        self._build_graph()         # main graph
        # self._best_metrics_record()
        self._init_tensorboard()    # tensorboard
        tf_utils.show_all_variables(logger=self.logger if self.is_train else None)

    def _build_graph(self):
        self.top_scope = tf.get_variable_scope()  # top-level scope

        with tf.compat.v1.variable_scope(self.name):
            self.mask_tfph = tf.compat.v1.placeholder(tf.float32, shape=[None, *self.input_img_shape], name='mask_tfph')
            self.img_tfph = tf.compat.v1.placeholder(tf.float32, shape=[None, *self.output_img_shape], name='img_tfph')
            # self.coord_tfph = tf.compat.v1.placeholder(tf.int32, shape=[None, 4], name='coord_tfph')# (x, y, h, w)
            self.rate_tfph = tf.compat.v1.placeholder(tf.float32, name='keep_prob_ph')
            # self.cls_tfph = tf.compat.v1.placeholder(dtype=tf.dtypes.int64, shape=[None, 1])
            self.trainMode = tf.compat.v1.placeholder(tf.bool, name='train_mode_ph')

            # Initialize generator & discriminator
            self.gen_obj = Generator(name='G', gen_c=self.gen_c, norm='instance', use_batch_norm=self.use_batch_norm,
                                     trainMode=self.trainMode, logger=self.logger, _ops=None)
            self.dis_obj = Discriminator(name='D', dis_c=self.dis_c, norm='instance',
                                         logger=self.logger,  _ops=None)

            # Transform img_train and seg_img_train
            self.img_GT = self.mask_backgound_GT(self.mask_tfph, self.img_tfph)

            input_mask = self.transform_img(self.mask_tfph)
            real_img = self.transform_img(self.img_GT)
            mask_latent = self.mask_latent_concat(self.mask_tfph, self.img_tfph)

            # Generator
            fake_img = self.gen_obj(mask_latent, self.rate_tfph)

            # Concatenation
            self.g_sample = self.inv_transform_img(fake_img)
            self.real_pair = tf.concat([input_mask, real_img], axis=3)
            self.fake_pair = tf.concat([input_mask, fake_img], axis=3)

            # Define generator loss
            self.gen_adv_loss = self.generator_loss(self.dis_obj, self.fake_pair)
            self.cond_loss = self.conditional_loss(pred_img=fake_img, gt=real_img, mask=self.mask_tfph)
            self.gen_loss = self.gen_adv_loss + self.cond_loss

            # Define discriminator loss
            self.dis_loss = self.discriminator_loss(self.dis_obj, self.real_pair, self.fake_pair)

            # Optimizers
            self.gen_optim = self.init_optimizer(loss=self.gen_loss, variables=self.gen_obj.variables, name='Adam_gen')
            self.dis_optim = self.init_optimizer(loss=self.dis_loss, variables=self.dis_obj.variables, name='Adam_dis')

    def mask_latent_concat(self, mask_tfph, img_tfph):
        # self.latent_mask = np.zeros_like(img_tfph, dtype=np.float32)
        # self.latent_mask[mask_tfph == [102, 255, 255]] = 1

        # latent_mask = tf.zeros_like(img_tfph, dtype=tf.float32)
        # latent_mask[tf.math.equal(mask_tfph, self.crop_latent_value)] = 1

        latent_mask = self.extract_latent_region(mask_tfph)
        # latent = img_tfph * latent_mask
        backgound = 255*tf.ones_like(latent_mask, dtype=tf.float32) - 255*latent_mask

        self.crop_pred_latent = img_tfph * latent_mask + backgound
        latent_img = tf.concat([mask_tfph, self.crop_pred_latent], axis=-1)

        return latent_img

    def mask_backgound_GT(self, mask_tfph, img_tfph):
        backgound_mask = self.extract_backgound_region(mask_tfph)
        fingerprint_mask = tf.ones_like(backgound_mask, dtype=tf.float32) - backgound_mask

        img_GT = img_tfph * fingerprint_mask + 255 * backgound_mask

        return img_GT

    def extract_latent_region(self, seg_mask):
        mask = tf.math.reduce_sum(tf.dtypes.cast(tf.math.equal(seg_mask, self.crop_latent_value), dtype=tf.float32),
                                  axis=-1, keepdims=True) / 2
        return mask

    def extract_generate_region(self, seg_mask):
        mask = tf.math.reduce_sum(tf.dtypes.cast(tf.math.equal(seg_mask, self.crop_generate_value), dtype=tf.float32),
                                  axis=-1, keepdims=True)
        return mask

    def extract_backgound_region(self, seg_mask):
        mask = tf.math.reduce_sum(tf.dtypes.cast(tf.math.equal(seg_mask, self.crop_backgound_value), dtype=tf.float32),
                                  axis=-1, keepdims=True)
        mask = mask - tf.ones_like(mask, dtype=tf.float32)
        return mask

    def _best_metrics_record(self):
        self.best_acc_tfph = tf.compat.v1.placeholder(tf.float32, name='best_acc')

        # Best accuracy variable
        self.best_acc = tf.compat.v1.get_variable(name='best_acc', dtype=tf.float32, initializer=tf.constant(0.),
                                                  trainable=False)
        self.assign_best_acc = tf.compat.v1.assign(self.best_acc, value=self.best_acc_tfph)

    def init_optimizer(self, loss, variables, name='Adam'):
        with tf.compat.v1.variable_scope(name):
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.lr
            end_leanring_rate = 0.
            start_decay_step = self.start_decay_step
            decay_steps = self.decay_steps

            learning_rate = (tf.where(tf.greater_equal(global_step, start_decay_step),
                                      tf.compat.v1.train.polynomial_decay(starter_learning_rate,
                                                                          global_step - start_decay_step,
                                                                          decay_steps, end_leanring_rate, power=1.0),
                                      starter_learning_rate))

            self.tb_lr = tf.compat.v1.summary.scalar('learning_rate', learning_rate)
            learn_step = tf.compat.v1.train.AdamOptimizer(learning_rate, beta1=0.5).\
                minimize(loss, global_step=global_step, var_list=variables)

            return learn_step

    @staticmethod
    def generator_loss(dis_obj, fake_img):
        d_logit_fake = dis_obj(fake_img)
        loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake ,
                                                                           labels=tf.ones_like(d_logit_fake)))
        return loss

    def conditional_loss(self, pred_img, gt, mask=None):
        loss = tf.constant(0., dtype=tf.float32)

        generate_mask = self.extract_generate_region(mask)
        latent_mask = self.extract_latent_region(mask)
        background_mask = self.extract_latent_region(mask)

        # cond_loss = tf.math.reduce_mean(tf.math.abs(pred_img - gt))
        # loss = self.labmda_1 * cond_loss
        cond_generate = tf.math.reduce_mean(tf.math.abs(generate_mask * pred_img - generate_mask * gt))
        cond_latent = tf.math.reduce_mean(tf.math.abs(latent_mask*pred_img - latent_mask*gt))
        cond_background = tf.math.reduce_mean(tf.math.abs(background_mask*pred_img - background_mask*gt))

        loss = self.lambda_1 * cond_generate + self.lambda_2 * cond_latent + self.lambda_3 * cond_background
        # ==============================================================================================================
        return loss

    @staticmethod
    def discriminator_loss(dis_obj, real_img, fake_img):
        d_logit_real = dis_obj(real_img)
        d_logit_fake = dis_obj(fake_img)

        error_real = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_real, labels=tf.ones_like(d_logit_real)))
        error_fake = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake)))

        loss = 0.5 * (error_real + error_fake)
        return loss

    def _init_tensorboard(self):
        self.tb_gen_loss = tf.compat.v1.summary.scalar('loss/G_loss', self.gen_loss)
        self.tb_adv_loss = tf.compat.v1.summary.scalar('loss/adv_loss', self.gen_adv_loss)
        self.tb_cond_loss = tf.compat.v1.summary.scalar('loss/cond_loss', self.cond_loss)
        self.tb_dis_loss = tf.compat.v1.summary.scalar('loss/D_lss', self.dis_loss)
        self.summary_op = tf.compat.v1.summary.merge(
            inputs=[self.tb_gen_loss, self.tb_adv_loss, self.tb_cond_loss, self.tb_dis_loss, self.tb_lr])

    @staticmethod
    def transform_img(img):
        return img / 127.5 - 1.

    @staticmethod
    def inv_transform_img(img):
        img = (img + 1.) * 127.5
        return img

    def convert_one_hot(self, data):
        data = tf.dtypes.cast(data, dtype=tf.uint8)
        data = tf.one_hot(data, depth=self.num_identities, name='one_hot')
        data = tf.reshape(data, shape=[-1, self.num_identities])
        return data


class Generator(object):
    def __init__(self, name=None, gen_c=None, norm='instance', use_batch_norm=True, trainMode=True, logger=None, _ops=None):
        self.name = name
        self.gen_c = gen_c
        self.norm = norm
        self.logger = logger
        self._ops = _ops
        self.reuse = False
        self.use_batch_norm = use_batch_norm
        self.trainMode = trainMode

    # Multiple code vector
    def __call__(self, x, keep_rate=0.5, cls_code=None):
        with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x, logger=self.logger)

            # E0: (320, 280) -> (320, 280)
            s0_conv1 = tf_utils.conv2d(x, output_dim=self.gen_c[0], k_h=3, k_w=3, d_h=1, d_w=1,
                                       initializer='He', logger=self.logger, name='s0_conv1')
            s0_conv1 = tf_utils.relu(s0_conv1, logger=self.logger, name='relu_s0_conv1')

            s0_conv2 = tf_utils.conv2d(x=s0_conv1, output_dim=2 * self.gen_c[0], k_h=3, k_w=3, d_h=1, d_w=1,
                                       initializer='He', logger=self.logger, name='s0_conv2')
            if self.use_batch_norm:
                s0_conv2 = tf_utils.norm(s0_conv2, name='s0_norm1', _type=self.norm, _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s0_conv2 = tf_utils.relu(s0_conv2, name='relu_s0_conv2', logger=self.logger)

            # E1: (320, 280) -> (160, 140)
            s1_maxpool = tf_utils.max_pool(x=s0_conv2, name='s1_maxpool2d', logger=self.logger)

            s1_conv1 = tf_utils.conv2d(x=s1_maxpool, output_dim=self.gen_c[0], k_h=3, k_w=3, d_h=1, d_w=1,
                                       initializer='He', name='s1_conv1', logger=self.logger)
            if self.use_batch_norm:
                s1_conv1 = tf_utils.norm(s1_conv1, name='s1_norm0', _type=self.norm, _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s1_conv1 = tf_utils.relu(s1_conv1, name='relu_s1_conv1', logger=self.logger)

            s1_conv2 = tf_utils.conv2d(x=s1_conv1, output_dim=self.gen_c[1], k_h=3, k_w=3, d_h=1, d_w=1,
                                       initializer='He', name='s1_conv2', logger=self.logger)
            if self.use_batch_norm:
                s1_conv2 = tf_utils.norm(s1_conv2, name='s1_norm1', _type=self.norm, _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s1_conv2 = tf_utils.relu(s1_conv2, name='relu_s1_conv2', logger=self.logger)

            # E2: (160, 140) -> (80, 70)
            s2_maxpool = tf_utils.max_pool(x=s1_conv2, name='s2_maxpool2d', logger=self.logger)
            s2_conv1 = tf_utils.conv2d(x=s2_maxpool, output_dim=self.gen_c[2], k_h=3, k_w=3, d_h=1, d_w=1,
                                       initializer='He', name='s2_conv1', logger=self.logger)
            if self.use_batch_norm:
                s2_conv1 = tf_utils.norm(s2_conv1, name='s2_norm0', _type=self.norm, _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s2_conv1 = tf_utils.relu(s2_conv1, name='relu_s2_conv1', logger=self.logger)

            s2_conv2 = tf_utils.conv2d(x=s2_conv1, output_dim=self.gen_c[3], k_h=3, k_w=3, d_h=1, d_w=1,
                                       initializer='He', name='s2_conv2', logger=self.logger)
            if self.use_batch_norm:
                s2_conv2 = tf_utils.norm(s2_conv2, name='s2_norm1', _type=self.norm, _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s2_conv2 = tf_utils.relu(s2_conv2, name='relu_s2_conv2', logger=self.logger)

            # E3: (80, 70) -> (40, 35)
            s3_maxpool = tf_utils.max_pool(x=s2_conv2, name='s3_maxpool2d', logger=self.logger)
            s3_conv1 = tf_utils.conv2d(x=s3_maxpool, output_dim=self.gen_c[4], k_h=3, k_w=3, d_h=1, d_w=1,
                                       initializer='He', name='s3_conv1', logger=self.logger)
            if self.use_batch_norm:
                s3_conv1 = tf_utils.norm(s3_conv1, name='s3_norm0', _type=self.norm, _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s3_conv1 = tf_utils.relu(s3_conv1, name='relu_s3_conv1', logger=self.logger)

            s3_conv2 = tf_utils.conv2d(x=s3_conv1, output_dim=self.gen_c[5], k_h=3, k_w=3, d_h=1, d_w=1,
                                       initializer='He', name='s3_conv2', logger=self.logger)
            if self.use_batch_norm:
                s3_conv2 = tf_utils.norm(s3_conv2, name='s3_norm1', _type=self.norm, _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s3_conv2 = tf_utils.relu(s3_conv2, name='relu_s3_conv2', logger=self.logger)

            # E4: (40, 35) -> (20, 18)
            s4_maxpool = tf_utils.max_pool(x=s3_conv2, name='s4_maxpool2d', logger=self.logger)
            s4_conv1 = tf_utils.conv2d(x=s4_maxpool, output_dim=self.gen_c[6], k_h=3, k_w=3, d_h=1, d_w=1,
                                       initializer='He', name='s4_conv1', logger=self.logger)
            if self.use_batch_norm:
                s4_conv1 = tf_utils.norm(s4_conv1, name='s4_norm0', _type=self.norm, _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s4_conv1 = tf_utils.relu(s4_conv1, name='relu_s4_conv1', logger=self.logger)

            s4_conv2 = tf_utils.conv2d(x=s4_conv1, output_dim=self.gen_c[7], k_h=3, k_w=3, d_h=1, d_w=1,
                                       initializer='He', name='s4_conv2', logger=self.logger)
            if self.use_batch_norm:
                s4_conv2 = tf_utils.norm(s4_conv2, name='s4_norm1', _type=self.norm, _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s4_conv2 = tf_utils.relu(s4_conv2, name='relu_s4_conv2', logger=self.logger)
            s4_conv2_drop = tf_utils.dropout(x=s4_conv2, keep_prob=keep_rate, name='s4_dropout',
                                             logger=self.logger)

            # E5: (20, 18) -> (10, 9)
            s5_maxpool = tf_utils.max_pool(x=s4_conv2_drop, name='s5_maxpool2d', logger=self.logger)
            s5_conv1 = tf_utils.conv2d(x=s5_maxpool, output_dim=self.gen_c[8], k_h=3, k_w=3, d_h=1, d_w=1,
                                       initializer='He', name='s5_conv1', logger=self.logger)
            if self.use_batch_norm:
                s5_conv1 = tf_utils.norm(s5_conv1, name='s5_norm0', _type=self.norm, _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s5_conv1 = tf_utils.relu(s5_conv1, name='relu_s5_conv1', logger=self.logger)

            s5_conv2 = tf_utils.conv2d(x=s5_conv1, output_dim=self.gen_c[9], k_h=3, k_w=3, d_h=1, d_w=1,
                                       initializer='He', name='s5_conv2', logger=self.logger)
            if self.use_batch_norm:
                s5_conv2 = tf_utils.norm(s5_conv2, name='s5_norm1', _type=self.norm, _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s5_conv2 = tf_utils.relu(s5_conv2, name='relu_s5_conv2', logger=self.logger)
            s5_conv2_drop = tf_utils.dropout(x=s5_conv2, keep_prob=keep_rate, name='s5_dropout',
                                             logger=self.logger)

            # E6: (10, 9) -> (20, 18)
            s6_deconv1 = tf_utils.deconv2d(x=s5_conv2_drop, output_dim=self.gen_c[10], k_h=2, k_w=2,
                                           initializer='He', name='s6_deconv1', logger=self.logger)
            if self.use_batch_norm:
                s6_deconv1 = tf_utils.norm(s6_deconv1, name='s6_norm0', _type=self.norm, _ops=self._ops,
                                           is_train=self.trainMode, logger=self.logger)
            s6_deconv1 = tf_utils.relu(s6_deconv1, name='relu_s6_deconv1', logger=self.logger)

            # Concat
            s6_concat = tf_utils.concat(values=[s6_deconv1, s4_conv2_drop], axis=3, name='s6_axis3_concat',
                                        logger=self.logger)

            s6_conv2 = tf_utils.conv2d(x=s6_concat, output_dim=self.gen_c[11], k_h=3, k_w=3, d_h=1, d_w=1,
                                       initializer='He', name='s6_conv2', logger=self.logger)
            if self.use_batch_norm:
                s6_conv2 = tf_utils.norm(s6_conv2, name='s6_norm1', _type=self.norm, _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s6_conv2 = tf_utils.relu(s6_conv2, name='relu_s6_conv2', logger=self.logger)

            # Addition
            s6_conv2 = tf_utils.identity(s6_conv2 + s4_conv1, name='stage6_add', logger=self.logger)

            s6_conv3 = tf_utils.conv2d(x=s6_conv2, output_dim=self.gen_c[12], k_h=3, k_w=3, d_h=1, d_w=1,
                                       initializer='He', name='s6_conv3', logger=self.logger)
            if self.use_batch_norm:
                s6_conv3 = tf_utils.norm(s6_conv3, name='s6_norm2', _type=self.norm, _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s6_conv3 = tf_utils.relu(s6_conv3, name='relu_s6_conv3', logger=self.logger)

            # E7: (20, 18) -> (40, 35)
            s7_deconv1 = tf_utils.deconv2d(x=s6_conv3, output_dim=self.gen_c[13], k_h=2, k_w=2, initializer='He',
                                           name='s7_deconv1', logger=self.logger)
            if self.use_batch_norm:
                s7_deconv1 = tf_utils.norm(s7_deconv1, name='s7_norm0',_type=self.norm, _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s7_deconv1 = tf_utils.relu(s7_deconv1, name='relu_s7_deconv1', logger=self.logger)

            # Cropping
            w1 = s3_conv2.get_shape().as_list()[2]
            w2 = s7_deconv1.get_shape().as_list()[2] - s3_conv2.get_shape().as_list()[2]
            s7_deconv1_split, _ = tf.split(s7_deconv1, num_or_size_splits=[w1, w2], axis=2, name='axis2_split')
            tf_utils.print_activations(s7_deconv1_split, logger=self.logger)

            # Concat
            s7_concat = tf_utils.concat(values=[s7_deconv1_split, s3_conv2], axis=3, name='s7_axis3_concat',
                                        logger=self.logger)

            s7_conv2 = tf_utils.conv2d(x=s7_concat, output_dim=self.gen_c[14], k_h=3, k_w=3, d_h=1, d_w=1,
                                       initializer='He', name='s7_conv2', logger=self.logger)
            if self.use_batch_norm:
                s7_conv2 = tf_utils.norm(s7_conv2, name='s7_norm1', _type=self.norm, _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s7_conv2 = tf_utils.relu(s7_conv2, name='relu_s7_conv2', logger=self.logger)

            # Addition
            s7_conv2 = tf_utils.identity(s7_conv2 + s3_conv1, name='stage7_add', logger=self.logger)

            s7_conv3 = tf_utils.conv2d(x=s7_conv2, output_dim=self.gen_c[15], k_h=3, k_w=3, d_h=1, d_w=1,
                                       initializer='He', name='s7_conv3', logger=self.logger)
            if self.use_batch_norm:
                s7_conv3 = tf_utils.norm(s7_conv3, name='s7_norm2', _type=self.norm, _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s7_conv3 = tf_utils.relu(s7_conv3, name='relu_s7_conv3', logger=self.logger)

            # Stage 8 (40, 35) -> (80, 70)
            s8_deconv1 = tf_utils.deconv2d(x=s7_conv3, output_dim=self.gen_c[16], k_h=2, k_w=2, initializer='He',
                                           name='s8_deconv1', logger=self.logger)
            if self.use_batch_norm:
                s8_deconv1 = tf_utils.norm(s8_deconv1, name='s8_norm0', _type=self.norm, _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s8_deconv1 = tf_utils.relu(s8_deconv1, name='relu_s8_deconv1', logger=self.logger)
            # Concat
            s8_concat = tf_utils.concat(values=[s8_deconv1,s2_conv2], axis=3, name='s8_axis3_concat',
                                        logger=self.logger)

            s8_conv2 = tf_utils.conv2d(x=s8_concat, output_dim=self.gen_c[17], k_h=3, k_w=3, d_h=1, d_w=1,
                                       initializer='He', name='s8_conv2', logger=self.logger)
            if self.use_batch_norm:
                s8_conv2 = tf_utils.norm(s8_conv2, name='s8_norm1', _type=self.norm, _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s8_conv2 = tf_utils.relu(s8_conv2, name='relu_s8_conv2', logger=self.logger)

            # Addition
            s8_conv2 = tf_utils.identity(s8_conv2 + s2_conv1, name='stage8_add', logger=self.logger)

            s8_conv3 = tf_utils.conv2d(x=s8_conv2, output_dim=self.gen_c[18], k_h=3, k_w=3, d_h=1, d_w=1,
                                       initializer='He', name='s8_conv3', logger=self.logger)
            if self.use_batch_norm:
                s8_conv3 = tf_utils.norm(s8_conv3, name='s8_norm2', _type=self.norm, _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s8_conv3 = tf_utils.relu(s8_conv3, name='relu_conv3', logger=self.logger)

            # Stage 9 (80, 70) -> (160, 140)
            s9_deconv1 = tf_utils.deconv2d(x=s8_conv3, output_dim=self.gen_c[19], k_h=2, k_w=2,
                                           initializer='He', name='s9_deconv1', logger=self.logger)
            if self.use_batch_norm:
                s9_deconv1 = tf_utils.norm(s9_deconv1, name='s9_norm0',  _type=self.norm, _ops=self._ops,
                                           is_train=self.trainMode, logger=self.logger)
            s9_deconv1 = tf_utils.relu(s9_deconv1, name='relu_s9_deconv1', logger=self.logger)

            # Concat
            s9_concat = tf_utils.concat(values=[s9_deconv1, s1_conv2], axis=3, name='s9_axis3_concat',
                                        logger=self.logger)

            s9_conv2 = tf_utils.conv2d(x=s9_concat, output_dim=self.gen_c[20], k_h=3, k_w=3, d_h=1, d_w=1,
                                       initializer='He', name='s9_conv2', logger=self.logger)
            if self.use_batch_norm:
                s9_conv2 = tf_utils.norm(s9_conv2, name='s9_norm1', _type=self.norm, _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s9_conv2 = tf_utils.relu(s9_conv2, name='relu_s9_conv2', logger=self.logger)

            # Addition
            s9_conv2 = tf_utils.identity(s9_conv2 + s1_conv1, name='stage9_add', logger=self.logger)

            s9_conv3 = tf_utils.conv2d(x=s9_conv2, output_dim=self.gen_c[21], k_h=3, k_w=3, d_h=1, d_w=1,
                                       initializer='He', name='s9_conv3', logger=self.logger)
            if self.use_batch_norm:
                s9_conv3 = tf_utils.norm(s9_conv3, name='s9_norm2', _type=self.norm, _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s9_conv3 = tf_utils.relu(s9_conv3, name='relu_s9_conv3', logger=self.logger)

            # Stage 10 (160, 140) -> (320, 280)
            s10_deconv1 = tf_utils.deconv2d(x=s9_conv3, output_dim=self.gen_c[22], k_h=2, k_w=2,
                                            initializer='He', name='s10_deconv1', logger=self.logger)
            if self.use_batch_norm:
                s10_deconv1 = tf_utils.norm(s10_deconv1, name='s10_norm0', _type=self.norm, _ops=self._ops,
                                            is_train=self.trainMode, logger=self.logger)
            s10_deconv1 = tf_utils.relu(s10_deconv1, name='relu_s10_deconv1', logger=self.logger)
            # Concat
            s10_concat = tf_utils.concat(values=[s10_deconv1, s0_conv2], axis=3, name='s10_axis3_concat',
                                         logger=self.logger)

            s10_conv2 = tf_utils.conv2d(s10_concat, output_dim=self.gen_c[22], k_h=3, k_w=3, d_h=1, d_w=1,
                                        initializer='He', name='s10_conv2', logger=self.logger)
            if self.use_batch_norm:
                s10_conv2 = tf_utils.norm(s10_conv2, name='s10_norm1', _type=self.norm, _ops=self._ops,
                                          is_train=self.trainMode, logger=self.logger)
            s10_conv2 = tf_utils.relu(s10_conv2, name='relu_s10_conv2', logger=self.logger)

            # Addition
            s10_conv2 = tf_utils.identity(s10_conv2 + s0_conv1, name='s10_add', logger=self.logger)

            s10_conv3 = tf_utils.conv2d(x=s10_conv2, output_dim=self.gen_c[22], k_h=3, k_w=3, d_h=1, d_w=1,
                                        initializer='He', name='s10_conv3', logger=self.logger)

            if self.use_batch_norm:
                s10_conv3 = tf_utils.norm(s10_conv3, name='s10_norm2', _type=self.norm, _ops=self._ops,
                                          is_train=self.trainMode, logger=self.logger)
            s10_conv3 = tf_utils.relu(s10_conv3, name='relu_s10_conv3', logger=self.logger)

            s10_conv4 = tf_utils.conv2d(s10_conv3, output_dim=self.gen_c[23], k_h=1, k_w=1, d_h=1, d_w=1,
                                     initializer='He', name='output', logger=self.logger)

            output = tf_utils.tanh(s10_conv4, logger=self.logger, name='output_tanh')

            # Set reuse=True for next call
            self.reuse = True
            self.variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='pix2pix/'+self.name)

        return output

    # def __call__(self, x, keep_rate=0.5, cls_code=None):
    #     with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
    #         tf_utils.print_activations(x, logger=self.logger)
    #
    #         # E0: (320, 280) -> (160, 140)
    #         e0_conv2d = tf_utils.conv2d(x, output_dim=self.gen_c[0], initializer='He', logger=self.logger,
    #                                     name='e0_conv2d')
    #         e0_lrelu = tf_utils.lrelu(e0_conv2d, logger=self.logger, name='e0_lrelu')
    #
    #         # E1: (160, 140) -> (80, 70)
    #         e1_conv2d = tf_utils.conv2d(e0_lrelu, output_dim=self.gen_c[1], initializer='He', logger=self.logger,
    #                                     name='e1_conv2d')
    #         e1_batchnorm = tf_utils.norm(e1_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='e1_norm')
    #         e1_lrelu = tf_utils.lrelu(e1_batchnorm, logger=self.logger, name='e1_lrelu')
    #
    #         # E2: (80, 70) -> (40, 35)
    #         e2_conv2d = tf_utils.conv2d(e1_lrelu, output_dim=self.gen_c[2], initializer='He', logger=self.logger,
    #                                     name='e2_conv2d')
    #         e2_batchnorm = tf_utils.norm(e2_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='e2_norm')
    #         e2_lrelu = tf_utils.lrelu(e2_batchnorm, logger=self.logger, name='e2_lrelu')
    #
    #         # E3: (40, 35) -> (20, 18)
    #         e3_conv2d = tf_utils.conv2d(e2_lrelu, output_dim=self.gen_c[3], initializer='He', logger=self.logger,
    #                                     name='e3_conv2d')
    #         e3_batchnorm = tf_utils.norm(e3_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='e3_norm')
    #         e3_lrelu = tf_utils.lrelu(e3_batchnorm, logger=self.logger, name='e3_lrelu')
    #
    #         # E4: (20, 18) -> (10, 9)
    #         e4_conv2d = tf_utils.conv2d(e3_lrelu, output_dim=self.gen_c[4], initializer='He', logger=self.logger,
    #                                     name='e4_conv2d')
    #         e4_batchnorm = tf_utils.norm(e4_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='e4_norm')
    #         e4_lrelu = tf_utils.lrelu(e4_batchnorm, logger=self.logger, name='e4_lrelu')
    #
    #         # E5: (10, 9) -> (5, 5)
    #         e5_conv2d = tf_utils.conv2d(e4_lrelu, output_dim=self.gen_c[5], initializer='He', logger=self.logger,
    #                                     name='e5_conv2d')
    #         e5_batchnorm = tf_utils.norm(e5_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='e5_norm')
    #         e5_lrelu = tf_utils.lrelu(e5_batchnorm, logger=self.logger, name='e5_lrelu')
    #
    #         # E6: (5, 5) -> (3, 3)
    #         e6_conv2d = tf_utils.conv2d(e5_lrelu, output_dim=self.gen_c[6], initializer='He', logger=self.logger,
    #                                     name='e6_conv2d')
    #         e6_batchnorm = tf_utils.norm(e6_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='e6_norm')
    #         e6_lrelu = tf_utils.lrelu(e6_batchnorm, logger=self.logger, name='e6_lrelu')
    #
    #         # E7: (3, 3) -> (2, 2)
    #         e7_conv2d = tf_utils.conv2d(e6_lrelu, output_dim=self.gen_c[7], initializer='He', logger=self.logger,
    #                                     name='e7_conv2d')
    #         e7_batchnorm = tf_utils.norm(e7_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='e7_norm')
    #         e7_relu = tf_utils.lrelu(e7_batchnorm, logger=self.logger, name='e7_relu')
    #
    #         # ID preserving feature
    #         # if gen_mode == 4:
    #         #     cls_code = tf.dtypes.cast(cls_code, dtype=tf.dtypes.float32)
    #         #     cls_code = tf.reshape(cls_code, shape=(-1, 1, 1, cls_code.get_shape()[-1]), name='cls_code')
    #         #     cls_feat = tf.tile(cls_code, [1, 2, 2, 1], name='cls_code_concat')
    #         #     e7_relu = tf.concat([e7_relu, cls_feat], axis=3, name='id_preserving')
    #         #     tf_utils.print_activations(e7_relu)
    #
    #         # D0: (2, 2) -> (3, 3)
    #         # Stage1: (2, 2) -> (4, 4)
    #         d0_deconv = tf_utils.deconv2d(e7_relu, output_dim=self.gen_c[8], initializer='He', logger=self.logger,
    #                                       name='d0_deconv2d')
    #         # Stage2: (4, 4) -> (3, 3)
    #         shapeA = e6_conv2d.get_shape().as_list()[1]
    #         shapeB = d0_deconv.get_shape().as_list()[1] - e6_conv2d.get_shape().as_list()[1]
    #         d0_split, _ = tf.split(d0_deconv, [shapeA, shapeB], axis=1, name='d0_split1')
    #
    #         shapeA = e6_conv2d.get_shape().as_list()[2]
    #         shapeB = d0_deconv.get_shape().as_list()[2] - e6_conv2d.get_shape().as_list()[2]
    #         d0_split, _ = tf.split(d0_split, [shapeA, shapeB], axis=2, name='d0_split2')
    #
    #         tf_utils.print_activations(d0_split, logger=self.logger)
    #
    #         # Stage3: Batch norm, concatenation, and relu
    #         d0_batchnorm = tf_utils.norm(d0_split, _type=self.norm, _ops=self._ops, logger=self.logger, name='d0_norm')
    #         d0_drop = tf_utils.dropout(d0_batchnorm, keep_prob=keep_rate, logger=self.logger, name='d0_dropout')
    #         d0_concat = tf.concat([d0_drop, e6_batchnorm], axis=3, name='d0_concat')
    #         d0_relu = tf_utils.relu(d0_concat, logger=self.logger, name='d0_relu')
    #
    #         # D1: (3, 3) -> (5, 5)
    #         # Stage1: (3, 3) -> (6, 6)
    #         d1_deconv = tf_utils.deconv2d(d0_relu, output_dim=self.gen_c[9], initializer='He', logger=self.logger,
    #                                       name='d1_deconv2d')
    #         # Stage2: (6, 6) -> (5, 5)
    #         shapeA = e5_batchnorm.get_shape().as_list()[1]
    #         shapeB = d1_deconv.get_shape().as_list()[1] - e5_batchnorm.get_shape().as_list()[1]
    #         d1_split, _ = tf.split(d1_deconv, [shapeA, shapeB], axis=1, name='d1_split1')
    #
    #         shapeA = e5_batchnorm.get_shape().as_list()[2]
    #         shapeB = d1_deconv.get_shape().as_list()[2] - e5_batchnorm.get_shape().as_list()[2]
    #         d1_split, _ = tf.split(d1_split, [shapeA, shapeB], axis=2, name='d1_split2')
    #
    #         tf_utils.print_activations(d1_split, logger=self.logger)
    #
    #         # Stage3: Batch norm, concatenation, and relu
    #         d1_batchnorm = tf_utils.norm(d1_split, _type=self.norm, _ops=self._ops, logger=self.logger, name='d1_norm')
    #         d1_drop = tf_utils.dropout(d1_batchnorm, keep_prob=keep_rate, logger=self.logger, name='d1_dropout')
    #         d1_concat = tf.concat([d1_drop, e5_batchnorm], axis=3, name='d1_concat')
    #         d1_relu = tf_utils.relu(d1_concat, logger=self.logger, name='d1_relu')
    #
    #         # D2: (5, 5) -> (10, 9)
    #         # Stage1: (5, 5) -> (10, 10)
    #         d2_deconv = tf_utils.deconv2d(d1_relu, output_dim=self.gen_c[10], initializer='He', logger=self.logger,
    #                                       name='d2_deconv2d')
    #         # Stage2: (10, 10) -> (10, 9)
    #         shapeA = e4_batchnorm.get_shape().as_list()[2]
    #         shapeB = d2_deconv.get_shape().as_list()[2] - e4_batchnorm.get_shape().as_list()[2]
    #         d2_split, _ = tf.split(d2_deconv, [shapeA, shapeB], axis=2, name='d2_split')
    #         tf_utils.print_activations(d2_split, logger=self.logger)
    #
    #         # Stage3: Batch norm, concatenation, and relu
    #         d2_batchnorm = tf_utils.norm(d2_split, _type=self.norm, _ops=self._ops, logger=self.logger, name='d2_norm')
    #         d2_drop = tf_utils.dropout(d2_batchnorm, keep_prob=keep_rate, logger=self.logger, name='d2_dropout')
    #         d2_concat = tf.concat([d2_drop, e4_batchnorm], axis=3, name='d2_concat')
    #         d2_relu = tf_utils.relu(d2_concat, logger=self.logger, name='d2_relu')
    #
    #         # D3: (10, 9) -> (20, 18)
    #         # Stage1: (10, 9) -> (20, 18)
    #         d3_deconv = tf_utils.deconv2d(d2_relu, output_dim=self.gen_c[11], initializer='He', logger=self.logger,
    #                                       name='d3_deconv2d')
    #         # # Stage2: (20, 18) -> (20, 18)
    #         # shapeA = e3_batchnorm.get_shape().as_list()[2]
    #         # shapeB = d3_deconv.get_shape().as_list()[2] - e3_batchnorm.get_shape().as_list()[2]
    #         # d3_split, _ = tf.split(d3_deconv, [shapeA, shapeB], axis=2, name='d3_split_2')
    #         # tf_utils.print_activations(d3_split, logger=self.logger)
    #
    #         # Stage3: Batch norm, concatenation, and relu
    #         d3_batchnorm = tf_utils.norm(d3_deconv, _type=self.norm, _ops=self._ops, logger=self.logger, name='d3_norm')#d3_split
    #         d3_concat = tf.concat([d3_batchnorm, e3_batchnorm], axis=3, name='d3_concat')
    #         d3_relu = tf_utils.relu(d3_concat, logger=self.logger, name='d3_relu')
    #
    #         # D4: (20, 18) -> (40, 35)
    #         # Stage1: (20, 18) -> (40, 36)
    #         d4_deconv = tf_utils.deconv2d(d3_relu, output_dim=self.gen_c[12], initializer='He', logger=self.logger,
    #                                       name='d4_deconv2d')
    #         # Stage2: (40, 36) -> (40, 35)
    #         shapeA = e2_batchnorm.get_shape().as_list()[2]
    #         shapeB = d4_deconv.get_shape().as_list()[2] - e2_batchnorm.get_shape().as_list()[2]
    #         d4_split, _ = tf.split(d4_deconv, [shapeA, shapeB], axis=2, name='d4_split')
    #         tf_utils.print_activations(d4_split, logger=self.logger)
    #         # Stage3: Batch norm, concatenation, and relu
    #         d4_batchnorm = tf_utils.norm(d4_split, _type=self.norm, _ops=self._ops, logger=self.logger, name='d4_norm')
    #         d4_concat = tf.concat([d4_batchnorm, e2_batchnorm], axis=3, name='d4_concat')
    #         d4_relu = tf_utils.relu(d4_concat, logger=self.logger, name='d4_relu')
    #
    #         # D5: (40, 35, 256) -> (80, 70, 128)
    #         d5_deconv = tf_utils.deconv2d(d4_relu, output_dim=self.gen_c[13], initializer='He', logger=self.logger,
    #                                       name='d5_deconv2d')
    #         d5_batchnorm = tf_utils.norm(d5_deconv, _type=self.norm, _ops=self._ops, logger=self.logger, name='d5_norm')
    #         d5_concat = tf.concat([d5_batchnorm, e1_batchnorm], axis=3, name='d5_concat')
    #         d5_relu = tf_utils.relu(d5_concat, logger=self.logger, name='d5_relu')
    #
    #         # D6: (80, 70, 128) -> (160, 140, 64)
    #         d6_deconv = tf_utils.deconv2d(d5_relu, output_dim=self.gen_c[14], initializer='He', logger=self.logger,
    #                                       name='d6_deconv2d')
    #         d6_batchnorm = tf_utils.norm(d6_deconv, _type=self.norm, _ops=self._ops, logger=self.logger, name='d6_norm')
    #         d6_concat = tf.concat([d6_batchnorm, e0_conv2d], axis=3, name='d6_concat')
    #         d6_relu = tf_utils.relu(d6_concat, logger=self.logger, name='d6_relu')
    #
    #         # D7: (160, 140, 64) -> (320, 280, 1)
    #         d7_deconv = tf_utils.deconv2d(d6_relu, output_dim=self.gen_c[15], initializer='He', logger=self.logger,
    #                                       name='d7_deconv2d')
    #         output = tf_utils.tanh(d7_deconv, logger=self.logger, name='output_tanh')
    #
    #         # Set reuse=True for next call
    #         self.reuse = True
    #         self.variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='pix2pix/'+self.name)
    #
    #     return output


class Discriminator(object):
    def __init__(self, name=None, dis_c=None, norm='instance', logger=None, _ops=None):
        self.name = name
        self.dis_c = dis_c
        self.norm = norm
        self.logger = logger
        self._ops = _ops
        self.reuse = False

    def __call__(self, x):
        with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x, logger=self.logger)

            # H1: (320, 280) -> (160, 140)
            h0_conv2d = tf_utils.conv2d(x, output_dim=self.dis_c[0], initializer='He', logger=self.logger,
                                        name='h0_conv2d')
            h0_lrelu = tf_utils.lrelu(h0_conv2d, logger=self.logger, name='h0_lrelu')

            # H2: (160, 140) -> (80, 70)
            h1_conv2d = tf_utils.conv2d(h0_lrelu, output_dim=self.dis_c[1], initializer='He', logger=self.logger,
                                        name='h1_conv2d')
            h1_norm = tf_utils.norm(h1_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='h1_norm')
            h1_lrelu = tf_utils.lrelu(h1_norm, logger=self.logger, name='h1_lrelu')

            # H3: (80, 70) -> (40, 35)
            h2_conv2d = tf_utils.conv2d(h1_lrelu, output_dim=self.dis_c[2], initializer='He', logger=self.logger,
                                        name='h2_conv2d')
            h2_norm = tf_utils.norm(h2_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='h2_norm')
            h2_lrelu = tf_utils.lrelu(h2_norm, logger=self.logger, name='h2_lrelu')

            # H4: (40, 35) -> (20, 18)
            h3_conv2d = tf_utils.conv2d(h2_lrelu, output_dim=self.dis_c[3], initializer='He', logger=self.logger,
                                        name='h3_conv2d')
            h3_norm = tf_utils.norm(h3_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='h3_norm')
            h3_lrelu = tf_utils.lrelu(h3_norm, logger=self.logger, name='h3_lrelu')

            # H5: (20, 18) -> (20, 18)
            output = tf_utils.conv2d(h3_lrelu, output_dim=self.dis_c[4], k_h=3, k_w=3, d_h=1, d_w=1,
                                     initializer='He', logger=self.logger, name='output_conv2d')

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='pix2pix/'+self.name)

        return output


    # def __call__(self, x): # [32, 64, 128, 256, 512, 1]
    #     with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
    #         tf_utils.print_activations(x, logger=self.logger)
    #
    #         # H1: (320, 280) -> (160, 140)
    #         h0_conv2d = tf_utils.conv2d(x, output_dim=self.dis_c[0], initializer='He', logger=self.logger,
    #                                     name='h0_conv2d')
    #         h0_lrelu = tf_utils.lrelu(h0_conv2d, logger=self.logger, name='h0_lrelu')
    #
    #         # H2: (160, 140) -> (80, 70)
    #         h1_conv2d = tf_utils.conv2d(h0_lrelu, output_dim=self.dis_c[1], k_h=3, k_w=3, d_h=1, d_w=1,
    #                                     initializer='He', logger=self.logger, name='h1_conv2d')
    #         h1_norm = tf_utils.norm(h1_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='h1_norm')
    #         h1_lrelu = tf_utils.lrelu(h1_norm, logger=self.logger, name='h1_lrelu')
    #
    #         h1_conv2d_2 = tf_utils.conv2d(h1_lrelu, output_dim=self.dis_c[1], k_h=3, k_w=3, d_h=1, d_w=1,
    #                                       initializer='He', logger=self.logger, name='h1_conv2d_2')
    #         h1_norm_2 = tf_utils.norm(h1_conv2d_2, _type=self.norm, _ops=self._ops, logger=self.logger, name='h1_norm_2')
    #         h1_lrelu_2 = tf_utils.lrelu(h1_norm_2, logger=self.logger, name='h1_lrelu_2')
    #
    #         h1_avg_pool = tf_utils.avg_pool(h1_lrelu_2, logger=self.logger, name='h1_avg_pool')
    #
    #         # H3: (80, 70) -> (40, 35)
    #         h2_conv2d = tf_utils.conv2d(h1_avg_pool, output_dim=self.dis_c[2], k_h=3, k_w=3, d_h=1, d_w=1,
    #                                     initializer='He', logger=self.logger, name='h2_conv2d')
    #         h2_norm = tf_utils.norm(h2_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='h2_norm')
    #         h2_lrelu = tf_utils.lrelu(h2_norm, logger=self.logger, name='h2_lrelu')
    #
    #         h2_conv2d_2 = tf_utils.conv2d(h2_lrelu, output_dim=self.dis_c[2], k_h=3, k_w=3, d_h=1, d_w=1,
    #                                       initializer='He', logger=self.logger, name='h2_conv2d_2')
    #         h2_norm_2 = tf_utils.norm(h2_conv2d_2, _type=self.norm, _ops=self._ops, logger=self.logger, name='h2_norm_2')
    #         h2_lrelu_2 = tf_utils.lrelu(h2_norm_2, logger=self.logger, name='h2_lrelu_2')
    #
    #         h2_avg_pool = tf_utils.avg_pool(h2_lrelu_2, logger=self.logger, name='h2_avg_pool')
    #
    #         # H4: (40, 35) -> (20, 18)
    #         h3_conv2d = tf_utils.conv2d(h2_avg_pool, output_dim=self.dis_c[3], k_h=3, k_w=3, d_h=1, d_w=1,
    #                                     initializer='He', logger=self.logger, name='h3_conv2d')
    #         h3_norm = tf_utils.norm(h3_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='h3_norm')
    #         h3_lrelu = tf_utils.lrelu(h3_norm, logger=self.logger, name='h3_lrelu')
    #
    #         h3_conv2d_2 = tf_utils.conv2d(h3_lrelu, output_dim=self.dis_c[3], k_h=3, k_w=3, d_h=1, d_w=1,
    #                                       initializer='He', logger=self.logger, name='h3_conv2d_2')
    #         h3_norm_2 = tf_utils.norm(h3_conv2d_2, _type=self.norm, _ops=self._ops, logger=self.logger, name='h3_norm_2')
    #         h3_lrelu_2 = tf_utils.lrelu(h3_norm_2, logger=self.logger, name='h3_lrelu_2')
    #
    #         h3_avg_pool = tf_utils.avg_pool(h3_lrelu_2, logger=self.logger, name='h3_avg_pool')
    #
    #         # H5: (20, 18) -> (10, 9)
    #         h4_conv2d = tf_utils.conv2d(h3_avg_pool, output_dim=self.dis_c[4], k_h=3, k_w=3, d_h=1, d_w=1,
    #                                     initializer='He', logger=self.logger, name='h4_conv2d')
    #         h4_norm = tf_utils.norm(h4_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='h4_norm')
    #         h4_lrelu = tf_utils.lrelu(h4_norm, logger=self.logger, name='h4_lrelu')
    #
    #         h4_conv2d_2 = tf_utils.conv2d(h4_lrelu, output_dim=self.dis_c[4], k_h=3, k_w=3, d_h=1, d_w=1,
    #                                       initializer='He', logger=self.logger, name='h4_conv2d_2')
    #         h4_norm_2 = tf_utils.norm(h4_conv2d_2, _type=self.norm, _ops=self._ops, logger=self.logger, name='h4_norm_2')
    #         h4_lrelu_2 = tf_utils.lrelu(h4_norm_2, logger=self.logger, name='h4_lrelu_2')
    #
    #         h4_avg_pool = tf_utils.avg_pool(h4_lrelu_2, logger=self.logger, name='h4_avg_pool')
    #
    #         # H5: (10, 9) -> (10, 9)
    #         output = tf_utils.conv2d(h4_avg_pool, output_dim=self.dis_c[5], k_h=3, k_w=3, d_h=1, d_w=1,
    #                                  initializer='He', logger=self.logger, name='output_conv2d')
    #
    #         # set reuse=True for next call
    #         self.reuse = True
    #         self.variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='pix2pix/'+self.name)
    #
    #     return output