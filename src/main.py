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
from datetime import datetime

import utils as utils
from solver import Solver
from pix2pix import Pix2pix
import dataset as dataset


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_integer('batch_size', 4, 'batch size for one iteration, default: 4')
tf.flags.DEFINE_float('resize_factor', 1.0, 'resize original input image, default: 1.0')
tf.flags.DEFINE_string('dataset', 'Fingerprint', 'dataset name, default: Fingerprint')
tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate for optimizer, default: 0.0002')
tf.flags.DEFINE_integer('epoch', 500, 'number of epoch, default: 200')
tf.flags.DEFINE_integer('print_freq', 50, 'print frequency for loss information, default: 50')
tf.flags.DEFINE_float('lambda_1', 100, '1st hyper-paramter for the conditional L1 loss, default: 100')
tf.flags.DEFINE_float('lambda_2', 10, '2nd hyper-paramter for the conditional L1 loss, default: 10')
tf.flags.DEFINE_float('lambda_3', 10, '3rd hyper-paramter for the conditional L1 loss, default: 10')
tf.flags.DEFINE_bool('use_batch_norm', True, 'use batch norm for the model, default: True')
tf.flags.DEFINE_integer('sample_freq', 1000, 'sample frequence for checking qualitative evaluation, default: 1000')
tf.flags.DEFINE_integer('sample_batch', 4, 'number of sampling images for check generator quality, default: 4')
tf.flags.DEFINE_integer('save_freq', 1000, 'save frequency for model, default: 50000')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model that you wish to continue training '
                                           '(e.g. 20191003-103205), default: None')


def print_main_parameters(logger, flags, is_train=False):
    if is_train:
        logger.info('gpu_index: \t\t\t{}'.format(flags.gpu_index))
        logger.info('batch_size: \t\t\t{}'.format(flags.batch_size))
        logger.info('resize_factor: \t\t{}'.format(flags.resize_factor))
        logger.info('dataset: \t\t\t{}'.format(flags.dataset))
        logger.info('is_train: \t\t\t{}'.format(flags.is_train))
        logger.info('learning_rate: \t\t{}'.format(flags.learning_rate))
        logger.info('epoch: \t\t\t{}'.format(flags.epoch))
        logger.info('print_freq: \t\t\t{}'.format(flags.print_freq))
        logger.info('lambda_1: \t\t\t{}'.format(flags.lambda_1))
        logger.info('lambda_2: \t\t\t{}'.format(flags.lambda_2))
        logger.info('lambda_3: \t\t\t{}'.format(flags.lambda_3))
        logger.info('use_batch_norm: \t\t{}'.format(flags.use_batch_norm))
        logger.info('sample_freq: \t\t{}'.format(flags.sample_freq))
        logger.info('sample_batch: \t\t{}'.format(flags.sample_batch))
        logger.info('save_freq: \t\t\t{}'.format(flags.save_freq))
        logger.info('load_model: \t\t\t{}'.format(flags.load_model))

    else:
        print('-- gpu_index: \t\t\t{}'.format(flags.gpu_index))
        print('-- batch_size: \t\t\t{}'.format(flags.batch_size))
        print('-- resize_factor: \t\t{}'.format(flags.resize_factor))
        print('-- dataset: \t\t\t{}'.format(flags.dataset))
        print('-- is_train: \t\t\t{}'.format(flags.is_train))
        print('-- learning_rate: \t\t{}'.format(flags.learning_rate))
        print('-- epoch: \t\t\t{}'.format(flags.epoch))
        print('-- print_freq: \t\t\t{}'.format(flags.print_freq))
        print('-- lambda_1: \t\t\t{}'.format(flags.lambda_1))
        print('-- lambda_2: \t\t\t{}'.format(flags.lambda_2))
        print('-- lambda_3: \t\t\t{}'.format(flags.lambda_3))
        print('-- use_batch_norm: \t\t{}'.format(flags.use_batch_norm))
        print('-- sample_freq: \t\t{}'.format(flags.sample_freq))
        print('-- sample_batch: \t\t{}'.format(flags.sample_batch))
        print('-- save_freq: \t\t\t{}'.format(flags.save_freq))
        print('-- load_model: \t\t\t{}'.format(flags.load_model))


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_index

    # Initialize model and log folders:
    if FLAGS.load_model is None:
        cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        cur_time = FLAGS.load_model

    model_dir, log_dir, sample_dir, _, test_dir = utils.make_folders(is_train=FLAGS.is_train,
                                                                     cur_time=cur_time,
                                                                     subfolder='generation')

    # Logger
    logger = logging.getLogger(__name__)  # logger
    logger.setLevel(logging.INFO)
    utils.init_logger(logger=logger, log_dir=log_dir, is_train=FLAGS.is_train, name='main')
    print_main_parameters(logger, flags=FLAGS, is_train=FLAGS.is_train)

    # Initialize Session
    sess = tf.compat.v1.Session()

    # Initialize dataset
    data = dataset.Dataset(name='generation', resize_factor=FLAGS.resize_factor,
                              is_train=FLAGS.is_train, log_dir=log_dir,  is_debug=False)

    # Initialize model
    pix2pix = Pix2pix(input_img_shape=data.input_img_shape,
                      session=sess,
                      lr=FLAGS.learning_rate,
                      total_iters=int(np.ceil((FLAGS.epoch * data.num_train_imgs) / FLAGS.batch_size)),
                      is_train=FLAGS.is_train,
                      log_dir=log_dir,
                      lambda_1=FLAGS.lambda_1,
                      lambda_2=FLAGS.lambda_2,
                      lambda_3=FLAGS.lambda_3,
                      use_batch_norm=FLAGS.use_batch_norm,
                      num_class=data.num_seg_class)

    # Initialize solver
    solver = Solver(data=data, gen_model=pix2pix, session=sess, flags=FLAGS, log_dir=log_dir)

    if FLAGS.is_train is True:
        train(solver, logger, model_dir, log_dir, sample_dir)
    else:
        test(solver, model_dir, log_dir, test_dir)


def train(solver, logger, model_dir, log_dir, sample_dir):
    iter_time = 0
    total_iters = int(np.ceil((FLAGS.epoch * solver.data.num_train_imgs) / FLAGS.batch_size))

    if FLAGS.load_model is not None:
        flag, iter_time = solver.load_model(logger=logger, model_dir=model_dir, is_train=True)

        if flag is True:
            logger.info(' [!] Load Success! Iter: {}'.format(iter_time))
        else:
            exit(' [!] Failed to restore model {}'.format(FLAGS.load_gan_model))

    # Tensorboard writer
    tb_writer = tf.compat.v1.summary.FileWriter(logdir=log_dir, graph=solver.sess.graph_def)

    while iter_time < total_iters:
        gen_loss, adv_loss, cond_loss, dis_loss, summary = solver.train()

        # Print loss information
        if iter_time % FLAGS.print_freq == 0:
            # Write to tensorboard
            tb_writer.add_summary(summary, iter_time)
            tb_writer.flush()

            msg = "[{0:7} / {1:7}] Dis_loss: {2:.5f} Gen_loss: {3:.3f}, Adv_loss: {4:.3f}, Cond_loss: {5:.3f}"
            print(msg.format(iter_time, total_iters, dis_loss, gen_loss, adv_loss, cond_loss))

        # Sampling generated imgs
        if iter_time % FLAGS.sample_freq == 0:
            solver.img_sample(iter_time, sample_dir, FLAGS.sample_batch)

        # Evaluating
        if (iter_time % FLAGS.save_freq == 0) or (iter_time + 1 == total_iters):
            solver.save_model(logger, model_dir, iter_time)

        iter_time += 1


def test(solver, model_dir, log_dir, test_dir):
    if FLAGS.load_model is not None:
        flag, iter_time = solver.load_model(logger=None, model_dir=model_dir, is_train=False)

        if flag is True:
            print(' [!] Load Success! Iter: {}'.format(iter_time))
        else:
            exit(' [!] Failed to restore model {}'.format(FLAGS.load_gan_model))

    latent, segs, outputs, GT = solver.generate_test_imgs()

    print('Saving imgs...')
    for i in range(segs.shape[0]):
        if i % 100 == 0:
            print('[{}/{}] saving...'.format(i, segs.shape[0]))

        utils.save_imgs(img_stores=[latent[i:i+1], segs[i:i+1], outputs[i:i+1], GT[i:i+1]], save_dir=test_dir,
                        img_name=os.path.basename(solver.data.test_paths[i]), is_vertical=False, margin=0)

        utils.save_test_and_GT(img_stores=[outputs[i:i+1], GT[i:i+1]], save_dir=test_dir,
                               img_name=os.path.basename(solver.data.test_paths[i]))


if __name__ == '__main__':
    tf.compat.v1.app.run()
