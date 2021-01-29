# --------------------------------------------------------------------------
# Tensorflow Implementation of Synthetic Fingerprint Generation
# Licensed under The MIT License [see LICENSE for details]
# Written by Young-Mook Kang
# Email: kym343@naver.com
# --------------------------------------------------------------------------
import os
import cv2
import argparse
import numpy as np

import utils as utils
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import pearsonr

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', dest='data_path', type=str,
                    default='../test/generation/20201003-091628',
                    help='Synthetic Fingerprint test dir path')
parser.add_argument('--overlap', dest='overlap', type=str,
                    default='total',
                    help='Overlap range [total, 0~20, 20~40, 40~60, 60~80, 80~100]')

args = parser.parse_args()


def load_GT_and_output(data_path, overlap='total'):
    dataPath = os.path.join(data_path, overlap)

    # Read GT data paths
    GT_Paths = utils.all_files_under(dataPath, subfolder='GT', endswith='.png')
    print('Number of GT images in img_paths: {}'.format(len(GT_Paths)))

    # Read output data paths
    output_Paths = utils.all_files_under(dataPath, subfolder='output', endswith='.png')
    print('Number of output images in img_paths: {}'.format(len(output_Paths)))

    return GT_Paths, output_Paths


def cal_MAE(GT, output):
    MAE = mean_absolute_error(GT, output)
    return MAE


def cal_RMSE(GT, output):
    MSE = mean_squared_error(GT, output)
    RMSE = np.sqrt(MSE)
    return RMSE


def cal_PSNR(GT, output):
    PSNR = psnr(GT, output)
    return PSNR


def cal_SSIM(GT, output):
    SSIM = ssim(GT, output)
    return SSIM


def cal_PCC(GT, output):
    GT_np = np.array(GT).reshape(-1)
    output_np = np.array(output).reshape(-1)
    PCC = pearsonr(GT_np, output_np)[0]
    return PCC


def main(data_path, overlap):
    GT_Paths, output_Paths = load_GT_and_output(data_path, overlap)
    numImgs = len(GT_Paths)

    total_metric = np.zeros((numImgs, 5))

    for i, GT_path in enumerate(GT_Paths):
        if i % 200 == 0:
            print('Processing {} / {}...'.format(i, numImgs))

        output_path = output_Paths[i]

        GT = cv2.imread(GT_path, cv2.IMREAD_GRAYSCALE)
        output = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)

        # Calculate each metric
        total_metric[i, 0] = cal_MAE(GT, output)    # MAE
        total_metric[i, 1] = cal_RMSE(GT, output)   # RMSE
        total_metric[i, 2] = cal_PSNR(GT, output)   # PSNR
        total_metric[i, 3] = cal_SSIM(GT, output)   # SSIM
        total_metric[i, 4] = cal_PCC(GT, output)    # PCC

    total_mean = np.mean(total_metric, axis=0)
    total_std = np.std(total_metric, axis=0)

    print("===================================")
    print("     MAE   :   {:.3f} +- {:.3f}".format(total_mean[0], total_std[0]))
    print("     RMSE  :   {:.3f} +- {:.3f}".format(total_mean[1], total_std[1]))
    print("     PSNR  :   {:.3f} +- {:.3f}".format(total_mean[2], total_std[2]))
    print("     SSIM  :   {:.3f} +- {:.3f}".format(total_mean[3], total_std[3]))
    print("     PCC   :   {:.3f} +- {:.3f}".format(total_mean[4], total_std[4]))
    print("===================================")

if __name__ == '__main__':
    main(args.data_path, args.overlap)
