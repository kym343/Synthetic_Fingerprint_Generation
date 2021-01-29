# --------------------------------------------------------------------------
# Tensorflow Implementation of Synthetic Fingerprint Generation
# Licensed under The MIT License [see LICENSE for details]
# Written by Young-Mook Kang
# Email: kym343@naver.com
# --------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import os
import cv2
import time
import argparse
import FP_matcher
import utils as utils

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

def main(data_path, overlap):
    GT_Paths, output_Paths = load_GT_and_output(data_path, overlap)
    numImgs = len(GT_Paths)

    # total_metric = np.zeros((numImgs, 2))

    FP_matcher.ObtainLicenses()

    for i in range(5800, 7200, 1):# for i, GT_path in enumerate(GT_Paths):
        GT_path = GT_Paths[i]
        if i % 200 == 0:
            print('Processing {} / {}...'.format(i, numImgs))

        output_path = output_Paths[i]

        score, GT_quality, Output_quality = FP_matcher.single_match_from_file(GT_path, output_path)
        # print("score: {}, GT_quality: {}, Output_quality: {}".format(score, GT_quality, Output_quality))
        print("{}, {}".format(score, Output_quality))

        # Calculate each metric
        # total_metric[i, 0] = score          # Matching Score
        # total_metric[i, 1] = GT_quality     # GT Quality
        # total_metric[i, 2] = Output_quality # Output Quality

        # total_metric[i, 0] = score          # Matching Score
        # total_metric[i, 1] = Output_quality # Output Quality

    # total_mean = np.mean(total_metric, axis=0)
    # total_std = np.std(total_metric, axis=0)

    # print("===============================================================")
    # print(" Veryfinger Matching Score   :   {:.3f} +- {:.3f}".format(total_mean[0], total_std[0]))
    # print(" GT image's Quality          :   {:.3f} +- {:.3f}".format(total_mean[1], total_std[1]))
    # print(" Output image's Quality      :   {:.3f} +- {:.3f}".format(total_mean[2], total_std[2]))
    # print("===============================================================")

    print("===============================================================")
    print(" Veryfinger Matching Score   :   {:.3f} +- {:.3f}".format(total_mean[0], total_std[0]))
    print(" Output image's Quality      :   {:.3f} +- {:.3f}".format(total_mean[1], total_std[2]))
    print("===============================================================")

if __name__ == '__main__':
    main(args.data_path, args.overlap)