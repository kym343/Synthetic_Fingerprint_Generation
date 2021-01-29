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
import argparse
import utils as utils

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', dest='data_path', type=str,
                    default='../test/generation/20201003-091628',
                    help='Synthetic Fingerprint test dir path')
parser.add_argument('--overlap', dest='overlap', type=str,
                    default='total',
                    help='Overlap range [total, 0~20, 20~40, 40~60, 60~80, 80~100]')
parser.add_argument('--output_dir', dest='output_dir', type=str,
                    default='../test/bmp/',
                    help='Convert PNG to BMP format, Output dir')
parser.add_argument('--direct', dest='direct', type=bool,
                    default=False,
                    help='Direct connection of data_path')

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


def load_direct(data_path):
    # Read data paths
    data_Paths = utils.all_files_under(data_path, endswith='.png')
    print('Number of GT images in img_paths: {}'.format(len(data_Paths)))

    return data_Paths


def main(data_path, overlap, output_dir, direct):
    if not direct:  # GT & Output
        GT_Paths, output_Paths = load_GT_and_output(data_path, overlap)
        numImgs = len(output_Paths)

        model=data_path.split('/')[-1]
        # GT_bmp = os.path.join(output_dir, '{}/{}/{}'.format(model, overlap,'GT'))
        output_bmp = os.path.join(output_dir, '{}/{}/{}'.format(model, overlap,'output'))
        print("output_bmp:{}".format(output_bmp))#

        # if not os.path.isdir(GT_bmp):
        #     os.makedirs(GT_bmp)

        if not os.path.isdir(output_bmp):
            os.makedirs(output_bmp)

        # GT_list = []
        output_list = []

        for i, GT_path in enumerate(output_Paths):
            if i % 200 == 0:
                print('Processing {} / {}...'.format(i, numImgs))

            output_path = output_Paths[i]

            # GT = cv2.imread(GT_path, 0)
            output = cv2.imread(output_path, 0)

            # GT_bmp_name = GT_bmp + '/' + GT_Paths[i].split('\\')[-1]
            output_bmp_name = output_bmp + '/' + output_Paths[i].split('/')[-1]

            # cv2.imwrite('{}.bmp'.format(GT_bmp_name[:-4]), GT)
            cv2.imwrite('{}.bmp'.format(output_bmp_name[:-4]), output)
            print('{}.bmp'.format(output_bmp_name[:-4]))
            # print('{}.bmp'.format(output_bmp_name[:-4]))

            # GT_list.append('{}.bmp'.format(GT_bmp_name[:-4]).split('/')[-1])
            output_list.append('{}.bmp'.format(output_bmp_name[:-4]).split('/')[-1])

        print("==================== FINISH ====================")
        #
        # for i in range(numImgs):
        #     print('NFIQ2 SINGLE {} BMP false false'.format(GT_list[i]))
        #     print('echo " "')

        print("==================== GT ====================")

        for i in range(numImgs):
            print('NFIQ2 SINGLE {} BMP false false'.format(output_list[i]))
            print('echo " "')

        print("==================== output ====================")

    else:
        data_Paths = load_direct(data_path)
        numImgs = len(data_Paths)

        data_bmp = output_dir

        if not os.path.isdir(data_bmp):
            os.makedirs(data_bmp)

        data_list = []

        for i, data_path in enumerate(data_Paths):
            if i % 200 == 0:
                print('Processing {} / {}...'.format(i, numImgs))

            data = cv2.imread(data_path, 0)

            data_bmp_name = data_bmp + data_Paths[i].split('/')[-1]

            cv2.imwrite('{}.bmp'.format(data_bmp_name[:-4]), data)

            data_list.append('{}.bmp'.format(data_bmp_name[:-4]).split('/')[-1])

        print("==================== FINISH ====================")

        for i in range(numImgs):
            print('NFIQ2 SINGLE {} BMP false false'.format(data_list[i]))
            print('echo " "')

if __name__ == '__main__':
    main(args.data_path, args.overlap, args.output_dir, args.direct)