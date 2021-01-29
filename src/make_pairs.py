# --------------------------------------------------------------------------
# Tensorflow Implementation of Synthetic Fingerprint Generation
# Licensed under The MIT License [see LICENSE for details]
# Written by Young-Mook Kang
# Email: kym343@naver.com
# --------------------------------------------------------------------------

import os
import cv2
import argparse

from utils import FGData

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', dest='data_path', type=str,
                    default='../../Data/Fingerprint/Synthetic_Fingerprint_Dataset/',
                    help='Fingeprint semantic segmentation data path')
parser.add_argument('--stage', dest='stage', type=str,
                    default='train',
                    help='Select one of the stage in [train|validation|test|overfitting]')
args = parser.parse_args()


def main(dataPath, stage):
    fgDataObj = FGData(dataPath, stage)
    numImgs = len(fgDataObj.img_paths)

    for i, imgPath in enumerate(fgDataObj.img_paths):
        if i % 200 == 0:
            print('Processing {} / {}...'.format(i, numImgs))

        labelPath = fgDataObj.label_paths[i]

        cnt = 10
        num = 0
        while num < cnt:
            canvas, userId, imgName, area = fgDataObj.back_info(imgPath, labelPath, stage=stage)

            if float(area) > 0.8:
                save_image(img=canvas, imgName=imgName.replace('.png', '') + '_' + userId + "#" + str(num) + "+" + area,
                           folder=os.path.join(dataPath, stage, 'paired')) #paired
                num += 1


def save_image(img, imgName, folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)

    cv2.imwrite(os.path.join(folder, imgName + '.png'), img)


if __name__ == '__main__':
    main(args.data_path, args.stage)


