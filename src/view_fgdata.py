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
from utils import FGData


parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', dest='data_path', type=str,
                    default='../../Data/Fingerprint/Synthetic_Fingerprint_Dataset',#
                    help='SyntheticFingerprint data path')
parser.add_argument('--stage', dest='stage', type=str,
                    default='train',
                    help='Select one of the stage in [train|validation|test|overfitting]')
parser.add_argument('--delay', dest='delay', type=int,
                    default=1000,
                    help='time delay when showing image')
parser.add_argument('--hide_img', dest='hide_img', action='store_true',
                    default=True,
                    help='show image or not')
parser.add_argument('--save_img', dest='save_img', action='store_true',
                    default=True,
                    help='save image in debegImgs folder or not')
args = parser.parse_args()


def show_image_paired(fgData, state='train', hPos=10, wPos=10, saveFolder='../paired_color'):
    preUserId = None
    for i, imgPath in enumerate(fgData.img_paths):
        if i % 200 == 0:
            print('Iteration: {:4d}'.format(i))

        # Read user ID
        # _, userId = fgData.jsonDataObj.find_id(target=os.path.basename(imgPath), data_set=state)

        # Find user ID
        imgPath_split = imgPath.split('/')
        imgPath_num_split = imgPath_split[-1].split('_')

        userId = imgPath_num_split[-1].split('#')[0]
        save_name = imgPath_split[-1].replace('.png', '')

        winName = None
        if not args.hide_img:
            # Window name
            winName = os.path.basename(imgPath.replace('.png', '')) + ' - ' + userId

            # Initialize window and move to the fixed display position
            cv2.namedWindow(winName)
            cv2.moveWindow(winName, wPos, hPos)

        # Read image and load npy file
        img = cv2.imread(imgPath)
        canvas = img

        # label 0~3 data convert to BGR [0~255, 0~255, 0~255] data
        canvas[:, int(img.shape[1]/2):, :] = utils.convert_color_label(img[:, int(img.shape[1]/2):, 0])

        # Intilize canvas and copy the images
        # h, w, c = img.shape
        # canvas = utils.init_canvas(h, w, c, img1=img[:, :img.shape[1]/2, :], img2=labelBgr, times=2, axis=1)

        # Show image
        if not args.hide_img:
            cv2.imshow(winName, canvas)
            if cv2.waitKey(args.delay) & 0xff == 27:
                exit('Esc clicked')

            # Delete all defined window
            cv2.destroyWindow(winName)

        # Save first image of the each user
        if args.save_img:# and (preUserId is not userId)
            #save_image(img=canvas, imgName=args.stage + '_' + userId, folder=saveFolder)
            save_image(img=canvas, imgName=save_name, folder=saveFolder)

        preUserId = userId

def save_image(img, imgName, folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)

    cv2.imwrite(os.path.join(folder, imgName + '.png'), img)


if __name__ == '__main__':
    fgDataObj = FGData(args.data_path, args.stage)
    show_image_paired(fgDataObj, args.stage, saveFolder='../../Data/Fingerprint/Synthetic_Fingerprint_Dataset/train/paired_255')
