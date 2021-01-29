# --------------------------------------------------------------------------
# Tensorflow Implementation of Synthetic Fingerprint Generation
# Licensed under The MIT License [see LICENSE for details]
# Written by Young-Mook Kang
# Email: kym343@naver.com
# --------------------------------------------------------------------------

import os
import cv2
import logging
import numpy as np
from scipy.ndimage import rotate
from numpy import sqrt, angle
from numpy.linalg import norm


class FGData(object):
    def __init__(self, data_path, stage):
        self.dataPath = os.path.join(data_path, stage)
        self.stage = stage if stage != 'overfitting' else 'train'

        # Read image paths
        self.img_paths = all_files_under(self.dataPath, subfolder='images', endswith='.png')# subfolder='images' or 'paired' or '0~20'
        print('Number of images in img_paths: {}'.format(len(self.img_paths)))

        # Read label paths
        self.label_paths = all_files_under(self.dataPath, subfolder='labels', endswith='.png')
        print('Number of labels in label_paths: {}'.format(len(self.label_paths)))

        # Read json file to find user ID
        # self.jsonDataObj = JsonData()

    def back_info(self, imgPath, labelPath=None, stage='train'):
        # Find user ID
        # flage, userId = self.jsonDataObj.find_id(target=os.path.basename(imgPath), data_set=stage)
        imgPath_split = imgPath.split('\\')
        imgPath_num_split = imgPath_split[-1].split('_')

        userId = 'U{:03d}'.format((int(imgPath_num_split[0]) - 1) * 10 + (int(imgPath_num_split[1]) - 1) * 5 + int(imgPath_num_split[2]) - 1)

        # Name of the image
        imgName = os.path.basename(imgPath)

        # Read img in grayscale
        img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        # Read label
        if labelPath is None:
            label = np.zeros_like(img)
        else:
            label = cv2.imread(labelPath, cv2.IMREAD_GRAYSCALE)

        new_label, area = latent_generator(seg_img=label)
        new_label = new_label + label

        data = init_canvas(img.shape[0], img.shape[1], channel=1, img1=img, img2=new_label, times=2, axis=1)
        area = '{:.04f}'.format(area)

        return data, userId, imgName, area


def latent_generator(seg_img, area_factor=0.9):
    latent_seg = np.zeros(seg_img.shape, np.uint8)

    # create random 3 point in seg_img
    type = np.random.randint(0, 2) # 0 or 1

    if type == 0:
        pts = np.array([[np.random.randint(0, seg_img.shape[0]), np.random.randint(-seg_img.shape[1]/2, 0)],
                       [np.random.randint(-seg_img.shape[0]/2, 0), np.random.randint(0, seg_img.shape[1])],
                       [np.random.randint(seg_img.shape[0], 2*seg_img.shape[0]), np.random.randint(seg_img.shape[1], 2*seg_img.shape[1])]])
    elif type == 1:
        pts = np.array([[np.random.randint(0, seg_img.shape[0]), np.random.randint(seg_img.shape[1], seg_img.shape[1] + seg_img.shape[1] / 2)],
                        [np.random.randint(-seg_img.shape[0] / 2, 0), np.random.randint(0, seg_img.shape[1])],
                        [np.random.randint(seg_img.shape[0], 2 * seg_img.shape[0]), np.random.randint(-2 * seg_img.shape[1], -seg_img.shape[1])]])

    # calculate centroid, semi_minor, semi_major, ang from random 3 point
    centroid, semi_minor, semi_major, ang = steiner_inellipse(pts)

    # gernerate ellipse(latent_seg) from centroid, semi_minor, semi_major, ang
    cv2.ellipse(latent_seg, (int(centroid[0]), int(centroid[1])), (int(semi_major), int(semi_minor)),
                ang, 0, 360, (1, 1, 1), -1) # (255, 255, 255)

    # calculate area
    union = seg_img * latent_seg
    union_area = np.sum(union)
    seg_area = np.sum(seg_img)
    area = union_area / seg_area

    return union, area


def steiner_inellipse(pts):
    # https://github.com/nicoguaro/ellipse_packing/blob/master/ellipse_packing/ellipse_packing.py

    # centroid
    centroid = np.mean(pts, axis=0)

    # Semiaxes
    A = norm(pts[0, :] - pts[1, :])
    B = norm(pts[1, :] - pts[2, :])
    C = norm(pts[2, :] - pts[0, :])
    Z = sqrt(A ** 4 + B ** 4 + C ** 4 - (A * B) ** 2 - (B * C) ** 2 - (C * A) ** 2)
    semi_minor = 1. / 6. * sqrt(A ** 2 + B ** 2 + C ** 2 - 2 * Z)
    semi_major = 1. / 6. * sqrt(A ** 2 + B ** 2 + C ** 2 + 2 * Z)

    # Angle
    z1 = pts[0, 0] + 1j * pts[0, 1]
    z2 = pts[1, 0] + 1j * pts[1, 1]
    z3 = pts[2, 0] + 1j * pts[2, 1]
    g = 1 / 3 * (z1 + z2 + z3)
    focus_1 = g + sqrt(g ** 2 - 1 / 3 * (z1 * z2 + z2 * z3 + z3 * z1))
    focus_2 = g - sqrt(g ** 2 - 1 / 3 * (z1 * z2 + z2 * z3 + z3 * z1))
    foci = focus_1 - focus_2
    ang = angle(foci, deg=True)
    return centroid, semi_minor, semi_major, ang


def init_canvas(h, w, channel, img1, img2, times=1, axis=0):
    canvas = None
    if axis==0:
        canvas = np.squeeze(np.zeros((times * h,  w, channel), dtype=np.uint8))
        canvas[:h, :] = img1
        canvas[h:, :] = img2
    elif axis==1:
        canvas = np.squeeze(np.zeros((h, times * w, channel), dtype=np.uint8))
        canvas[:, :w] = img1
        canvas[:, w:] = img2

    return canvas


def convert_color_label(img):
    yellow = [102, 255, 255]    # 2: latent - yellow
    green = [102, 204, 0]       # 1: generation - green
    violet = [102, 0, 102]      # 0: background - violet

    img_rgb = np.zeros([*img.shape, 3], dtype=np.uint8)

    for i, color in enumerate([violet, green, yellow]):
        img_rgb[img == i] = color

    return img_rgb


def all_files_under(folder, subfolder=None, endswith='.png'):
    if subfolder is not None:
        new_folder = os.path.join(folder, subfolder)
    else:
        new_folder = folder

    if os.path.isdir(new_folder):
        file_names = [os.path.join(new_folder, fname)
                       for fname in os.listdir(new_folder) if fname.endswith(endswith)]
        return sorted(file_names)
    else:
        return []


def init_logger(logger, log_dir, name, is_train):
    logger.propagate = False  # solve print log multiple times problem
    file_handler, stream_handler = None, None

    if is_train:
        formatter = logging.Formatter(' - %(message)s')

        # File handler
        file_handler = logging.FileHandler(os.path.join(log_dir, name + '.log'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        # Stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        # Add handlers
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)

    return logger, file_handler, stream_handler


def make_folders(is_train=True, cur_time=None, subfolder=None):
    model_dir = os.path.join('../model', subfolder, '{}'.format(cur_time))
    log_dir = os.path.join('../log', subfolder, '{}'.format(cur_time))
    sample_dir = os.path.join('../sample', subfolder, '{}'.format(cur_time))
    val_dir, test_dir = None, None

    if is_train:
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        if not os.path.isdir(sample_dir):
            os.makedirs(sample_dir)
    else:
        val_dir = os.path.join('../val', subfolder, '{}'.format(cur_time))
        test_dir = os.path.join('../test', subfolder, '{}'.format(cur_time))

        if not os.path.isdir(val_dir):
            os.makedirs(val_dir)

        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

    return model_dir, log_dir, sample_dir, val_dir, test_dir


def make_folders_simple(cur_time=None, subfolder=None):
    model_dir = os.path.join('../model', subfolder, '{}'.format(cur_time))
    log_dir = os.path.join('../log', subfolder, '{}'.format(cur_time))

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    return model_dir, log_dir


def save_imgs(img_stores, iter_time=None, save_dir=None, margin=5, img_name=None, name_append='', is_vertical=True):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    num_categories = len(img_stores)
    num_imgs, h, w = img_stores[0].shape[0:3]

    if is_vertical:
        canvas = np.zeros((num_categories * h + (num_categories + 1) * margin,
                           num_imgs * w + (num_imgs + 1) * margin, 3), dtype=np.uint8)

        for i in range(num_imgs):
            for j in range(num_categories):
                canvas[(j + 1) * margin + j * h:(j + 1) * margin + (j + 1) * h,
                (i + 1) * margin + i * w:(i + 1) * (margin + w), :] = img_stores[j][i]
    else:
        canvas = np.zeros((num_imgs * h + (num_imgs + 1) * margin,
                           num_categories * w + (num_categories + 1) * margin, 3), dtype=np.uint8)

        for i in range(num_imgs):
            for j in range(num_categories):
                canvas[(i+1)*margin+i*h:(i+1)*(margin+h), (j+1)*margin+j*w:(j+1)*margin+(j+1)*w, :] = img_stores[j][i]

    if img_name is None:
        cv2.imwrite(os.path.join(save_dir, str(iter_time).zfill(6) + '.png'), canvas)
    else:
        cv2.imwrite(os.path.join(save_dir, name_append+img_name), canvas)


def save_test_and_GT(img_stores, save_dir=None, img_name=None, name_append=''):
    test_output = os.path.join(save_dir, 'output')
    test_GT = os.path.join(save_dir, 'GT')

    if not os.path.isdir(test_output):
        os.makedirs(test_output)

    if not os.path.isdir(test_GT):
        os.makedirs(test_GT)

    num_imgs, h, w = img_stores[0].shape[0:3]

    if num_imgs is 1:
        name_append = '[Output]'
        cv2.imwrite(os.path.join(test_output, name_append + img_name), img_stores[0][0])
        name_append = '[GT]'
        cv2.imwrite(os.path.join(test_GT, name_append + img_name), img_stores[1][0])
    else:
        for i in range(num_imgs):
            name_append = '[Output]'
            cv2.imwrite(os.path.join(test_output, name_append + img_name), img_stores[0][i])
            name_append = '[GT]'
            cv2.imwrite(os.path.join(test_GT, name_append + img_name), img_stores[1][i])