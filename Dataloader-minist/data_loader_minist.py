"""
function read: read png img to ndarray and read labels accordingly
function save: save ndarray to .png img file and return path
"""
import numpy as np
from PIL import Image
from common.path import MINIST_TRAIN_IMG, MINIST_TEST_IMG, MINIST_TRAIN_LABELDIR, MINIST_TEST_LABELDIR, \
    MINIST_TEST_IMG_ADV,MINIST_TEST_IMG_NOISE
from tensorflow import keras
from keras.utils.np_utils import *
from torchvision import transforms
#import cv2
import tensorflow as tf



#Turn img into array

def data_loader(flag, start, end):
    img_list = []
    file_dir = ''
    label_path = ''
    if flag == 'TRAIN':
        file_dir = MINIST_TRAIN_IMG
        label_path = MINIST_TRAIN_LABELDIR + 'train_label.txt'
    elif flag == 'TEST':
        file_dir = MINIST_TEST_IMG
        label_path = MINIST_TEST_LABELDIR + 'test_label.txt'
    elif flag == 'TEST_NOISE':
        file_dir = MINIST_TEST_IMG_NOISE
        label_path = MINIST_TEST_LABELDIR + 'test_label.txt'

    # read img
    for i in range(start, end):
        img = file_dir + str(i) + '.png'
        img2 = Image.open(img)
        img3 = np.array(img2)/255.
        img_list.append(img3)
    img_array = np.array(img_list, dtype=np.float32)
    img_array = np.expand_dims(img_array, -1)

    # read img labels
    with open(label_path, 'r') as f:
        labels_str = f.read()
        label_list = labels_str.split(',')[start:end]
        label_list = list(map(int, label_list))
        label_array = np.array(label_list)
        label_matrix = to_categorical(label_array, 10)
    return img_array, label_matrix


# def data_loader_adv(flag, start, end, atkid):
#     img_list = []
#     file_dir = ''
#     if flag == 'TEST_ADV':
#         file_dir = MINIST_TEST_IMG_ADV
#
#     for i in range(start, end):
#         img = file_dir + '_'+str(atkid)+'_' + str(i)+'_' + '.png'
#         image = cv2.imread(img,0)
#         img_list.append(image)
#     img_array = np.array(img_list, dtype=np.float32)
#     img_array = np.expand_dims(img_array, -1)
#
#     return img_array


if __name__ == '__m
