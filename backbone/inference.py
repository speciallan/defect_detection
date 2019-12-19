#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import sys
import os
import argparse
sys.path.append('../..')

import cv2
import numpy as np
from keras import optimizers
from keras.layers import Input
from keras.utils.np_utils import to_categorical

from taurus_cv.models.resnet.resnet import resnet50
from taurus_cv.utils.spe import *
from taurus_projects.defect_detection.config import current_config as config
from taurus_projects.defect_detection.backbone.preprocessing import *

if __name__ == '__main__':

    dagm_data_path = os.path.join(config.voc_path, config.voc_sub_dir)

    clf_data_dir = os.path.join(config.voc_path, config.voc_sub_dir, 'ClfImages/')
    img_data_dir = os.path.join(config.voc_path, config.voc_sub_dir, 'JPEGImages/')
    clf_data_list = dagm_data_path + '/ImageSets/Main/clf_test.txt'
    model_weight_path = config.backbone_weight_path

    input = Input(shape=(224,224,3))

    # compile
    model = resnet50(input, classes_num=2, layer_num=50)

    # load
    model.load_weights(model_weight_path, by_name=True)


    # gt_labels 1 0 1 1 1 0 0 0
    # test_data = ['0576', '0577', '0578', '0579', '0580', '0581', '0582', '0583']

    X_test_data, y_test_data = [], []

    # 读取测试集label文件
    with open(clf_data_list, 'r') as f:

        for line in f.readlines():

            img_file, label = line.replace('\n', '').split(' ')

            X_test_data.append(img_file)
            y_test_data.append(int(label))

    # 生成样本数
    batch_size = 200

    total_num = len(X_test_data)
    id_list = range(total_num)
    ids = random.sample(id_list, batch_size)

    # 构建测试集
    X_test = np.array([np.zeros((224, 224, 3))])
    y_test = np.array([0])
    img_info = np.array([0])

    # -------------------- 自测试 ---------------------
    # clf_data_dir = '/home/speciallan/Documents/python/data/DAGM/ClfImages/'
    # list = os.listdir(clf_data_dir)
    # filename_arr = []
    # for k,v in enumerate(list):
    #     filename = clf_data_dir + v
    #     filename_arr.append(v)
    #     img = cv2.imread(filename)
    #     img_resized = cv2.resize(img, (224,224))
    #     X_test = np.append(X_test, [img_resized], axis=0)
    #
    # X_test = np.delete(X_test, 0, axis=0)
    # pred = model.predict(X_test)
    # # print('pred:', pred)
    # y = np.argmax(pred, axis=1)
    # # print('y:', y)
    #
    # for i in range(len(X_test)):
    #     print(filename_arr[i], pred[i], y[i])
    #
    # exit()
    # -------------------- 自测试 ---------------------


    for i in ids:

        test_img = clf_data_dir + X_test_data[i] + '.jpg'

        # 存在分类图，则读取 否则从原图中截取
        if y_test_data[i] == 1:

            img = cv2.imread(test_img)
            img_resized = cv2.resize(img, (224,224))
            X_test = np.append(X_test, [img_resized], axis=0)
            y_test = np.append(y_test, [1], axis=0)

        elif y_test_data[i] == 0:

            # 分类反例 从原图截取
            test_img = os.path.join(img_data_dir, X_test_data[i] + '.jpg')
            img = cv2.imread(test_img)
            img_resized = cv2.resize(img[:224, :224, :], (224, 224))
            X_test = np.append(X_test, [img_resized], axis=0)
            y_test = np.append(y_test, [0], axis=0)

        img_info = np.append(img_info, [X_test_data[i]], axis=0)

    X_test = np.delete(X_test, 0, axis=0)
    y_test = np.delete(y_test, 0, axis=0)
    img_info = np.delete(img_info, 0, axis=0)

    print('X_test:', X_test.shape, 'y_test:', y_test.shape)

    # 预处理
    X_test = get_mean_img(X_test)

    pred = model.predict(X_test)
    # print('pred:', pred)

    y = np.argmax(pred, axis=1)
    print('y:', y)
    print('y_test:', y_test)
