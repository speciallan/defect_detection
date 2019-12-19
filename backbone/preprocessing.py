#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import cv2
import random
import numpy as np

def get_mean_img(X_train):

    # 平均值，标准差运算
    # X_train[:, :, :, 0] = (X_train[:, :, :, 0] - np.mean(X_train[:, :, :, 0])) / np.std(X_train[:, :, :, 0])
    # X_train[:, :, :, 1] = (X_train[:, :, :, 1] - np.mean(X_train[:, :, :, 1])) / np.std(X_train[:, :, :, 1])
    # X_train[:, :, :, 2] = (X_train[:, :, :, 2] - np.mean(X_train[:, :, :, 2])) / np.std(X_train[:, :, :, 2])

    # 去均值
    # X_train[:, :, :, 0] = (X_train[:, :, :, 0] - np.mean(X_train[:, :, :, 0]))
    # X_train[:, :, :, 1] = (X_train[:, :, :, 1] - np.mean(X_train[:, :, :, 1]))
    # X_train[:, :, :, 2] = (X_train[:, :, :, 2] - np.mean(X_train[:, :, :, 2]))

    return X_train


# 数据增强
def data_augumentation(X_train):

    X_train = random_flip(X_train)
    X_train = random_crop(X_train, (200,200))

    # test_img = X_train[0]
    # cv2.imwrite('pre2.jpg', test_img)
    # exit(11)

    return X_train


# 随机裁剪
def random_crop(batch, crop_size):

    oshape = np.shape(batch[0])
    padding = oshape[0] - crop_size[0]

    for i in range(len(batch)):

        left = random.randint(0, padding)
        up = random.randint(0, padding)
        # print(padding, batch[i].shape, left, up)

        # 截取
        crop_img = batch[i][up:up+crop_size[0], left:left+crop_size[1], :]

        # 还原尺寸
        batch[i] = cv2.resize(crop_img, (224,224))

    return batch

# 翻转
def random_flip(batch):

    for i in range(len(batch)):

        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])

        if bool(random.getrandbits(1)):
            batch[i] = np.flipud(batch[i])

    return batch
