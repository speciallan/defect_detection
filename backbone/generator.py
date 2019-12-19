#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import random
import numpy as np

from taurus_projects.defect_detection.backbone.preprocessing import data_augumentation


def get_train_generator(X_train, y_train, batch_size):

    id_list = range(len(X_train))

    while True:

        ids = random.sample(id_list, batch_size)

        X_batch, y_batch = [], []

        for i in ids:
            X_batch.append(X_train[i])
            y_batch.append(y_train[i])

        # 数据增强
        X_batch = data_augumentation(X_batch)

        yield [np.asarray(X_batch), np.asarray(y_batch)]


def get_valid_generator(X_valid, y_valid, batch_size):

    length = len(X_valid)
    id_list = range(length)

    # 如果验证集小于batch数量，就取验证集数量
    if batch_size <= length:
        batch_size = length

    while True:

        ids = random.sample(id_list, batch_size)

        X_batch, y_batch = [], []

        for i in ids:
            X_batch.append(X_valid[i])
            y_batch.append(y_valid[i])

        yield [np.asarray(X_batch), np.asarray(y_batch)]

if __name__ == '__main__':

    get_train_generator()