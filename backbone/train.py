#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import sys
import os
import argparse
sys.path.append('../..')

import numpy as np
import math
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.layers import Input
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint

from taurus_cv.models.resnet.resnet import resnet50
from taurus_cv.utils.spe import *

from taurus_projects.defect_detection.config import current_config as config
from taurus_projects.defect_detection.backbone.preprocessing import get_mean_img
from taurus_projects.defect_detection.backbone.generator import get_train_generator, get_valid_generator

def main():

    pretrained_model_path = '../../taurus_cv/pretrained_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    dagm_data_path = os.path.join(config.voc_path, config.voc_sub_dir)
    img_data_dir = dagm_data_path + '/JPEGImages/'
    clf_data_dir = dagm_data_path + '/ClfImages/'
    clf_data_list = dagm_data_path + '/ImageSets/Main/clf_trainval.txt'

    num_classes = 2

    X_train = np.array([np.zeros((224, 224, 3))])
    y_train = np.array([0])

    # 这里自己实现图像增强
    positive, negative = 0, 0

    with open(clf_data_list, 'r') as f:

        for line in f.readlines():

            img_file, label = line.replace('\n', '').split(' ')

            # 分类正例
            if label == '1':

                # 分类正例 从clfimages里找
                img_filepath = os.path.join(clf_data_dir, img_file + '.jpg')
                img = cv2.imread(img_filepath)
                img_resized = cv2.resize(img, (224,224))

                X_train = np.append(X_train, [img_resized], axis=0)
                y_train = np.append(y_train, [1], axis=0)
                positive += 1

            elif label == '0':

                # 分类反例 从全图截取
                img_filepath = os.path.join(img_data_dir, img_file + '.jpg')
                img = cv2.imread(img_filepath)
                img_resized = cv2.resize(img[:224, :224, :], (224,224))

                X_train = np.append(X_train, [img_resized], axis=0)
                y_train = np.append(y_train, [0], axis=0)
                negative += 1


    X_train = np.delete(X_train, 0, axis=0)
    y_train = np.delete(y_train, 0, axis=0)

    # X去均值 y编码
    X_train = get_mean_img(X_train)
    y_train = to_categorical(y_train, num_classes)

    print(X_train.shape, y_train.shape)
    print('positive-negative:', positive, negative)

    # 拆分训练集验证集
    split = 0.2
    split_num = int(len(X_train) * split)

    # 验证集
    # X_valid, y_valid = X_train[:split_num], y_train[:split_num]
    # X_train, y_train = X_train[split_num:], y_train[split_num:]
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=split)

    # print('train-valid:', X_train.shape, X_valid.shape)
    # print('y_valid:', y_valid)

    # 模型定义
    input = Input(shape=(224,224,3))

    # compile
    model = resnet50(input, classes_num=2, is_transfer_learning=True, layer_num=40)

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    # load & fit
    model.load_weights(pretrained_model_path, by_name=True)
    # model.summary()

    batch_size = int(config.backbone_train_batch_size)
    epoch = int(config.backbone_train_epoch)

    tensorboad = TensorBoard(log_dir='./backbone/logs')
    lr_reducer = ReduceLROnPlateau(monitor='loss', # 监视值
                                   factor=0.1,     # 减少学习率的因子，学习率将以lr = lr*factor的形式被减少
                                   cooldown=0,     # 学习率减少后，会经过cooldown个epoch才重新进行检查
                                   patience=5,     # 经过patience个epoch后，如果检测值没变化，则出发学习率减少
                                   min_lr=0.0001)       # 最小学习率

    train_generator = get_train_generator(X_train, y_train, batch_size)
    valid_generator = get_valid_generator(X_valid, y_valid, batch_size)

    # spe(len(X_train) // batch_size, len(X_valid) // batch_size + 1)

    # model.fit(X_train,
    #           y_train,
    #           batch_size=batch_size,
    #           epochs=epoch,
    #           # validation_split=0.2,
    #           validation_data=(X_valid, y_valid),
    #           callbacks=[lr_reducer, tensorboad])

    model.fit_generator(train_generator,
                        epochs=epoch,
                        steps_per_epoch=math.ceil(len(X_train) / batch_size),
                        validation_data=valid_generator,
                        validation_steps=math.ceil(len(X_valid) / batch_size),
                        callbacks=[lr_reducer, tensorboad])

    # save
    model.save_weights('./backbone/dd-resnet50.h5')


if __name__ == '__main__':
    main()