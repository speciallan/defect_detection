#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import sys
import argparse
sys.path.append('..')

import os
from math import ceil

from keras.utils import plot_model

from defect_detection.config import Config
from defect_detection.env import set_runtime_environment
from defect_detection.model.callbacks import get_callbacks
from defect_detection.model.generator import get_generators
from defect_detection.model.loss import getLoss
from defect_detection.model.optimizer import get_optimizer
from defect_detection.model.resnet import resnet_retinanet

set_runtime_environment()

# 获取配置
config = Config('configRetinaNet.json')

# 如果使用resnet
if config.type.startswith('resnet'):
    model, bodyLayers = resnet_retinanet(num_classes=len(config.classes), backbone=config.type, weights='imagenet', nms=True, config=config)
    model.summary()
else:
    model = None
    bodyLayers = None
    print("不存在相关网络({})".format(config.type))
    exit(1)

print("backend: ", config.type)

# 载入预训练权重
model.load_weights('./h5/result.h5', by_name=True, skip_mismatch=True)
# wpath = '../taurus_cv/pretrained_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
# model.load_weights(wpath, by_name=True, skip_mismatch=True)

# if config.do_freeze_layers:
#     conta = 0
#     for l in bodyLayers:
#         if l.name == config.freeze_layer_stop_name:
#             break
#         l.trainable = False
#         print(l.name)
#         conta += 1
#     print("freeze " + str(conta) + " layers")
# model.summary()

# 编译模型
model.compile(loss=getLoss(),
              optimizer=get_optimizer(config.base_lr),
              metrics=['accuracy'])

if config.model_image:
    plot_model(model, to_file='model_image.jpg')


# 数据生成器
train_generator, val_generator, n_train_samples, n_val_samples = get_generators(config.images_path,
                                                                                config.annotations_path,
                                                                                config.train_val_split,
                                                                                config.batch_size,
                                                                                config.classes,
                                                                                img_min_size=config.img_min_size,
                                                                                img_max_size=config.img_max_size,
                                                                                transform=config.augmentation,
                                                                                debug=False)
callbacks = get_callbacks(config)
# (1, 512, 512, 3) (1, 196416, 5) (1, 196416, 8)
# t = next(train_generator)
# print(t[0].shape, t[1][0].shape, t[1][1].shape)
# exit()

model.fit_generator(generator=train_generator,
                    steps_per_epoch=ceil(n_train_samples / config.batch_size),
                    epochs=config.epochs,
                    callbacks=callbacks,
                    validation_data=val_generator,
                    validation_steps=ceil(n_val_samples / config.batch_size))

model.save_weights(config.trained_weights_path)
