#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import tensorflow as tf
import keras
import os

def set_runtime_environment():
    """
    GPU设置，设置后端，包括字符精度
    :return:
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True  # 不要启动的时候占满gpu显存，按需申请空间
    session = tf.Session(config=cfg)     # 生成tf.session
    keras.backend.set_session(session)   # 设置后端为tensorflow
    # keras.backend.set_floatx('float16')  # 设置字符精度，默认float32，使用float16会提高训练效率，但是可能导致精度不够，梯度出现问题。
