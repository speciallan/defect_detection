#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import sys
import configparser
import numpy as np

sys.path.append('../..')

from taurus_cv.pretrained_models.get import get as get_pretrained_model
from taurus_cv.models.faster_rcnn.config import LinuxVocConfig

class ProjectLinuxVocConfig(LinuxVocConfig):

    NAME = 'Defect Detection'

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_INPUT_SHAPE = (IMAGE_MAX_DIM, IMAGE_MAX_DIM, 3)

    # batch_size
    IMAGES_PER_GPU = 1
    BATCH_SIZE = IMAGES_PER_GPU

    # 缺陷检测
    # 一阶段二分类 用于roihead的分类数
    NUM_CLASSES = 1 + 6
    CLASS_MAPPING = {'bg': 0,
                     '1': 1,
                     '2': 2,
                     '3': 3,
                     '4': 4,
                     '5': 5,
                     '6': 6
                     }

    # 并行GPU数量
    GPU_COUNT = 1

    # 每次训练的图片数量 1080ti可以2 8g显存为1
    IMAGES_PER_GPU = 1

    # 每个epoch需要训练的次数
    STEPS_PER_EPOCH = 1000

    # CNN的架构
    BACKBONE = 'resnet50'

    # BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    FPN_CLF_FC_SIZE = 1024

    # zazhi 4/16/0.5,1,2//0.5,1,2
    # gangdaiyin 32/128/1,1.5,2/0.5,1,2

    # anchors shift 网络步长 anchor_base_size/4   根据骨干网缩小倍数8
    BACKBONE_STRIDE = 8
    RPN_ANCHOR_BASE_SIZE = 16 # 缺陷在16,32,64,128左右
    RPN_ANCHOR_SCALES = [0.5, 1, 2, 16] # 1,2
    RPN_ANCHOR_RATIOS = [0.3, 1, 3]

    # BACKBONE_STRIDE = 4
    # RPN_ANCHOR_BASE_SIZE = 16 # 缺陷在16,32,64,128左右
    # RPN_ANCHOR_SCALES = [0.5, 1, 2] # 1,2
    # RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_ANCHOR_NUM = len(RPN_ANCHOR_SCALES) * len(RPN_ANCHOR_RATIOS)

    # 不同数据集这个阈值不同 0.28
    RPN_SCORE_THRESHOLD = 0.4

    # RPN提议框非极大抑制阈值(训练时可以增加该值来增加提议框) 越小框越少 训练时用0.9有很多rois
    RPN_NMS_THRESHOLD_TRAIN = 0.9 # 获取几乎所有正确rois
    RPN_NMS_THRESHOLD_INFERENCE = 0.01 # 获取最好的无重叠roi

    # 每张图像训练anchors个数
    RPN_TRAIN_ANCHORS_PER_IMAGE = 100

    # 训练和预测阶段NMS后保留的ROIs数
    POST_NMS_ROIS_TRAIN = 100
    # rpn输出roi个数  output num
    POST_NMS_ROIS_INFERENCE = 100

    # 检测网络训练rois数和正样本比
    TRAIN_ROIS_PER_IMAGE = 100
    ROI_POSITIVE_RATIO = 0.33

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # 最大ground_truth实例
    MAX_GT_INSTANCES = 100

    # 最大最终检测实例数
    DETECTION_MAX_INSTANCES = 100

    # 检测最小置信度
    DETECTION_MIN_CONFIDENCE = 0.4

    # 检测MNS阈值
    DETECTION_NMS_THRESHOLD = 0.2

    # 训练参数
    LEARNING_RATE = 0.01
    LEARNING_MOMENTUM = 0.9

    # 权重衰减
    WEIGHT_DECAY = 0.0001

    # 梯度裁剪
    GRADIENT_CLIP_NORM = 1.0

    # 损失函数权重
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "rcnn_class_loss": 1.,
        "rcnn_bbox_loss": 1.
    }

    username = 'speciallan'
    data_path = '/home/speciallan/Documents/python/data'
    voc_path = '/home/speciallan/Documents/python/data/VOCdevkit'
    voc_sub_dir = 'VOC2007'

# 当前配置
current_config = ProjectLinuxVocConfig()

# 获取用户配置
cf = configparser.ConfigParser()
cf.read(current_config.config_filepath)
sections = cf.sections()

for k,section in enumerate(sections):
    user_config = cf.items(section)
    for k2,v in enumerate(user_config):
        current_config.__setattr__(v[0], v[1])

# 使用imagenet的resnet50
# current_config.pretrained_weights = get_pretrained_model(weight_path=current_config.backbone_weight_path)

