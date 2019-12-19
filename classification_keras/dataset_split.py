#!/usr/bin/env python
#coding=utf-8

############################
## How to use
############################
# This file is used to split data to train/val(val),
# the split path is a directory like:
# split_path
#   ./label1
#       ./label1_1.jpg
#       ./label1_2.jpg
#   ./label2
#       ./label2_1.jpg
#       ./label2_2.jpg
# the split result has the same directory structure with the split_path


import os
from random import shuffle
import shutil
from tqdm import tqdm

path = '/Users/Lavector/dataset/plant_disease'
split_path = os.path.join(path, 'val1')
train_path = os.path.join(path, 'train1')
val_path = os.path.join(path, 'val2')
# split_ratio * img_number will be assigned to train
split_ratio = 0.9

if __name__ == '__main__':
    if os.path.exists(train_path) or os.path.exists(train_path):
        print 'train/val path is existed, please check it! %s/train or val'%(train_path)
    else:
        os.mkdir(train_path)
        os.mkdir(val_path)

        labels = os.listdir(split_path)
        for label in tqdm(labels):
            label_path = os.path.join(split_path, label)
            if not os.path.isdir(label_path):
                continue

            label_imgs = os.listdir(label_path)
            label_imgs_len = len(label_imgs)
            shuffle(label_imgs)
            for i, img in enumerate(label_imgs):
                src_img_path = os.path.join(label_path, img)
                if i < split_ratio * label_imgs_len:
                    dst_label_path = os.path.join(train_path, label)
                    if not os.path.exists(dst_label_path):
                        os.mkdir(dst_label_path)
                    dst_img_path = os.path.join(dst_label_path, img)
                else:
                    dst_label_path = os.path.join(val_path, label)
                    if not os.path.exists(dst_label_path):
                        os.mkdir(dst_label_path)
                    dst_img_path = os.path.join(dst_label_path, img)
                shutil.copy(src_img_path, dst_img_path)
