#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os, sys
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element,ElementTree,tostring

sys.path.append('../..')

from taurus_cv.utils.spe import spe
from taurus_projects.defect_detection.config import current_config as config


def main():

    config.voc_sub_dir = 'zazhi'

    origin_data_dir = os.path.join(config.voc_path, config.voc_sub_dir)
    target_data_dir = os.path.join(config.voc_path, config.voc_sub_dir, 'ClfImages')

    # 确认文件夹存在
    if not os.path.exists(origin_data_dir + '/ClfImages'):
        os.mkdir(origin_data_dir + '/ClfImages')

    if not os.path.exists(origin_data_dir + '/ClfImages/train'):
        os.mkdir(origin_data_dir + '/ClfImages/train')
    if not os.path.exists(origin_data_dir + '/ClfImages/train/positive'):
        os.mkdir(origin_data_dir + '/ClfImages/train/positive')
    if not os.path.exists(origin_data_dir + '/ClfImages/train/negative'):
        os.mkdir(origin_data_dir + '/ClfImages/train/negative')

    if not os.path.exists(origin_data_dir + '/ClfImages/test'):
        os.mkdir(origin_data_dir + '/ClfImages/test')
    if not os.path.exists(origin_data_dir + '/ClfImages/test/positive'):
        os.mkdir(origin_data_dir + '/ClfImages/test/positive')
    if not os.path.exists(origin_data_dir + '/ClfImages/test/negative'):
        os.mkdir(origin_data_dir + '/ClfImages/test/negative')

    if not os.path.exists(origin_data_dir + '/ClfImages/valid'):
        os.mkdir(origin_data_dir + '/ClfImages/valid')
    if not os.path.exists(origin_data_dir + '/ClfImages/valid/positive'):
        os.mkdir(origin_data_dir + '/ClfImages/valid/positive')
    if not os.path.exists(origin_data_dir + '/ClfImages/valid/negative'):
        os.mkdir(origin_data_dir + '/ClfImages/valid/negative')

    train_file_path = os.path.join(origin_data_dir, 'ImageSets', 'Main', 'train.txt')
    valid_file_path = os.path.join(origin_data_dir, 'ImageSets', 'Main', 'val.txt')
    test_file_path = os.path.join(origin_data_dir, 'ImageSets', 'Main', 'plan.txt')

    with open(train_file_path, 'r') as f:

        target_train_data_dir = os.path.join(target_data_dir, 'train')
        for line in f.readlines():
            filename = line.split('\n')[0]
            # print(filename)
            p_idx, n_idx = cut_img(filename, origin_data_dir, target_train_data_dir, stride=4, p_ratio=0.8, n_ratio=0.3)
            print('train_img_file:{}, p:{}, n:{}\n'.format(filename, p_idx, n_idx))

    with open(valid_file_path, 'r') as f:
        target_valid_data_dir = os.path.join(target_data_dir, 'valid')
        for line in f.readlines():
            filename = line.split('\n')[0]
            # print(filename)
            p_idx, n_idx = cut_img(filename, origin_data_dir, target_valid_data_dir, stride=4, p_ratio=0.7, n_ratio=0.3)
            print('valid_img_file:{}, p:{}, n:{}'.format(filename, p_idx, n_idx))

    with open(test_file_path, 'r') as f:
        target_test_data_dir = os.path.join(target_data_dir, 'test')
        for line in f.readlines():
            filename = line.split('\n')[0]
            p_idx, n_idx = cut_img(filename, origin_data_dir, target_test_data_dir, stride=4, p_ratio=0.6, n_ratio=0.4)
            print('test_img_file:{}, p:{}, n:{}'.format(filename, p_idx, n_idx))

def cut_img(filename, origin_data_dir, target_data_dir, patch_size=(8,8), stride=1, p_ratio=0.7, n_ratio=0.3):

    annotation_filepath = os.path.join(origin_data_dir, 'Annotations', filename + '.xml')

    origin_data_dir = os.path.join(origin_data_dir, 'JPEGImages')
    origin_img = cv2.imread(origin_data_dir + '/' + filename + '.jpg')

    # spe(filename, origin_data_dir, target_data_dir)
    # spe(origin_img.shape)

    # 解析xml
    et = ET.parse(annotation_filepath)
    element = et.getroot()

    # 解析基础图片数据
    element_objs = element.findall('object')
    element_filename = element.find('filename').text
    element_width = int(element.find('size').find('width').text)
    element_height = int(element.find('size').find('height').text)

    # 加入类别映射
    p_idx, n_idx = 0, 0

    for element_obj in element_objs:

        class_name = element_obj.find('name').text
        obj_bbox = element_obj.find('bndbox')

        # voc的坐标格式
        x1 = int(round(float(obj_bbox.find('xmin').text)))
        y1 = int(round(float(obj_bbox.find('ymin').text)))
        x2 = int(round(float(obj_bbox.find('xmax').text)))
        y2 = int(round(float(obj_bbox.find('ymax').text)))
        difficulty = int(element_obj.find('difficult').text) == 1

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        w, h = x2 - x1, y2 - y1
        # w, h = 50, 50

        # x,y为截取部分左上角的坐标 绝对坐标
        # 如果是杂质 1
        if class_name == '1':

            print('{}:{}'.format(filename, [x for x in range(x1-w,x2,stride)]))

            for x in range(x1-w, x2, stride):

                for y in range(y1-h, y2, stride):

                    if h==0 or w==0:
                        continue


                    # 判断正负例
                    # 前面判断滑动窗口 往目标坐标滑动，后面4个判断是不能出原图边界
                    if x < x1 and y>=0 and y+h<=element_height and x>=0 and x+w<=element_width:

                        cut_img = origin_img[y:y+h, x:x+w, :]

                        if not os.path.exists(target_data_dir + '/negative/'):
                            spe(target_data_dir + '/negative/')

                        # n
                        if x1-x >= int(w*(1-n_ratio)):
                            cv2.imwrite(target_data_dir + '/negative/' + str(filename) + '_' + str(n_idx) + '.jpg', cut_img)
                            # print(target_data_dir + '/negative/' + str(filename) + '_' + str(n_idx) + '.jpg')
                            n_idx += 1

                        # p   8 * 0.3
                        elif x1-x <= int(w*(1-p_ratio)):

                            # if filename == '000070' and p_idx == 10:
                            #     print(y1,w,y1-w,y2)
                            #     print(cut_img)
                            #     exit()

                            cv2.imwrite(target_data_dir + '/positive/' + str(filename) + '_' + str(p_idx) + '.jpg', cut_img)
                            p_idx += 1

                        # spe(target_data_dir + '/positive/' + str(filename) + '_' + str(idx) + '.jpg', cut_img.shape)

                    elif x < x2 and y>=0 and y+h<=element_height and x>=0 and x+w<=element_width:

                        cut_img = origin_img[y:y+h, x:x+w, :]

                        # p
                        if x - x1 <= int(w*(1-p_ratio)):

                            cv2.imwrite(target_data_dir + '/positive/' + str(filename) + '_' + str(p_idx) + '.jpg', cut_img)
                            p_idx += 1
                        # n
                        elif x - x1 >= int(w*(1-n_ratio)):
                            cv2.imwrite(target_data_dir + '/negative/' + str(filename) + '_' + str(n_idx) + '.jpg', cut_img)
                            n_idx += 1

    return p_idx, n_idx


if __name__ == '__main__':
    main()
