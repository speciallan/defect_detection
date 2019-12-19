#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os, sys
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element,ElementTree,tostring

sys.path.append('..')

from taurus_cv.utils.spe import spe
from taurus_projects.defect_detection.config import current_config as config


def main():

    origin_data_dir = os.path.join(config.data_path, 'pic1')
    root_target_data_dir = os.path.join(config.voc_path, config.voc_sub_dir)

    class_dirs = os.listdir(origin_data_dir)
    class_dirs.sort()

    train_label = 'Train' #Test

    for k, class_dir in enumerate(class_dirs):

        lower_class = class_dir.lower()
        train_dataset_path = os.path.join(origin_data_dir, class_dir, class_dir, 'Train')
        test_dataset_path = os.path.join(origin_data_dir, class_dir, class_dir, 'Test')

        target_data_dir = root_target_data_dir

        # 测试阶段只要class2
        if lower_class != 'class2':
           continue

        # 确认文件夹存在
        for i in ['train', 'test']:
            for j in ['Annotations', 'JPEGImages', 'ClfImages', 'ImageSets/Main']:
                if not os.path.exists(os.path.join(target_data_dir, i, j)):
                    os.makedirs(os.path.join(target_data_dir, i, j))

        # 找到lables.txt文件
        train_labels_file_path = os.path.join(train_dataset_path, 'Label', 'Labels.txt')
        test_labels_file_path = os.path.join(test_dataset_path, 'Label', 'Labels.txt')

        train_arr, test_arr, clf_train_labels_arr, clf_test_labels_arr = [], [], [], []

        # 读取训练集labels，只要有label的
        with open(train_labels_file_path, 'r') as f:

            next(f)
            for line in f.readlines():


                # print(line.replace('\n', '').split('\t'))
                img_num, has_label, img_file, _, label_file = line.replace('\n', '').split('\t')[:5]

                # 写xml
                if has_label == '1':

                    try:
                        # 对分类中每一张图片生成新图片和xml
                        generate_target_img(train_dataset_path, os.path.join(target_data_dir, 'train', 'JPEGImages'), filename=img_file, class_name=lower_class)

                        generate_target_xml(train_dataset_path, os.path.join(target_data_dir, 'train', 'Annotations'), filename=label_file, class_name=lower_class)

                    except Exception as e:
                        spe(e, img_num, has_label)

                    # 记录到Main/train.txt
                    train_arr.append(img_num)
                    clf_train_labels_arr.append(' '.join([img_num, '1']))


                # 记录到Main/train.txt
                elif has_label == '0':
                    train_arr.append(img_num)
                    clf_train_labels_arr.append(' '.join([img_num, '0']))

        # 读取测试集labels， 0和1都要
        with open(test_labels_file_path, 'r') as f:

            next(f)
            for line in f.readlines():

                # print(line.replace('\n', '').split('\t'))
                img_num, has_label, img_file, _, label_file = line.replace('\n', '').split('\t')[:5]

                # 对分类中每一张图片生成新图片和xml
                generate_target_img(test_dataset_path, os.path.join(target_data_dir, 'test', 'JPEGImages'), filename=img_file, class_name=lower_class)

                # 写xml
                if has_label == '1':

                    try:
                        generate_target_xml(test_dataset_path, os.path.join(target_data_dir, 'test', 'Annotations'), filename=label_file, class_name=lower_class)

                    except Exception as e:
                        spe(e, img_num, has_label)

                    # 记录到Main/train.txt
                    test_arr.append(img_num)
                    clf_test_labels_arr.append(' '.join([img_num, '1']))


                # 记录到Main/train.txt
                elif has_label == '0':
                    test_arr.append(img_num)
                    clf_test_labels_arr.append(' '.join([img_num, '0']))

        # 写Main/train.txt
        train_txt_str = '\n'.join(train_arr)
        test_txt_str = '\n'.join(test_arr)
        clf_train_labels_str = '\n'.join(clf_train_labels_arr)
        clf_test_labels_str = '\n'.join(clf_test_labels_arr)

        f = open(os.path.join(target_data_dir, 'train', 'ImageSets', 'Main', 'trainval.txt'), 'w')
        f.write(train_txt_str)

        f = open(os.path.join(target_data_dir, 'test', 'ImageSets', 'Main', 'plan.txt'), 'w')
        f.write(test_txt_str)
        f.close()

        f = open(os.path.join(target_data_dir, 'train', 'ImageSets', 'Main', 'clf_trainval.txt'), 'w')
        f.write(clf_train_labels_str)
        f.close()

        f = open(os.path.join(target_data_dir, 'test', 'ImageSets', 'Main', 'clf_test.txt'), 'w')
        f.write(clf_test_labels_str)
        f.close()
        print('{}数据生成完毕'.format(class_dir))

def generate_target_img(origin_img_dir, target_img_dir, filename, class_name):

    img = cv2.imread(os.path.join(origin_img_dir, filename))
    new_filename = class_name + '_' + filename.replace('.PNG', '.jpg')
    new_filepath = os.path.join(target_img_dir, new_filename)
    cv2.imwrite(new_filepath, img)


def generate_target_xml(origin_img_dir, target_xml_dir, filename, class_name):

    origin_filename = class_name + '_' + filename.replace('_label.PNG', '.jpg')
    # spe(os.path.join(origin_img_label_dir, filename))
    # spe(111,filename)

    # 生成xml
    root = ET.Element('annotation')
    folder_node = ET.Element('folder')
    folder_node.text = 'DAGM'
    root.append(folder_node)
    filename_node = ET.Element('filename')
    filename_node.text = origin_filename
    root.append(filename_node)

    # 图片基本信息
    label_img = cv2.imread(os.path.join(origin_img_dir, 'Label', filename), cv2.IMREAD_GRAYSCALE)
    w, h = np.shape(label_img)

    size_node = ET.SubElement(root, 'size')
    w_node = ET.Element('width')
    w_node.text = str(w)
    h_node = ET.Element('height')
    h_node.text = str(h)
    c_node = ET.Element('depth')
    c_node.text = '3'
    size_node.append(w_node)
    size_node.append(h_node)
    size_node.append(c_node)

    t = cv2.findContours(label_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = t

    # 最大连通域 写object
    for i in range(len(contours)):
        contour = contours[i]

        # xywh -> xmin ymin xmax ymax
        rect = cv2.boundingRect(contour)
        xmin, ymin, xmax, ymax = rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]

        # xml object节点
        object_node = ET.SubElement(root, 'object')
        n_node = ET.Element('name')
        n_node.text = class_name
        object_node.append(n_node)

        diff_node = ET.Element('difficult')
        diff_node.text = str(0)
        object_node.append(diff_node)

        bndbox_node = ET.SubElement(object_node, 'bndbox')
        xmin_node = ET.Element('xmin')
        xmin_node.text = str(xmin)
        ymin_node = ET.Element('ymin')
        ymin_node.text = str(ymin)

        xmax_node = ET.Element('xmax')
        xmax_node.text = str(xmax)
        ymax_node = ET.Element('ymax')
        ymax_node.text = str(ymax)

        bndbox_node.append(xmin_node)
        bndbox_node.append(ymin_node)
        bndbox_node.append(xmax_node)
        bndbox_node.append(ymax_node)

        # 抠图 分类图片
        target_clf_img_dir = target_xml_dir.replace('Annotations', 'ClfImages')

        # pic1原图
        origin_filename = filename.replace('_label.PNG', '.PNG')
        target_filename = filename.replace('_label.PNG', '.jpg')

        origin_img_filepath = os.path.join(origin_img_dir, origin_filename)
        origin_img = cv2.imread(origin_img_filepath)
        cut_img = origin_img[ymin:ymax, xmin:xmax, :]
        cv2.imwrite(os.path.join(target_clf_img_dir, class_name + '_' + target_filename), cut_img)

    prettyXml(root, '\t', '\n')

    xml_filename = class_name + '_' + filename.replace('_label.PNG', '.xml')
    target_filepath = os.path.join(target_xml_dir, xml_filename)

    et = ElementTree(root)
    et.write(target_filepath, 'utf-8', True, method='xml')


def generate_clf_images():
    pass


def prettyXml(element, indent, newline, level = 0): # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
    if element:  # 判断element是否有子元素
        if element.text == None or element.text.isspace(): # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
    #else:  # 此处两行如果把注释去掉，Element的text也会另起一行
        #element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element) # 将elemnt转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1): # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        prettyXml(subelement, indent, newline, level = level + 1) # 对子元素进行递归操作


if __name__ == '__main__':
    main()
