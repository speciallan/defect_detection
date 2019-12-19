#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os
import numpy as np
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from libtiff import TIFF
from PIL import Image
import cv2
import sys
import time
import argparse
sys.path.append('../..')

from defect_detection.env import set_runtime_environment
from defect_detection.model.image import read_image_bgr, preprocess_image, resize_image, read_image_rgb
from defect_detection.model.resnet import resnet_retinanet
from defect_detection.config import Config


def tiff_to_image_array(tiff_image_name, out_folder, out_type):
    # tiff文件解析成图像序列
    # tiff_image_name: tiff文件名；
    # out_folder：保存图像序列的文件夹
    # out_type：保存图像的类型，如.jpg、.png、.bmp等

    tif = TIFF.open(tiff_image_name, mode="r")
    idx = 0
    for im in list(tif.iter_images()):
        print(im.shape)
        im_name = out_folder + str(idx) + out_type
        img = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(im_name, img)

        print(im_name, 'successfully saved!!!')
        idx = idx + 1
    return


def image_array_to_tiff(image_dir, file_name, image_type, image_num):
    # 图像序列保存成tiff文件
    # image_dir：图像序列所在文件夹
    # file_name：要保存的tiff文件名
    # image_type:图像序列的类型
    # image_num:要保存的图像数目

    out_tiff = TIFF.open(file_name, mode='w')

    # 这里假定图像名按序号排列
    for i in range(0, image_num):
        image_name = image_dir + str(i) + image_type
        image_array = Image.open(image_name)
        # 缩放成统一尺寸
        img = image_array.resize((480, 480), Image.ANTIALIAS)
        out_tiff.write_image(img, compression=None, write_rgb=True)

    out_tiff.close()
    return


def cut_imaegs():
    '''切图'''

    imgs = []
    return imgs

def merge_images():
    '''将最后的真值文本映射回原图'''
    pass


def inference():

    set_runtime_environment()
    start_time = time.time()

    config = Config('configRetinaNet.json')

    wpath = config.trained_weights_path
    result_path = config.test_result_path
    txt_path = result_path.replace('results', 'txt')
    classes = config.classes

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)

    model, _ = resnet_retinanet(len(classes), backbone=config.type, weights='imagenet', nms=True, config=config)
    model.load_weights(wpath, by_name=True, skip_mismatch=True)

    files = sorted(os.listdir(config.test_images_path))

    for nimage, imgf in enumerate(files):

        # if imgf not in ['ship0201606110201801.jpg', 'ship0201606110201902.jpg', 'ship02016061102012014.jpg', 'ship0201711030202902.jpg']:
        #     continue

        # if nimage >= int(len(files) * 0.1):
        #     break

        imgfp = os.path.join(config.test_images_path, imgf)

        if os.path.isfile(imgfp):

            try:
                img = read_image_bgr(imgfp)
            except:
                continue

            img = preprocess_image(img.copy())
            img, scale = resize_image(img, min_side=config.img_min_size, max_side=config.img_max_size)
            _, _, detections = model.predict_on_batch(np.expand_dims(img, axis=0))

            # bbox要取到边界内
            detections[:, :, 0] = np.maximum(0, detections[:, :, 0])
            detections[:, :, 1] = np.maximum(0, detections[:, :, 1])
            detections[:, :, 2] = np.minimum(img.shape[1], detections[:, :, 2])
            detections[:, :, 3] = np.minimum(img.shape[0], detections[:, :, 3])
            detections[0, :, :4] /= scale

            scores = detections[0, :, 4:]

            # 推测置信度
            indices = np.where(detections[0, :, 4:] >= 0.05)

            scores = scores[indices]

            scores_sort = np.argsort(-scores)[:100]

            image_boxes = detections[0, indices[0][scores_sort], :4]
            image_scores = np.expand_dims(detections[0, indices[0][scores_sort], 4 + indices[1][scores_sort]], axis=1)
            image_detections = np.append(image_boxes, image_scores, axis=1)
            image_predicted_labels = indices[1][scores_sort]

            txtfile = imgf.replace('.jpg', '.txt')
            realpath = os.path.join(txt_path, txtfile)
            f = open(realpath, 'w', encoding='utf-8')

            if len(image_boxes) > 0:
                for i, box in enumerate(image_boxes):
                    xmin = int(box[0])
                    ymin = int(box[1])
                    xmax = int(box[2])
                    ymax = int(box[3])
                    # print(xmin, ymin, xmax, ymax)

                    f.write('{} {} {} {} {} {}\n'.format(classes[image_predicted_labels[i]], xmin, ymin, xmax, ymax, image_scores[i][0]))

            f.close()

            print("生成txt '" + txtfile + "'" + ' time:{}, 目标框：{}'.format(time.time() - start_time, len(image_boxes)))
            start_time = time.time()


def main():
    '''
    比赛用得到最后结果

    1、读取tiff
    2、切图成固定大小，并预测
    3、拼接预测结果，得到图和坐标
    4、讲结果映射回原图
    '''

    # tiff_to_image_array('ROIs1158_spring_lc_21_p123.tif', './data/', '.jpg')

    imgs = cut_imaegs()

    inference()

    merge_images()



if __name__ == '__main__':
    main()

