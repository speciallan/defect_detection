#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import sys
sys.path.append('../..')

from keras.models import load_model
import numpy as np
import cv2
import time
import os

from taurus_cv.utils.spe import spe

model = load_model('backbone/snet.h5')
model.summary()
start_time = time.time()

# 000345
img_dir = '/home/speciallan/Documents/python/data/VOCdevkit/zazhi/Images/'
stride = 8 #5
threshold = 0.85

"""
positive: 332
negative: 100
acc: 0.9351851851851852
precision: 0.9759036144578314
recall: 0.9418604651162791
fp:['000053.jpg', '000222.jpg', '000006.jpg', '000050.jpg', '000141.jpg', '000047.jpg', '000020.jpg', '000002.jpg']   8
fn:['54.jpg', '49.jpg', '39.jpg', '12.jpg', '63.jpg', '20.jpg', '32.jpg', '55.jpg', '18.jpg', '42.jpg', '48.jpg', '53.jpg', '75.jpg', '64.jpg', '58.jpg', '60.jpg', '84.jpg', '22.jpg', '17.jpg', '24.jpg']   20

"""

def main():

    if not os.path.exists('backbone/results'):
        os.mkdir('backbone/results')

    positive_imgs = os.listdir(img_dir + '1/')
    negative_imgs = os.listdir(img_dir + '0/')

    p_total = len(positive_imgs)
    n_total = len(negative_imgs)

    total = p_total + n_total
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    fp_files, fn_files = [], []

    for i in range(len(positive_imgs)):

        rect_total = classify(positive_imgs[i], 1)

        # 找到至少一个杂质
        if rect_total > 0:
            tp += 1
        else:
            print('false positive:{}'.format(positive_imgs[i]))
            fp_files.append(positive_imgs[i])
            fp += 1

    for i in range(len(negative_imgs)):

        rect_total = classify(negative_imgs[i], 0)

        # 找到至少一个杂质
        if rect_total > 0:
            print('false negative:{}'.format(negative_imgs[i]))
            fn_files.append(negative_imgs[i])
            fn += 1
        else:
            tn += 1

    acc = (tp + tn) / total
    # 在这里这两个指标没有意义 对检测点才有
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)

    print('positive: {}'.format(p_total))
    print('negative: {}'.format(n_total))
    print('acc: {}'.format(acc))
    # print('precision: {}'.format(precision))
    # print('recall: {}'.format(recall))
    print('fp:{}'.format(fp_files))
    print('fn:{}'.format(fn_files))

def classify(filename, label=0):

    img = cv2.imread(os.path.join(img_dir, str(label), filename))

    batch = np.array([np.zeros((10,10,3))])
    info = np.array([np.zeros(2)])
    print(img.shape)
    w, h, c = img.shape
    for y in range(0, h-10, stride):
        for x in range(0, w-10, stride):
            cut_img = img[x:x+10, y:y+10, :]
            batch = np.append(batch, [cut_img], axis=0)
            info = np.append(info, [(int(x), int(y))], axis=0)
    batch = np.delete(batch, 0, axis=0)
    info = np.delete(info, 0, axis=0)
    batch = batch / 255

    info = info.astype('int')
    print('cut done', time.time() - start_time)

    # print(batch.shape)
    predict_result = model.predict(batch)
    # y = np.argmax(predict_result, axis=1)

    boxes = []
    #
    total = 0
    img_rect = img
    font = cv2.FONT_HERSHEY_SIMPLEX

    predict_result_sorted = []
    for i in range(len(predict_result)):

        total += 1
        predict_result_sorted.append([predict_result[i], info[i]])

    predict_result_sorted.sort(key=lambda x:x[0][1], reverse=True)
    predict_result_sorted = predict_result_sorted[:5]
    # print(predict_result_sorted[0][0][1])
    # exit()

    print('sort done', time.time() - start_time)
    start_time2 = time.time()

    rect_total = 0

    for i in range(len(predict_result_sorted)):

        print('prob:{}, cord:{}'.format(predict_result_sorted[i][0][1], predict_result_sorted[i][1]))

        if predict_result_sorted[i][0][1] > threshold:

            y, x = predict_result_sorted[i][1]
            # boxes.append((x,y,x+10,y+10))
            # rect
            img_rect = cv2.rectangle(img_rect, (x, y), (x + 10, y + 10), (0, 0, 255), 1)
            img_rect = cv2.putText(img_rect, str(predict_result_sorted[i][0][1]), (x, y), font, 0.5, (255, 255, 255), 1)
            rect_total += 1

    print(str(label) + '_' + filename + ' predict done', time.time() - start_time2)
    # print('total:', total)

    cv2.imwrite('backbone/results/' + str(label) + '_' + filename, img_rect)

    print('----------------------------------------')

    return rect_total


if __name__ == '__main__':
    main()
