#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import sys
sys.path.append('../..')

from keras.models import load_model
import numpy as np
import cv2
import time
from keras.preprocessing import image
# img = image.load_img('./dif_validation/fuli/247.jpg', target_size=(150, 150))
# img_to_array = image.img_to_array(img).astype('float')/255
# img_predict = np.expand_dims(img_to_array, axis=0)
model = load_model('backbone/snet.h5')
model.summary()
# predict_result = model.predict(img_predict)
# y = np.argmax(predict_result, axis=1)
# print(y)


# import os
# img_dir = '/home/mlg1504/whg/DAGM_validation/DAGM_zheng/'
# # img_dir = './impurity_my/'
# imgs = os.listdir(img_dir)
# X_batch = np.array([np.zeros((100,100,3))])
# img_info = np.array([0])
#
# for k,v in enumerate(imgs):
#     img_path = img_dir + v
#     img = image.load_img(img_path, target_size=(100, 100))
#     img_to_array = image.img_to_array(img).astype('float') / 255
#     # print(img_to_array.shape)
#     X_batch = np.append(X_batch, [img_to_array], axis=0)
#     img_info = np.append(img_info, [v], axis=0)
#     # print(img_path)
#
# X_batch = np.delete(X_batch, 0, axis=0)
# img_info = np.delete(img_info, 0, axis=0)
# # print(X_batch.shape)
# # exit()
#
# predict_result = model.predict(X_batch)
# y = np.argmax(predict_result, axis=1)
#
# for i in range(len(predict_result)):
#     if y[i] == 1:
#         print(img_info[i])
#         print(predict_result[i])
#         print(y[i])
# # print(img_info)
# # exit()


start_time = time.time()


# 000345
img = cv2.imread('/home/speciallan/Documents/python/data/VOCdevkit/zazhi/JPEGImages/000000.jpg')
stride = 10 #5
threshold = 0.7

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


print(batch.shape)
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

for i in range(len(predict_result_sorted)):

    print('prob:{}, cord:{}'.format(predict_result_sorted[i][0][1], predict_result_sorted[i][1]))

    if predict_result_sorted[i][0][1] > threshold:

        y, x = predict_result_sorted[i][1]
        # boxes.append((x,y,x+10,y+10))
        # rect
        img_rect = cv2.rectangle(img_rect, (x, y), (x + 10, y + 10), (0, 0, 255), 1)
        img_rect = cv2.putText(img_rect, str(predict_result_sorted[i][0][1]), (x, y), font, 0.5, (255, 255, 255), 1)

print('predict done', time.time() - start_time2)
print('total:', total)

cv2.imwrite('backbone/pred.jpg', img_rect)
