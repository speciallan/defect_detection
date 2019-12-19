import sys
import argparse
sys.path.append('..')

import matplotlib as mpl

mpl.use('Agg')

import os
import time
import numpy as np
import cv2
from matplotlib import pyplot as plt

from defect_detection.env import set_runtime_environment
from defect_detection.model.pascal_voc import save_annotations
from defect_detection.model.image import read_image_bgr, preprocess_image, resize_image, read_image_rgb
from defect_detection.model.resnet import resnet_retinanet
from defect_detection.config import Config

set_runtime_environment()

start_time = time.time()

config = Config('configRetinaNet.json')

wname = 'BASE'
wpath = config.trained_weights_path
classes = config.classes

if config.type.startswith('resnet'):
    model, _ = resnet_retinanet(len(classes), backbone=config.type, weights='imagenet', nms=True, config=config)
else:
    model = None
    print("模型 ({})".format(config.type))
    exit(1)

print("backend: ", config.type)

weights = './h5/result.h5'
model.load_weights(weights, by_name=True, skip_mismatch=True)
print('load weights {}'.format(weights))

start_index = config.test_start_index
font = cv2.FONT_HERSHEY_SIMPLEX

def random_color():
    import random
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r,g,b)

files = sorted(os.listdir(config.test_images_path))
for nimage, imgf in enumerate(files):

    # if imgf not in ['ship0201711030203001.jpg', 'ship0201606110201801.jpg', 'ship0201606110201902.jpg', 'ship02016061102012014.jpg', 'ship0201711030202902.jpg']:
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

        orig_image = read_image_rgb(imgfp)

        _, _, detections = model.predict_on_batch(np.expand_dims(img, axis=0))
        # print(detections[0][:5], detections[0][:5][4:])
        # exit()

        # bbox要取到边界内
        detections[:, :, 0] = np.maximum(0, detections[:, :, 0])
        detections[:, :, 1] = np.maximum(0, detections[:, :, 1])
        detections[:, :, 2] = np.minimum(img.shape[1], detections[:, :, 2])
        detections[:, :, 3] = np.minimum(img.shape[0], detections[:, :, 3])

        detections[0, :, :4] /= scale

        scores = detections[0, :, 4:]

        # 推测置信度 这里没用
        indices = np.where(detections[0, :, 4:] >= 0.05)

        scores = scores[indices]

        scores_sort = np.argsort(-scores)[:100]

        image_boxes = detections[0, indices[0][scores_sort], :4]
        image_scores = np.expand_dims(detections[0, indices[0][scores_sort], 4 + indices[1][scores_sort]], axis=1)
        image_detections = np.append(image_boxes, image_scores, axis=1)
        image_predicted_labels = indices[1][scores_sort]


        orig_image = cv2.imread(imgfp)
        show_img = orig_image.copy()

        # color = random_color()
        color = (0, 255, 255)

        # plt.gca().add_patch(plt.Rectangle(xy=(cat_dict['bbox'][i][1], cat_dict['bbox'][i][0]),
        #                                   width=cat_dict['bbox'][i][3] - cat_dict['bbox'][i][1],
        #                                   height=cat_dict['bbox'][i][2] - cat_dict['bbox'][i][0],
        #                                   edgecolor=[c / 255 for c in label_colors[cat_idx]],
        #                                   fill=False, linewidth=2))


        if len(image_boxes) > 0:
            for i, box in enumerate(image_boxes):
                xmin = int(box[0])
                ymin = int(box[1])
                xmax = int(box[2])
                ymax = int(box[3])
                # print(xmin, ymin, xmax, ymax)

                show_img = cv2.rectangle(show_img, (xmin, ymin), (xmax, ymax), color, 1)
                show_img = cv2.putText(show_img, '{} {:.2f}'.format(classes[image_predicted_labels[i]], image_scores[i][0]), (xmin, ymin-2), font, 0.5, color, 1)

        cv2.imwrite(os.path.join(config.test_result_path, imgf), show_img)
        # plt.savefig(os.path.join(config.test_result_path, imgf))
        # plt.close()

        print("生成图片 '" + imgf + "'" + ' time:{}, 目标框：{}'.format(time.time() - start_time, len(image_boxes)))
        start_time = time.time()

