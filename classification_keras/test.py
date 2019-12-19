import os
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from common import processing_function, write2html
import numpy as np
import cv2

val_img_path = '/Users/Lavector/dataset/plant_disease/val'
categories = '/Users/Lavector/dataset/plant_disease/categories.txt'
weight_path = './models/weights.00056.hdf5'

f = open(categories, mode='r')
classes = [line.strip() for line in f.readlines()]

nb_classes = len(classes)
image_size = (299, 299)
batch_size = 1

model = load_model(weight_path)

# gen = ImageDataGenerator(preprocessing_function=processing_function)
# val_generator = gen.flow_from_directory(val_img_path, target_size=image_size, classes=classes, shuffle=True,
#                                           batch_size=batch_size)

# score = model.evaluate_generator(val_generator, val_generator.classes.size/batch_size)
# print score

result_path = './result.txt'
if os.path.isfile(result_path):
    os.remove(result_path)
f = open(result_path, mode='a')
labels = os.listdir(val_img_path)
for label in tqdm(labels):
    label_path = os.path.join(val_img_path, label)
    if not os.path.isdir(label_path):
        continue

    label_imgs = os.listdir(label_path)
    for img in label_imgs:
        if img[0] == '.':
            continue
        img_path = os.path.join(label_path, img)
        src = img_to_array(load_img(img_path))
        src = cv2.resize(src, image_size)
        src_process = processing_function(src)
        src_process = np.expand_dims(src_process,axis=0)
        result = model.predict(src_process)
        idx = np.argmax(result)
        score = result[0,idx]
        output = '%s\t%s\t%.3f\t%s\t%s\n'%(label, classes[idx], score, label==classes[idx], img_path)
        print output
        f.write(output)
f.close()

html_path = './display.html'
write2html(result_path, html_path)