#!/usr/bin/env python
#coding=utf-8

# Copyright (c) 2017 Guo Xiaolu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

############################
## How to use
############################
# This file is used for objects classification

############################
## How to use
############################
#

import os
os.environ['KERAS_BACKEND']='tensorflow'
import keras.backend as K
K.set_image_dim_ordering('tf')
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from common import processing_function
from common import InceptionV3_model, InceptionV3_deform_model

train_img_path = './train'
val_img_path = './val'
categories = './categories.txt'

f = open(categories, mode='r')
classes = [line.strip() for line in f.readlines()]

nb_classes = len(classes)
image_size = (224, 224)
batch_size = 32
nb_epoch = 100

model = InceptionV3_deform_model(nb_classes)
model.summary()

# datagenerator
gen = ImageDataGenerator(preprocessing_function=processing_function)
train_generator = gen.flow_from_directory(train_img_path, target_size=image_size, classes=classes, shuffle=True,
                                          batch_size=batch_size)

val_generator = gen.flow_from_directory(val_img_path, target_size=image_size, classes=classes, shuffle=True,
                                          batch_size=batch_size)

sgd = SGD(lr=0.1, momentum=0.9, decay=1e-5, nesterov=False)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
mc = ModelCheckpoint('./models/weights.{epoch:05d}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
model.fit_generator(train_generator, train_generator.classes.size/batch_size, nb_epoch=nb_epoch, callbacks=[tb, mc], validation_data=val_generator, validation_steps=val_generator.classes.size/batch_size, initial_epoch=0)

# evaluate
score = model.evaluate_generator(val_generator, val_generator.classes.size/batch_size)
print(score)
