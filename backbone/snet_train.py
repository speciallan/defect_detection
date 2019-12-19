#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import sys
sys.path.append('../..')

from keras import layers
from keras import optimizers
from keras import models
from keras.applications import VGG16
from keras.preprocessing import image
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from taurus_cv.models.resnet.snet import snet

train_dir = '/home/speciallan/Documents/python/data/VOCdevkit/zazhi/ClfImages/train'
validation_dir = '/home/speciallan/Documents/python/data/VOCdevkit/zazhi/ClfImages/valid'

input = layers.Input(shape=(10,10,3))
model = snet(input, classes_num=2)

model.summary()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(10, 10),
    batch_size=16,
    class_mode='categorical',
    shuffle=True,
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(10, 10),
    class_mode='categorical',
    batch_size=20
)
print(train_generator.class_indices)

model.compile(
    optimizer=optimizers.rmsprop(lr=0.001),
    loss='binary_crossentropy',
    metrics=['acc']
)
history = model.fit_generator(train_generator,
                              steps_per_epoch=10,
                              epochs=100,
                              validation_data=validation_generator,
                              validation_steps=10
                              )
model.save('backbone/snet.h5')

