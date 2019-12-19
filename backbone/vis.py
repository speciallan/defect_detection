#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import sys
sys.path.append('../..')

import numpy as np

import cv2

from keras.models import Model, Input
from keras import backend as K
from taurus_cv.models.resnet.resnet import resnet50
from taurus_cv.models.resnet.snet import snet
import matplotlib.pyplot as plt
import keras_resnet.models
from taurus_projects.defect_detection.config import current_config as config

import utils
import model


def conv_output(model, layer_name, img):
    """Get the output of conv layer.
    Args:
           model: keras model.
           layer_name: name of layer in the model.
           img: processed input image.
    Returns:
           intermediate_output: feature map.
    """
    # this is the placeholder for the input images
    input_img = model.input

    try:
        # this is the placeholder for the conv output
        out_conv = model.get_layer(layer_name).output
    except:
        raise Exception('Not layer named {}!'.format(layer_name))

    # get the intermediate layer model
    intermediate_layer_model = Model(inputs=input_img, outputs=out_conv)

    # get the output of intermediate layer model
    intermediate_output = intermediate_layer_model.predict(img)

    return intermediate_output[0]


def conv_filter(model, layer_name, img):
    """Get the filter of conv layer.
    Args:
           model: keras model.
           layer_name: name of layer in the model.
           img: processed input image.
    Returns:
           filters.
    """
    # this is the placeholder for the input images
    input_img = model.input

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    try:
        layer_output = layer_dict[layer_name].output
    except:
        raise Exception('Not layer named {}!'.format(layer_name))

    kept_filters = []
    for i in range(layer_output.shape[-1]):
        loss = K.mean(layer_output[:, :, :, i])
        # compute the gradient of the input picture with this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = utils.normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.
        # run gradient ascent for 20 steps
        fimg = img.copy()

        for j in range(40):
            loss_value, grads_value = iterate([fimg])
            fimg += grads_value * step

        # decode the resulting input image
        fimg = utils.deprocess_image(fimg[0])
        kept_filters.append((fimg, loss_value))

        # sort filter result
        kept_filters.sort(key=lambda x: x[1], reverse=True)

    return np.array([f[0] for f in kept_filters])


def output_heatmap(model, last_conv_layer, img):
    """Get the heatmap for image.
    Args:
           model: keras model.
           last_conv_layer: name of last conv layer in the model.
           img: processed input image.
    Returns:
           heatmap: heatmap.
    """
    # predict the image class
    preds = model.predict(img)
    # find the class index
    index = np.argmax(preds[0])
    # This is the entry in the prediction vector
    target_output = model.output[:, index]

    # get the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer)

    # compute the gradient of the output feature map with this target class
    grads = K.gradients(target_output, last_conv_layer.output)[0]

    # mean the gradient over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # this function returns the output of last_conv_layer and grads
    # given the input picture
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img])

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the target class

    for i in range(conv_layer_output_value.shape[-1]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap


if __name__ == '__main__':

    # img_path = 'backbone/images/ship1.png'
    img_path = 'backbone/images/ship02018010902014021.jpg'

    layer_name = 'res5c_relu'
    last_conv_layer = 'res5c_relu'

    # model, preprocess_input = model.get_model('vgg16')
    # img, img_resized = utils.read_img(img_path, preprocess_input, (224, 224))

    shape = (224,224)
    input = Input((shape[0],shape[1],3))
    model = keras_resnet.models.ResNet50(input, include_top=False, freeze_bn=True)
    # model = snet(input)
    config.pretrained_weights = './h5/result.h5'
    # print(config.pretrained_weights)
    model.load_weights(config.pretrained_weights, by_name=True)

    # for k,v in enumerate(model.layers):
    #     print(v.name)
    # model.summary()
    # exit()

    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, shape)
    img_batch = np.expand_dims(img_resized, axis=0)

    # layer_names = ['snet_conv1', 'snet_conv2', 'snet_conv3', 'snet_conv4', 'snet_conv5', 'snet_conv6', 'snet_pool']
    layer_names = ['conv1_relu', 'res2c_relu', 'res3d_relu', 'res4f_relu', 'res5c_relu']

    # plt.figure()
    show = cv2.resize(img, shape)
    for i in range(len(layer_names)):
        cout = conv_output(model, layer_names[i], img_batch)
        # plt.subplot(5,1,i+1)
        # plt.imshow(cv2.resize(cout[0], shape))
        utils.vis_conv(cout, 4, layer_names[i], 'conv')
        # show = np.vstack([show, cout])

    print('可视化conv {}完毕'.format(layer_names))

    pimg = np.random.random((1, shape[0], shape[1], 3)) * 20 + 128.

    # fout = conv_filter(model, layer_name, pimg)
    # utils.vis_conv(fout, 8, layer_name, 'filter')
    # print('可视化filter {}完毕'.format(layer_name))

    heatmap = output_heatmap(model, last_conv_layer, pimg)
    utils.vis_heatmap(img_batch, heatmap)
    print('可视化headmap完毕')
