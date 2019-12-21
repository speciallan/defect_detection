import sys
import keras
import keras_resnet
import keras_resnet.models

sys.path.append('../../..')

from defect_detection.model.retinanet import custom_objects, retinanet_bbox
from defect_detection.model.layers import ConvOffset2D
from keras.utils import get_file
from keras.layers import *
from keras import Model, layers, backend
from keras.applications.densenet import DenseNet121
# from keras.applications import imagenet_utils
# from keras.applications.imagenet_utils import imagenet_utils

resnet_filename = 'ResNet-{}-model.keras.h5'
resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)

custom_objects = custom_objects.copy()
custom_objects.update(keras_resnet.custom_objects)


def download_imagenet(backbone):
    filename = resnet_filename.format(backbone[6:])
    resource = resnet_resource.format(backbone[6:])
    if backbone == 'resnet50':
        checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
    elif backbone == 'resnet101':
        checksum = '05dc86924389e5b401a9ea0348a3213c'
    elif backbone == 'resnet152':
        checksum = '6ee11ef2b135592f8031058820bb9e71'
    else:
        raise ValueError("Il backbone '{}' non è riconosciuto.".format(backbone))

    return get_file(
        filename,
        resource,
        cache_subdir='models',
        md5_hash=checksum
    )


def resnet_retinanet(num_classes, backbone='resnet50', inputs=None, weights='imagenet', config=None, **kwargs):
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(shape=(config.img_min_size, config.img_max_size, 3))

    # determine which weights to load
    # if weights == 'imagenet':
    #     weights_path = download_imagenet(backbone)
    # else:
    #     weights_path = None
    # print(weights_path)
    # elif weights is None:
    #     weights_path = None
    # else:
    #     weights_path = weights
    # weights_path = './h5/result_snetplus.h5'

    # create the resnet backbone
    # splus > snet > snetpp
    # resnet = resnet50(inputs, is_extractor=True)
    # resnet = snet(inputs, is_extractor=True)
    model = DenseNet121(input_tensor=inputs, classes=num_classes, include_top=False, weights='imagenet')
    # resnet = keras_resnet.models.ResNet50(inputs, include_top=False, freeze_bn=True)
    # resnet = keras_resnet.models.ResNet18(inputs, include_top=False, freeze_bn=True)
    # resnet = resnet4(inputs, is_extractor=True)
    # resnet = resnetplus(inputs, is_extractor=True)
    # resnet = snetplus(inputs, is_extractor=True)
    # resnet = vgg16(inputs, is_extractor=True)
    # resnet = fastnet(inputs, is_extractor=True)
    # resnet = snetpp(inputs, is_extractor=True)

    # from taurus_cv.models.resnet.resnet import resnet50_fpn
    # resnet = resnet50_fpn(inputs)
    # resnet.summary()
    # print(resnet.outputs)
    # exit()

    # 生成完整模型
    retinanet_model = retinanet_bbox(inputs=inputs, num_classes=num_classes, backbone_outputs=model.outputs[0:4], **kwargs)

    # optionally load weights
    # if weights_path:
    #     model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    #     print("BACKEND")

    return retinanet_model, model.layers


def snet(input, classes_num=2, is_extractor=False, output_layer_name='snet_pool'):

    x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv1_1')(input)
    # x = ConvOffset2D(32, name='conv1_offset')(x)
    x = MaxPooling2D((2, 2), name='snet_pool1')(x)
    c1 = x

    x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv2_1')(x)
    # x = ConvOffset2D(64, name='conv2_offset')(x)
    x = MaxPooling2D((2, 2), name='snet_pool2')(x)
    c2 = x

    x = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv3_1')(x)
    # x = ConvOffset2D(128, name='conv3_offset')(x)
    x = MaxPooling2D((2, 2), name='snet_pool3')(x)
    c3 = x

    x = Conv2D(256, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv4_1')(x)
    # x = ConvOffset2D(256, name='conv4_offset')(x)
    # x = Conv2D(256, (3, 3), strides=2, activation='relu', padding='same', name='snet_conv4_2')(x)
    # x = Conv2D(256, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv4_3')(x)
    # x = BatchNormalization(name='conv4_bn')(x)
    # x = AveragePooling2D((2, 2), name='snet_pool4')(x)
    x = MaxPooling2D((2, 2), name='snet_pool4')(x)
    c4 = x

    if is_extractor:
        outputs = [c1, c2, c3, c4]
        model = Model(input, outputs=outputs)
        return model


def resnet4(input, classes_num=2, is_extractor=False, output_layer_name='snet_pool'):

    bn_axis = 3

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input)

    # conv1 [64]*1
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    c1 = x

    # 池化
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # conv2 [64,64,256]*3
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    c2 = x

    # conv3 [128,128,512]*4
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    c3 = x

    # conv4 [256,256,1024]*6
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    c4 = x

    if is_extractor:
        outputs = [c1, c2, c3, c4]
        model = Model(input, outputs=outputs)
        return model

def resnetplus(input, classes_num=2, is_extractor=False, output_layer_name='snet_pool'):

    # 第一层尺寸大
    x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv_a1')(input)
    x = Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', name='snet_conv_a2')(x)
    # x = MaxPooling2D((2, 2), name='snet_pool1')(x)
    # (128,128,32)
    conv1 = x

    b1 = Conv2D(64, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_b1')(x)
    b2 = AveragePooling2D(strides=(1, 1), padding='same')(x)
    b2 = Conv2D(64, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_b2')(b2)
    b3 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv_b3')(x)
    # b3 = ConvOffset2D(64, name='conv2_offset')(b3)
    x = Concatenate(axis=-1)([b1, b2, b3])
    x = Conv2D(64, (1, 1), strides=1, activation='relu', padding='same')(x)

    # res
    shortcut2 = Conv2D(64, (1, 1), padding='same')(conv1)
    x = Add()([shortcut2, x])
    x = Activation(activation='relu')(x)

    x = MaxPooling2D((2, 2), name='snet_pool2')(x)
    conv2 = x

    c1 = Conv2D(128, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_c1')(x)
    c2 = AveragePooling2D(strides=(1, 1), padding='same')(x)
    c2 = Conv2D(128, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_c2')(c2)
    c3 = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv_c3')(x)
    # c3 = ConvOffset2D(128, name='conv3_offset')(c3)
    x = Concatenate(axis=-1)([c1, c2, c3])
    x = Conv2D(128, (1, 1), strides=1, activation='relu', padding='same')(x)

    shortcut3 = Conv2D(128, (1, 1), padding='same')(conv2)
    x = Add()([shortcut3, x])
    x = Activation(activation='relu')(x)

    x = MaxPooling2D((2, 2), name='snet_pool3')(x)
    conv3 = x

    d1 = Conv2D(256, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_d1')(x)
    d2 = AveragePooling2D(strides=(1, 1), padding='same')(x)
    d2 = Conv2D(256, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_d2')(d2)
    d3 = Conv2D(256, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv_d3')(x)
    # d3 = ConvOffset2D(256, name='conv4_offset')(d3)
    x = Concatenate(axis=-1)([d1, d2, d3])
    x = Conv2D(256, (1, 1), strides=1, activation='relu', padding='same')(x)

    shortcut4 = Conv2D(256, (1, 1), padding='same')(conv3)
    x = Add()([shortcut4, x])
    x = Activation(activation='relu')(x)

    x = MaxPooling2D((2, 2), name='snet_pool4')(x)
    conv4 = x

    if is_extractor:
        outputs = [conv1, conv2, conv3, conv4]
        model = Model(input, outputs=outputs)
        model.summary()
        return model

def fastnet(input, classes_num=2, is_extractor=False):

    x = Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', name='snet_conv1_1')(input)
    c1 = x

    x = Conv2D(64, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv2_1')(x)
    x = Conv2D(64, (3, 3), strides=2, activation='relu', padding='same', name='snet_conv2_2')(x)
    c2 = x

    x = Conv2D(128, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv3_1')(x)
    x = Conv2D(128, (3, 3), strides=2, activation='relu', padding='same', name='snet_conv3_2')(x)
    c3 = x

    x = Conv2D(256, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv4_1')(x)
    x = Conv2D(256, (3, 3), strides=2, activation='relu', padding='same', name='snet_conv4_2')(x)
    c4 = x

    if is_extractor:
        outputs = [c1, c2, c3, c4]
        model = Model(input, outputs=outputs)
        return model

    else:

        # x = MaxPooling2D()(x)
        x = Flatten()(x)
        x = Dense(classes_num, activation='softmax', kernel_initializer='normal')(x)
        model = Model(input, x)

        return model


def snetplus(input, classes_num=2, is_extractor=False, output_layer_name='snet_pool'):

    # 第一层尺寸大
    x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv_a1')(input)
    x = MaxPooling2D((2, 2), name='snet_pool1')(x)
    conv1 = x

    b1 = Conv2D(64, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_b1')(x)
    b2 = AveragePooling2D(strides=(1, 1), padding='same')(x)
    b2 = Conv2D(64, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_b2')(b2)
    b3 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv_b3')(x)
    # b3 = ConvOffset2D(64, name='conv2_offset')(b3)
    x = Concatenate(axis=-1)([b1, b2, b3])
    x = Conv2D(64, (1, 1), strides=1, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), name='snet_pool2')(x)
    conv2 = x

    c1 = Conv2D(128, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_c1')(x)
    c2 = AveragePooling2D(strides=(1, 1), padding='same')(x)
    c2 = Conv2D(128, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_c2')(c2)
    c3 = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv_c3')(x)
    # c3 = ConvOffset2D(128, name='conv3_offset')(c3)
    x = Concatenate(axis=-1)([c1, c2, c3])
    x = Conv2D(128, (1, 1), strides=1, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), name='snet_pool3')(x)
    conv3 = x

    d1 = Conv2D(256, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_d1')(x)
    d2 = AveragePooling2D(strides=(1, 1), padding='same')(x)
    d2 = Conv2D(256, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_d2')(d2)
    d3 = Conv2D(256, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv_d3')(x)
    # d3 = ConvOffset2D(256, name='conv4_offset')(d3)
    x = Concatenate(axis=-1)([d1, d2, d3])
    x = Conv2D(256, (1, 1), strides=1, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), name='snet_pool4')(x)
    conv4 = x

    if is_extractor:
        outputs = [conv1, conv2, conv3, conv4]
        model = Model(input, outputs=outputs)
        model.summary()
        return model

def snetpp(input, classes_num=2, is_extractor=False, output_layer_name='snet_pool'):

    # a1 = Conv2D(32, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_a1')(input)
    # a2 = AveragePooling2D(strides=(1, 1), padding='same')(input)
    # a2 = Conv2D(32, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_a2')(a2)
    # a3 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv_a3')(input)
    # a3 = ConvOffset2D(32, name='conv1_offset')(a3)
    # x = Concatenate(axis=-1)([a1, a2, a3])
    # x = Conv2D(32, (1, 1), strides=1, activation='relu', padding='same')(x)

    # 第一层尺寸大
    x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv_a1')(input)
    x = MaxPooling2D((2, 2), name='snet_pool1')(x)
    conv1 = x

    b1 = Conv2D(64, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_b1')(x)
    b2 = AveragePooling2D(strides=(1, 1), padding='same')(x)
    b2 = Conv2D(64, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_b2')(b2)
    b3 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv_b3')(x)
    # b3 = ConvOffset2D(64, name='conv2_offset')(b3)
    b4 = Conv2D(64, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_b4_1')(x)
    b4 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv_b4_2')(b4)
    b4 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv_b4_3')(b4)
    x = Concatenate(axis=-1)([b1, b2, b3, b4])
    x = Conv2D(64, (1, 1), strides=1, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), name='snet_pool2')(x)
    conv2 = x

    c1 = Conv2D(128, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_c1')(x)
    c2 = AveragePooling2D(strides=(1, 1), padding='same')(x)
    c2 = Conv2D(128, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_c2')(c2)
    c3 = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv_c3')(x)
    # c3 = ConvOffset2D(128, name='conv3_offset')(c3)
    c4 = Conv2D(128, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_c4_1')(x)
    c4 = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv_c4_2')(c4)
    c4 = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv_c4_3')(c4)
    x = Concatenate(axis=-1)([c1, c2, c3, c4])
    x = Conv2D(128, (1, 1), strides=1, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), name='snet_pool3')(x)
    conv3 = x

    d1 = Conv2D(256, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_d1')(x)
    d2 = AveragePooling2D(strides=(1, 1), padding='same')(x)
    d2 = Conv2D(256, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_d2')(d2)
    d3 = Conv2D(256, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv_d3')(x)
    # d3 = ConvOffset2D(256, name='conv4_offset')(d3)
    d4 = Conv2D(256, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_d4_1')(x)
    d4 = Conv2D(256, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv_d4_2')(d4)
    d4 = Conv2D(256, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv_d4_3')(d4)
    x = Concatenate(axis=-1)([d1, d2, d3, d4])
    x = Conv2D(256, (1, 1), strides=1, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), name='snet_pool4')(x)
    conv4 = x

    if is_extractor:
        outputs = [conv1, conv2, conv3, conv4]
        model = Model(input, outputs=outputs)
        model.summary()
        return model

    else:

        # x = MaxPooling2D()(x)
        x = Flatten()(x)
        x = Dense(classes_num, activation='softmax', kernel_initializer='normal')(x)
        model = Model(input, x)

        return model


def vgg16(input, is_extractor=False):
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    c1 = x

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    c2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    c3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    c4 = x

    if is_extractor:
        outputs = [c1, c2, c3, c4]
        model = Model(input, outputs=outputs)
        return model

def resnet50(input, classes_num=1000, layer_num=50, is_extractor=False, output_layer_name = None, is_transfer_learning=False):
    """
    ResNet50
    :param input: 输入Keras.Input
    :param is_extractor: 是否用于特征提取
    :param layer_num: 可选40、50，40用于训练frcnn的时候速度过慢的问题
    :return:
    """

    bn_axis = 3

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input)

    # conv1 [64]*1
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    c1 = x

    # 池化
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    c2 = x

    # conv2 [64,64,256]*3
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    c3 = x

    # conv3 [128,128,512]*4
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    c4 = x

    # conv4 [256,256,1024]*6
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    c5 = x

    # conv5 [512,512,2048]*3
    if layer_num == 50:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
        c6 = x

    # 确定fine-turning层
    # outputs = [c1, c2, c3, c4, c5]
    outputs = [c1, c2, c3, c4]
    model = Model(input, outputs=outputs)
    return model

    # 用作特征提取器做迁移学习
    if is_extractor:

        # 冻结参数，停止学习
        for l in no_train_model.layers:
            l.trainable = False

            # if isinstance(l, layers.BatchNormalization):
            #     l.trainable = True
            # else:
            #     l.trainable = False

        if output_layer_name:
            return no_train_model.get_layer(output_layer_name).output

        return x

    elif is_transfer_learning:

        x = layers.AveragePooling2D()(x)
        x = layers.Flatten()(x)

        preds = layers.Dense(units=classes_num, activation='softmax', kernel_initializer='he_normal')(x)

        model = Model(input, preds, name='resnet50')

        # 3 4 6 3=16 * 3 前2个block 21层冻结
        for layer in model.layers[:21]:
            layer.trainable = False

        return model


    # 完整CNN模型
    else:

        # x = layers.MaxPooling2D(pool_size=(7, 7))(x)
        x = layers.AveragePooling2D()(x)
        x = layers.Flatten()(x)

        preds = layers.Dense(units=classes_num, activation='softmax', kernel_initializer='he_normal')(x)

        return Model(input, preds, name='resnet50')




def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    残差连接
    :param input_tensor: 输入张量
    :param kernel_size: 卷积核大小
    :param filters: 卷积核个数
    :param stage: 阶段标记
    :param block: 生成层名字
    :return: Tensor
    """

    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)

    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """
    膨胀卷积

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)

    shortcut = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)

    return x


if __name__ == '__main__':

    a = np.ones((64,64,64))
    b = np.ones((128,128,64))

