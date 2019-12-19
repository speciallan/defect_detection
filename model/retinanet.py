import keras
import numpy as np

from defect_detection.model.initializers import PriorProbability
from defect_detection.model.loss import smooth_l1, focal
from defect_detection.model.misc import UpsampleLike, RegressBoxes, NonMaximumSuppression, Anchors
from defect_detection.model.attention import *
from defect_detection.model.bilinear_upsampling import BilinearUpsampling

custom_objects = {
    'UpsampleLike': UpsampleLike,
    'PriorProbability': PriorProbability,
    'RegressBoxes': RegressBoxes,
    'NonMaximumSuppression': NonMaximumSuppression,
    'Anchors': Anchors,
    '_smooth_l1': smooth_l1(),
    '_focal': focal(),
}


def default_classification_model(num_classes,
                                 num_anchors,
                                 pyramid_feature_size=256,
                                 prior_probability=0.01,
                                 classification_feature_size=256,
                                 name='classification_submodel'):
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
    }

    inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.zeros(),
        bias_initializer=PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_regression_model(num_anchors,
                             pyramid_feature_size=256,
                             regression_feature_size=256,
                             name='regression_submodel'):
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros'
    }

    inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(num_anchors * 4, name='pyramid_regression', **options)(outputs)
    outputs = keras.layers.Reshape((-1, 4), name='pyramid_regression_reshape')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def __create_pyramid_features(C2, C3, C4, C5, feature_size=256, with_ca=False):

    P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='P5')(C5)
    P5_upsampled = UpsampleLike(name='P5_upsampled')([P5, C4])

    P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4 = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)
    P4_upsampled = UpsampleLike(name='P4_upsampled')([P4, C3])

    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    P3_upsampled = UpsampleLike(name='P3_upsampled')([P3, C2])
    P2 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C2_reduced')(C2)
    P2 = keras.layers.Add(name='P2_merged')([P3_upsampled, P2])
    P2 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P2')(P2)

    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    if with_ca:
        P2 = ChannelWiseAttention(P2, name='p2_ca')
        P3 = ChannelWiseAttention(P3, name='p3_ca')
        P4 = ChannelWiseAttention(P4, name='p4_ca')
        P5 = ChannelWiseAttention(P5, name='p5_ca')
        P6 = ChannelWiseAttention(P6, name='p6_ca')

    # c1 2 3 4 5
    return P2, P3, P4, P5, P6


class AnchorParameters:
    def __init__(self, sizes, strides, ratios, scales):
        self.sizes = sizes
        self.strides = strides
        self.ratios = ratios
        self.scales = scales

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


AnchorParameters.default = AnchorParameters(
    # sizes=[32, 64, 128, 256, 512],
    # strides=[8, 16, 32, 64, 128],
    # sizes是步长的4倍
    # sizes=[16, 32, 64, 128, 256],
    # sizes =[8, 16, 32, 64, 128],
    # strides=[4, 8, 16, 32, 64],
    sizes=[4, 8, 16, 32, 64],
    strides=[2, 4, 8, 16, 32],
    # ratios=np.array([0.5, 1, 2], keras.backend.floatx()),
    ratios=np.array([0.25, 0.5, 1, 2, 4], keras.backend.floatx()),
    scales=np.array([0.5, 2 ** 0, 2 ** 1], keras.backend.floatx())
    # scales=np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
)


def default_submodels(num_classes, anchor_parameters):
    return [
        ('regression', default_regression_model(anchor_parameters.num_anchors())),
        ('classification', default_classification_model(num_classes, anchor_parameters.num_anchors()))
    ]


def __build_model_pyramid(name, model, features):
    return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])


def __build_pyramid(models, features):
    return [__build_model_pyramid(n, m, features) for n, m in models]


def __build_anchors(anchor_parameters, features):
    anchors = []
    for i, f in enumerate(features):
        anchors.append(Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f))
    return keras.layers.Concatenate(axis=1)(anchors)


def retinanet(
        inputs,
        backbone_outputs,
        num_classes,
        anchor_parameters=AnchorParameters.default,
        create_pyramid_features=__create_pyramid_features,
        submodels=None,
        name='retinanet'
):
    if submodels is None:
        submodels = default_submodels(num_classes, anchor_parameters)

    C2, C3, C4, C5 = backbone_outputs

    features = create_pyramid_features(C2, C3, C4, C5, with_ca=False)
    # print(features)
    # exit()

    pyramid = __build_pyramid(submodels, features)
    anchors = __build_anchors(anchor_parameters, features)

    return keras.models.Model(inputs=inputs, outputs=[anchors] + pyramid, name=name)


def retinanet_bbox(inputs, num_classes, backbone_outputs, nms=True, name='retinanet-bbox', *args, **kwargs):
    model = retinanet(inputs=inputs, backbone_outputs=backbone_outputs, num_classes=num_classes, *args, **kwargs)

    # [batch_size, ?, 4]
    anchors = model.outputs[0]
    regression = model.outputs[1]
    classification = model.outputs[2]

    boxes = RegressBoxes(name='boxes')([anchors, regression])
    detections = keras.layers.Concatenate(axis=2)([boxes, classification] + model.outputs[3:])

    if nms:
        # 多个候选框之间 iou<0.5 则保留
        detections = NonMaximumSuppression(name='nms', top_k=None, max_boxes=300, nms_threshold=0.1, score_threshold=0.2)([boxes, classification, detections])

    return keras.models.Model(inputs=inputs, outputs=model.outputs[1:] + [detections], name=name)
