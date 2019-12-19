import keras
import numpy as np

from defect_detection.model.anchors import generate_anchors
from defect_detection.model.common import shift, bbox_transform_inv
from defect_detection.model.tensorflow_backend import top_k, non_max_suppression, resize_images


class Anchors(keras.layers.Layer):

    def __init__(self, size, stride, ratios=None, scales=None, *args, **kwargs):

        self.size = size
        self.stride = stride
        self.ratios = ratios
        self.scales = scales

        if ratios is None:
            self.ratios = np.array([0.5, 1, 2], keras.backend.floatx())
        elif isinstance(ratios, list):
            self.ratios = np.array(ratios)

        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx())
        elif isinstance(scales, list):
            self.scales = np.array(scales)

        self.num_anchors = len(ratios) * len(scales)
        self.anchors = keras.backend.variable(generate_anchors(
            base_size=size,
            ratios=ratios,
            scales=scales,
        ))

        super(Anchors, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):

        # FPN每一层的特征 (None, 64, 64, 256) -> 32,16,8,4 -> (None, 4, 4, 256)
        features = inputs
        features_shape = keras.backend.shape(features)[:3]

        # (?,4)
        anchorsShift = shift(features_shape[1:3], self.stride, self.anchors)

        # (1,?,4) -> (batch,?,4)
        anchors = keras.backend.tile(keras.backend.expand_dims(anchorsShift, axis=0), (features_shape[0], 1, 1))

        return anchors

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            total = np.prod(input_shape[1:3]) * self.num_anchors
            return (input_shape[0], total, 4)
        else:
            return (input_shape[0], None, 4)

    def get_config(self):
        config = super(Anchors, self).get_config()
        config.update({
            'size': self.size,
            'stride': self.stride,
            'ratios': self.ratios.tolist(),
            'scales': self.scales.tolist(),
        })

        return config


class NonMaximumSuppression(keras.layers.Layer):

    def __init__(self, nms_threshold=0.4, score_threshold=0.05, top_k=None, max_boxes=300, *args, **kwargs):
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.top_k = top_k
        self.max_boxes = max_boxes
        super(NonMaximumSuppression, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        boxes, classification, detections = inputs

        boxes = boxes[0]
        classification = classification[0]
        detections = detections[0]

        scores = keras.backend.max(classification, axis=1)

        if self.top_k:
            scores, indices = top_k(scores, self.top_k, sorted=False)
            boxes = keras.backend.gather(boxes, indices)
            classification = keras.backend.gather(classification, indices)
            detections = keras.backend.gather(detections, indices)

        # 只保留阈值以上的
        indices = non_max_suppression(boxes, scores, max_output_size=self.max_boxes, iou_threshold=self.nms_threshold, score_threshold=self.score_threshold)

        boxes = keras.backend.gather(boxes, indices)
        scores = keras.backend.gather(scores, indices)
        detections = keras.backend.gather(detections, indices)
        # print(boxes, scores, indices)
        # exit()

        # 从0.5iou中筛选出一个
        indices1 = non_max_suppression(boxes, scores, max_output_size=self.max_boxes, iou_threshold=0.05)

        detections = keras.backend.gather(detections, indices1)

        return keras.backend.expand_dims(detections, axis=0)

    def compute_output_shape(self, input_shape):
        return (input_shape[2][0], None, input_shape[2][2])

    def get_config(self):
        config = super(NonMaximumSuppression, self).get_config()
        config.update({
            'nms_threshold': self.nms_threshold,
            'top_k': self.top_k,
            'max_boxes': self.max_boxes,
        })

        return config


class UpsampleLike(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        return resize_images(source, (target_shape[1], target_shape[2]))

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


class RegressBoxes(keras.layers.Layer):
    def __init__(self, mean=None, std=None, *args, **kwargs):
        if mean is None:
            mean = np.array([0, 0, 0, 0])
        if std is None:
            std = np.array([0.1, 0.1, 0.2, 0.2])

        if isinstance(mean, (list, tuple)):
            mean = np.array(mean)
        elif not isinstance(mean, np.ndarray):
            raise ValueError('平均值必须是np数组: {}'.format(type(mean)))

        if isinstance(std, (list, tuple)):
            std = np.array(std)
        elif not isinstance(std, np.ndarray):
            raise ValueError('标准差必须是np数组:{}'.format(type(std)))

        self.mean = mean
        self.std = std
        super(RegressBoxes, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        anchors, regression = inputs
        return bbox_transform_inv(anchors, regression, mean=self.mean, std=self.std)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        return {
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
        }
