import tensorflow


def top_k(*args, **kwargs):
    return tensorflow.nn.top_k(*args, **kwargs)


def resize_images(*args, **kwargs):
    return tensorflow.image.resize_images(*args, **kwargs)


def non_max_suppression(*args, **kwargs):
    return tensorflow.image.non_max_suppression(*args, **kwargs)


def range(*args, **kwargs):
    return tensorflow.range(*args, **kwargs)


def gather_nd(*args, **kwargs):
    return tensorflow.gather_nd(*args, **kwargs)


def meshgrid(*args, **kwargs):
    return tensorflow.meshgrid(*args, **kwargs)


def where(*args, **kwargs):
    return tensorflow.where(*args, **kwargs)
