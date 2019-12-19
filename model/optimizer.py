import keras


def get_optimizer(base_lr):
    # return keras.optimizers.adam(lr=base_lr, clipnorm=0.001)
    return keras.optimizers.sgd(lr=base_lr, momentum=0.9)
