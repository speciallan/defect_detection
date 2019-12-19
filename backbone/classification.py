from keras import layers
from keras import optimizers
from keras import models
from keras.applications import VGG16
from keras.preprocessing import image
import os
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
train_dir = '/home/mlg1504/whg/dif_train'
validation_dir = '/home/mlg1504/whg/dif_validation'
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

conv_base.summary()
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
conv_base.trainable = False

# set_trainable = False
# for layer in conv_base.layers:
#     if layer.name == 'res3a_branch2a':
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1./255)
augmented_dir = os.path.join('./', 'augmented_dir')
if not os.path.exists(augmented_dir):
    os.mkdir(augmented_dir)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical',
    shuffle=True,
    # save_to_dir=augmented_dir,
    # save_prefix='augmented',
    # save_format='jpg'

)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    class_mode='categorical',
    batch_size=20
)
print(train_generator.class_indices)
i = 0
j = 0
augmented = os.path.join('./', 'augmented')
if not os.path.exists(augmented):
    os.mkdir(augmented)


# x,y = next(train_generator)
# print(np.array(x).shape, len(y))
# exit()



for batch_imgs, batch_labels in train_generator:
    print(batch_labels)
    # for x in batch_imgs:
    #     x = np.array(x)
    #     image.save_img(augmented+str(j)+'.jpg',x)
    #     j += 1
    #     print()
    i += 1
    if i > 3:
        break
model.compile(
    optimizer=optimizers.rmsprop(lr=2e-5),
    loss='categorical_crossentropy',
    metrics=['acc']
)
history = model.fit_generator(train_generator,
                              steps_per_epoch=10,
                              epochs=50,
                              validation_data=validation_generator,
                              validation_steps=10
                            )

model.save('classification.h5')
