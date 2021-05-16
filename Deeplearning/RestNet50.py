# Transfer learning on already trained ResNet50. Compile process.

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam

categories = 3

TRAIN_DIR = 'Data_Train/'
TEST_DIR = 'Data_Test/'
TARGETS = ["glaucoma", "normal", "other"]

model = ResNet50(include_top = False, weights = 'imagenet',input_shape=(256,256,3), classifier_activation='softmax',pooling='max')

# The fully connected top layer of ResNet50 is not to added in this model
flattened = Flatten()(model.output)
# All inputs and outputs are connected to neurons (Dense Layers)
# ReLu activation can be used here. Difference --?
fc1 = Dense(3, activation='softmax', name="AddedDense2")(flattened)

full_model = Model(inputs=model.input, outputs=fc1)
full_model.layers[0].trainable = False

adam = Adam(learning_rate=0.001)

full_model.compile(adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])

full_model.summary()

train_data = image_dataset_from_directory(
    TRAIN_DIR,
    batch_size=8,
    image_size=(256, 256),
    label_mode="categorical",
    shuffle=True,
    seed=0,
    validation_split=0.2,
    subset="training",
)

validate_data = image_dataset_from_directory(
    TRAIN_DIR,
    batch_size=8,
    image_size=(256, 256),
    label_mode="categorical",
    shuffle=True,
    seed=0,
    validation_split=0.2,
    subset="validation",
)

test_data = image_dataset_from_directory(
    TEST_DIR,
    batch_size=8,
    image_size=(256, 256),
    label_mode="categorical",
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
)

checkpoint_filepath = 'Model/resnet50_best.h5'
checkpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model_fitting = full_model.fit_generator(
        train_data,
        epochs = 20,
        validation_data=validate_data,
        callbacks=[checkpoint]
)

full_model.save('model_20ep.h5')

from shutil import copyfile
copyfile('model_20ep.h5', '../muteluh-fundus/model.h5')


import matplotlib.pyplot as plt

plt.plot(model_fitting.history['accuracy'], label='accuracy')
plt.plot(model_fitting.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig('resnet.jpg')

test_loss, test_acc = model.evaluate(test_data, verbose=2)

plt.savefig('resnet.jpg')

full_model.fit_generator(
        train_data,
        epochs = 10,
        validation_data=validate_data,
        callbacks=[checkpoint]
)

full_model.save('resnet50_test.h5')

import cv2
import numpy as np
from keras.models import load_model
import matplotlib.image as mpimg
import tensorflow as tf

img = cv2.imread('Data_Train/class_normal/normal_10_rotated_127.jpg')
class_names = ["glaucoma", "normal", "other"]
model = load_model('resnet50_test.h5')
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
img = cv2.resize(img,(256,256))
img = np.reshape(img,[1,256,256,3])
predictions = model.predict(img)
argmax = np.argmax(predictions > 0.5).astype("int32")
score = tf.nn.softmax(predictions[0])
print(argmax)

print(predictions)

full_model.fit_generator(train_data,
        epochs = 70,
        validation_data=validate_data,
        callbacks=[checkpoint])


full_model.save('resnet50_50ep.h5')

copy('resnet50_50ep.h5','../../muteluh-fundus/resnet50_50ep.h5')

from shutil import copyfile
copyfile('resnet50_50ep.h5', '../../muteluh-fundus/resnet50_50ep.h5')

import cv2
import numpy as np
from keras.models import load_model
import matplotlib.image as mpimg
import tensorflow as tf

img = cv2.imread('Data_Train/class_other/other_10_rotated_1.jpg')
class_names = ["glaucoma", "normal", "other"]
model = load_model('resnet50_50ep.h5')

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
img = cv2.resize(img,(256,256))
img = np.reshape(img,[1,256,256,3])
predictions = model.predict(img)
print(predictions)
argmax = np.argmax(predictions)
score = tf.nn.softmax(predictions[0])
print(argmax)

full_model.fit_generator(
        train_data,
        epochs = 10,
        validation_data=validate_data,
        callbacks=[checkpoint]
)