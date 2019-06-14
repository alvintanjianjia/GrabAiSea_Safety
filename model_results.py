from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
import numpy
import sklearn
from tensorflow import keras

# Helper libraries
#import numpy as np
#import matplotlib.pyplot as plt

print(tf.__version__)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from sklearn import metrics

config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
session = tf.Session(config=config)
train_data_dir = 'C:/Users/tanji/Desktop/safety/SpeedImages/Train'
validation_data_dir = 'C:/Users/tanji/Desktop/safety/SpeedImages/Test'
nb_train_samples = 20000
nb_validation_samples = 800
epochs = 10
batch_size = 16
img_width, img_height = 450, 300

test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
#test_steps_per_epoch = numpy.math.ceil(test_data_generator.samples / test_data_generator.batch_size)


model = load_model('model_150_50.h5')

    #model.compile(loss='binary_crossentropy',
    #             optimizer='rmsprop',
    #             metrics=['accuracy'])

predictions = model.predict_generator(validation_generator, steps=nb_validation_samples // batch_size)
# Get most likely class
predicted_classes = numpy.argmax(predictions, axis=1)


session.close()

true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())


report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)
