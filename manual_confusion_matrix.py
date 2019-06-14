from __future__ import absolute_import, division, print_function
import cv2
import numpy as np
# TensorFlow and tf.keras
import tensorflow as tf
import numpy
import sklearn
from tensorflow import keras

# Helper libraries
#import numpy as np
#import matplotlib.pyplot as plt

print(tf.__version__)
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from sklearn import metrics

train_data_dir = 'C:/Users/tanji/Desktop/safety/SpeedImages/Train'
validation_data_dir = 'C:/Users/tanji/Desktop/safety/SpeedImages/Test'

def predict_from_spectogram(model_path='model_150_50.h5', img_path='C:/Users/tanji/Desktop/safety/SpeedImages/Test/Safe/0.png'):
    model = load_model(model_path)

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    img = cv2.imread(img_path)
    img = cv2.resize(img, (450, 300))
    img = np.reshape(img, [1, 450, 300, 3])

    classes = model.predict_classes(img)
    if classes[0][0] == 1:
        print('Crazy Driver')
        return 'Crazy Driver'
    else:
        print('Normal Driver')
        return 'Normal Driver'


def predict_from_model(model, img_path):

    img = cv2.imread(img_path)
    img = cv2.resize(img, (450, 300))
    img = np.reshape(img, [1, 450, 300, 3])

    classes = model.predict_classes(img)
    if classes[0][0] == 1:
        print('Crazy Driver')
        return 'Crazy Driver'
    else:
        print('Normal Driver')
        return 'Normal Driver'

def get_confusion_matrix(model_path='model_150_50.h5'):
    # [(Yes Yes), (Yes No), (No Yes), (No No)]
    mat = [0,0,0,0]
    model = load_model(model_path)

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    for x in os.listdir('C:/Users/tanji/Desktop/safety/SpeedImages/Test/Safe/'):
        dir = 'C:/Users/tanji/Desktop/safety/SpeedImages/Test/Safe/' + x
        print(dir)
        ans = predict_from_model(model, dir)
        if ans == 'Normal Driver':
            mat[0] += 1
        else:
            mat[1] += 1

    for x in os.listdir('C:/Users/tanji/Desktop/safety/SpeedImages/Test/Unsafe/'):
        dir = 'C:/Users/tanji/Desktop/safety/SpeedImages/Test/Unsafe/' + x
        print(dir)
        ans = predict_from_model(model, dir)
        if ans == 'Crazy Driver':
            mat[3] += 1
        else:
            mat[2] += 1

    return mat



mat = get_confusion_matrix()
print(mat)


#for x in os.listdir('C:/Users/tanji/Desktop/safety/SpeedImages/Test/Safe/'):
    #dir = 'C:/Users/tanji/Desktop/safety/SpeedImages/Test/Safe/' + x
    #print(dir)
#test = os.listdir('C:/Users/tanji/Desktop/safety/SpeedImages/Test/Safe/')






