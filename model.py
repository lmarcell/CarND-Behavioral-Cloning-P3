#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import csv
import cv2
import numpy as np
import os
import random
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, AveragePooling2D, Cropping2D
from sklearn.model_selection import train_test_split

os.environ['KMP_DUPLICATE_LIB_OK']='True'

samples = []
images = []
measurements = []

def LoadMeasurements(dirs):
    '''
        Load the measurement from all listed directories
    '''
    for dir in dirs:
        LoadMeasurement(dir)

def LoadMeasurement(dir):
    '''
        Load measurement from one directory
    '''
    with open(dir + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append((line, dir))

LoadMeasurements(['data/track_1_data/', 'data/recoveries2/'])

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    '''
        Setup generator to generation input data
    '''
    num_samples = len(samples)
    while 1:
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                line, dir = batch_sample
                image_dir = dir + 'IMG/'

                filename = line[0].split('/')[-1]
                image = cv2.imread(image_dir + filename)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Use left and write pictures with a correction factor
                filename_left = line[1].split('/')[-1]
                image_left = cv2.imread(image_dir + filename_left)
                #Convert from BGR to RGB
                image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)

                filename_right = line[2].split('/')[-1]
                image_right = cv2.imread(image_dir + filename_right)
                #Convert from BGR to RGB
                image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)

                images.append(image)
                images.append(image_left)
                images.append(image_right)

                correction = 0.2

                measurement = float(line[3])
                measurement_left = measurement + correction
                measurement_right = measurement - correction

                measurements.append(measurement)
                measurements.append(measurement_left)
                measurements.append(measurement_right)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

X_train = np.array(images)
y_train = np.array(measurements)

def MorePowerfulNet():
    '''
        Setup NVidia conv net architecture
    '''
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x/255.0) - 0.5))
    model.add(Conv2D(filters=24, strides=(2,2), kernel_size=(5, 5), activation='relu'))
    model.add(Conv2D(filters=36,  strides=(2,2), kernel_size=(5, 5), activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

model = Sequential()
MorePowerfulNet()
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size),
                    validation_data=validation_generator,
                    validation_steps=math.ceil(len(validation_samples)/batch_size), epochs=3, verbose=1)
model.save('model.h5')