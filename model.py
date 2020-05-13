#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import csv
import cv2
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

images = []
measurements = []

def LoadMeasurements(dirs):
    for dir in dirs:
        LoadMeasurement(dir)

def LoadMeasurement(dir):
    lines = []
    with open(dir + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = dir + 'IMG/' + filename
        image = cv2.imread(current_path)

        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

LoadMeasurements(['data/track_1_data/', 'data/track_1_data_r/', 'data/track_1_data_flipped/', 'data/track_1_data_r_flipped/'])

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')