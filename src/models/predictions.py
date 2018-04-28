#!/usr/bin/env python3

import logging
import os
from keras import callbacks
from models.base_model import BaseModel
import csv
import collections

logger = logging.getLogger("cil_project.models.cnn_model")
file_path = os.path.dirname(os.path.abspath(__file__))

import os

os.environ["MKL_THREADING_LAYER"] = "GNU"

import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers.core import Activation, Flatten, Reshape
import time


class Prediction_model():
    def __init__(self, test_generator, restored_model):
        
        self.test_generator = test_generator
        self.prediction_model = restored_model
    
    def prediction_given_model():

        test_generator = self.test_generator
        model = self.restored_model

        predictions = []
        for img_patches in test_generator:
            img_predictions = predict_on_batch(restored_model, img_patches)
            predictions.append(img_predictions)

        save_predictions_to_csv(images_ids = test_generator.images_ids, img_predictions=img_predictions)

    def save_predictions_to_csv(images_ids, img_predictions):

        ts = int(time.time())
        print("Writing to submission file with timestamp: ",ts)

        writer = csv.writer(open("../submission_"+ts, 'w'))
        writer.writerow("Id Prediction")
        #TODO to be completed with the image id + x and y coordinates
        for row in img_predictions:
            writer.writerow(row)
        
        return 0



