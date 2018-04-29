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
        patch_size = test_generator.patch_size
        width_image, height_image, channels = test_generator.data_set[0].shape


        predictions = []
        for img_patches in test_generator:
            img_predictions = predict_on_batch(restored_model, img_patches)
            predictions.append(img_predictions)
            print("First 10 predictions are ",img_predictions[0:10])

        save_predictions_to_csv(images_ids = test_generator.images_ids, img_predictions=img_predictions,
                                 w_image = width_image, h_image = height_image, patch_size=patch_size)

    def save_predictions_to_csv(images_ids, img_predictions, w_image, h_image, patch_size):

        ts = int(time.time())
        print("Writing to submission file: submission_"+ts+".csv")

        writer = csv.writer(open("../submission_"+ts+".csv", 'w'), delimiter=",")
        writer.writerow(["Id", "Prediction"])
        #TODO see if it works, might be necessary to change the delimiter to
        # "," and to place that symbol as intearleaving between id and prediciton

        id_idx = 0
        for prediction_batch in img_predictions:
            id_image = images_ids[id_idx]
            idx_row = 0
            idx_column = 0
            for prediction in prediction_batch:
                full_row = [str(id_image)+"_"+str(idx_row)+"_"+str(idx_column), str(prediction)]
                writer.writerow(full_row)

                if idx_row + patch_size < w_image:
                    idx_row = idx_row + patch_size
                else:
                    idx_row = 0
                    idx_column = idx_column + patch_size
            id_idx = id_idx+1

        print("Submission csv file written to disk successfully!")



