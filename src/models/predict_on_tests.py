#!/usr/bin/env python3

import csv
import logging

import os

logger = logging.getLogger("cil_project.models.cnn_model")
file_path = os.path.dirname(os.path.abspath(__file__))

os.environ["MKL_THREADING_LAYER"] = "GNU"


class Prediction_model():
    def __init__(self, test_generator_class, restored_model):

        self.test_generator_class = test_generator_class
        self.prediction_model = restored_model

    def prediction_given_model(self):

        test_generator = self.test_generator_class.generate_test_patches()
        model = self.prediction_model

        predictions = []
        for img_patches in test_generator:
            img_predictions = model.predict_on_batch(img_patches)
            predictions.append(img_predictions)
            # print("First 10 predictions are ",img_predictions[0:10])
            print("Done with prediction on the image")

        return predictions

    def save_predictions_to_csv(self, predictions, submission_file):
        """Requires :
            predictions: 2-d array -> patches for each image
        """
        dimensions = list(self.test_generator_class.data_set[0].shape)
        w_image = dimensions[0] - self.test_generator_class.context_padding * 2
        h_image = dimensions[1] - self.test_generator_class.context_padding * 2
        patch_size = self.test_generator_class.patch_size
        images_ids = self.test_generator_class.images_ids

        writer = csv.writer(open(submission_file, 'w'), delimiter=",")
        writer.writerow(["Id", "Prediction"])

        id_idx = 0
        for prediction_batch in predictions:
            id_image = images_ids[id_idx]
            idx_row = 0
            idx_column = 0
            for prediction_probabilities in prediction_batch:
                prediction = list(prediction_probabilities).index(max(list(prediction_probabilities)))
                full_row = [str(id_image) + "_" + str(idx_column) + "_" + str(idx_row), str(prediction)]
                writer.writerow(full_row)

                if idx_row + patch_size < w_image:
                    idx_row = idx_row + patch_size
                else:
                    idx_row = 0
                    idx_column = idx_column + patch_size
            id_idx = id_idx + 1

        print("Submission csv file written to disk successfully!")
