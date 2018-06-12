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
            id_length = len(id_image)
            #Padding with zeros the ids
            if id_length<3:
                zero_padding = 3-id_length
                id_image = "0"*zero_padding+str(id_image)

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

"""def look_matching_pred(csv, id):
    for row_csv2 in reader_csv2:
        print("cy")
        if str(row_csv2[0]) == str(row_csv1[0]):
            new_submission_entries.append(row_csv2[1])"""
def hashmap_given_csv(reader_csv1):
    ids = []
    for entry in reader_csv1:
        ids.append(entry)

def merge_predictions(csv1_path, csv2_path, submission_file):

    #Naive way implementation, can be done in several ways : hash map, sorting ids..
    reader_csv1 = csv.reader(open(csv1_path, 'r')) #, delimiter=",")
    reader_csv2 = csv.reader(open(csv2_path, 'r'))#, delimiter=",")
    
    hashmap_csv1 = dict(list(reader_csv1))
    hashmap_csv2 = dict(list(reader_csv2))

    new_submission_entries = []
    
    #Copy csv1 as new submission to later integrate with csv2
    new_submission_csv = dict(list(reader_csv1))

    reader_csv1 = list(hashmap_csv1)[1:len(list(hashmap_csv1))]

    for entry in reader_csv1:
        if not hashmap_csv1[entry] == '1':
            hashmap_csv1[entry] = hashmap_csv2[entry]

    new_sub = list(hashmap_csv1)

    writer = csv.writer(open(submission_file, 'w'), delimiter=",")

    for key, value in hashmap_csv1.items():
        writer.writerow([str(key), str(value)])
