#!/usr/bin/env python3

import numpy as np
import keras

import utility

from models.abstract_model import AbstractModel


class CnnBaseModel(AbstractModel):
    """A model template for all models used by this program."""

    def __init__(self, train_path, validation_path, patch_size=16, context_padding=28, load_images=True):
        # Seed the RNG to ensure the results are reproducible.
        np.random.seed(0)

        self.train_path = train_path
        self.validation_path = validation_path
        self.patch_size = patch_size
        self.context_padding = context_padding
        self.window_size = patch_size + 2 * context_padding

    def load_images(self):
        self.data_set, self.verifier_set = utility.load_training_set(self.train_path, self.context_padding)

    def create_train_batch(self, batch_size=100):
        while True:
            batch_data = np.empty((batch_size, self.window_size, self.window_size, 3))
            batch_verifier = np.empty((batch_size, 2))

            for idx in range(batch_size):
                img_num = np.random.choice(self.data_set.shape[0])
                data_patch, veri_patch = utility.get_random_image_patch(self.data_set[img_num],
                                                                        self.verifier_set[img_num],
                                                                        self.patch_size,
                                                                        self.patch_size,
                                                                        self.context_padding)
                label = (np.mean(veri_patch) > 0.25) * 1
                label = keras.utils.to_categorical(label, num_classes=2)
                batch_data[idx] = data_patch
                batch_verifier[idx] = label

            if keras.backend.image_dim_ordering() == "th":
                batch_data = np.rollaxis(batch_data, 3, 1)

            yield (batch_data, batch_verifier)
