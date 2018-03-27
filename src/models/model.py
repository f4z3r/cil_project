#!/usr/bin/env python3

import numpy as np
import keras

import utility
from models import model


class Model:
    """A model template for all models used by this program."""

    def __init__(self, train_path, patch_size=16, context_padding=28, load_images=True):
        """Initialise the model.

        Args:
            train_path (str): path to training data.
            patch_size (int): default=16 - the size of the patch to analyse.
            context_padding (int): default=28 - padding on each side of the analysed patch.
            load_images (bool): ONLY DISABLE FOR CODE CHECKS
        """
        # Seed the RNG to ensure the results are reproducible.
        np.random.seed(0)

        self.train_path = train_path
        self.patch_size = patch_size
        self.context_padding = context_padding
        self.window_size = patch_size + 2 * context_padding



    def load_images(self):
        """Load all images into memory."""
        self.data_set, self.verifier_set = utility.load_training_set(self.train_path,
                                                                     self.context_padding)

    def create_batch(self, batch_size=100):
        """Create a batch to feed to the neural network for training.

        Args:
            batch_size (int): size of each batch.
        """
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


    def train(self, verbosity, epochs=150, steps=5000, print_at_end=True):
        """Train the model.

        Args:
            verbosity (bool): if the training should be verbose.
            epochs (int): default: 150 - epochs to train.
            steps (int): default: 5000 - batches per epoch to train.
            print_at_end (bool): print history at the end of the training.
        """
        pass

    def evaluate(self):
        """Evaluate the efficiency of the model."""
        pass


    def save(self, filename):
        """Save the weights of the trained model.

        Args:
            filename (str): filename for the weights.
        """
        pass
