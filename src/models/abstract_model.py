#!/usr/bin/env python3

import numpy as np


class AbstractModel:
    """A model template for all models used by this program."""

    def __init__(self, train_path, validation_path, patch_size=16, context_padding=28, load_images=True):
        """Initialise the model.

        Args:
            train_path (str): path to training data.
            patch_size (int): default=16 - the size of the patch to analyse.
            context_padding (int): default=28 - padding on each side of the analysed patch.
            load_images (bool): ONLY DISABLE FOR CODE CHECKS
        """
        pass

    def load_images(self):
        """Load all images into memory."""
        pass

    def create_train_batch(self, batch_size=100):
        """Create a batch to feed to the neural network for training.

        Args:
            batch_size (int): size of each batch.
        """
        pass

    def create_validation_batch(self, batch_size=100):
        """Create a batch to feed to the neural network for validation.

        Args:
            batch_size (int): size of each batch.
        """
        pass

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
