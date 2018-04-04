#!/usr/bin/env python3

import numpy as np


class AbstractModel:
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
        pass

    def create_batch(self, batch_size=100):
        """Create a batch to feed to the neural network for training.

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
