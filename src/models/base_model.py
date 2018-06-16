#!/usr/bin/env python3

from keras.models import load_model


class BaseModel:
    """A model template for all models used by this program."""

    def __init__(self, train_generator, validation_generator):
        """Initialise the model.
           Remark: the validation generator feeded is empty if the model is predicting,
           it is not empty otherwise
        """
        self.train_generator = train_generator
        self.validation_generator = validation_generator

    def train(self, epochs=150, steps=5000, print_at_end=True):
        """Train the model.

        Args:
            epochs (int): default: 150 - epochs to train.
            steps (int): default: 5000 - batches per epoch to train.
            print_at_end (bool): print history at the end of the training.
        """
        raise NotImplementedError

    def predict(self):
        """Predict with the model.

        """
        raise NotImplementedError

    def save(self, path):
        """Save the model of the trained model.

        Args:
            path (path): path for the model file.
        """
        raise NotImplementedError

    def load(self, path):
        """Load a (partly) trained model.

        Args:
            path (path): path for the model file.

        Returns:
            The model.
        """
        self.model = load_model(path)
