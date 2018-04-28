#!/usr/bin/env python3


class BaseModel:
    """A model template for all models used by this program."""

    def __init__(self, train_generator, validation_generator):
        """Initialise the model.
        """
        self.train_generator = train_generator
        self.validation_generator = validation_generator

    def train(self, verbosity, epochs=150, steps=5000, print_at_end=True):
        """Train the model.

        Args:
            verbosity (bool): if the training should be verbose.
            epochs (int): default: 150 - epochs to train.
            steps (int): default: 5000 - batches per epoch to train.
            print_at_end (bool): print history at the end of the training.
        """
        raise NotImplementedError
    
    def predict(self):
        """Predict with the model.

        """
        raise NotImplementedError

    def save(self, filename):
        """Save the weights of the trained model.

        Args:
            filename (str): filename for the weights.
        """
        raise NotImplementedError
