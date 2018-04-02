#!/usr/bin/env python3

import os, sys, logging
from keras import models, layers, regularizers, optimizers, losses, callbacks

from src.models.RoadSequenceGenerator import RoadSequenceGenerator

import utility
from models import model

logger = logging.getLogger("cil_project.models.RobinClassifier")

file_path = os.path.dirname(os.path.abspath(__file__))


class RobinClassifier(model.Model):
    """CNN model implementing a classifier using leaky ReLU and dropouts."""

    def __init__(self, train_path, validation_path, dimension=72):
        # Define the model
        self.model = models.Sequential()

        # Define the first wave of layers
        self.model.add(layers.Convolution2D(filters=64,
                                            kernel_size=(5, 5),
                                            padding="same",
                                            input_shape=(dimension, dimension, 3)))
        self.model.add(layers.LeakyReLU(alpha=0.1))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2),
                                           padding="same"))
        self.model.add(layers.Dropout(rate=0.25))

        # Define the second wave of layers
        self.model.add(layers.Convolution2D(filters=128,
                                            kernel_size=(3, 3),
                                            padding="same"))
        self.model.add(layers.LeakyReLU(alpha=0.1))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2),
                                           padding="same"))
        self.model.add(layers.Dropout(rate=0.25))

        # Define the third wave of layers
        self.model.add(layers.Convolution2D(filters=256,
                                            kernel_size=(3, 3),
                                            padding="same"))
        self.model.add(layers.LeakyReLU(alpha=0.1))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2),
                                           padding="same"))
        self.model.add(layers.Dropout(rate=0.25))

        # Define the fourth wave of layers
        self.model.add(layers.Convolution2D(filters=256,
                                            kernel_size=(3, 3),
                                            padding="same"))
        self.model.add(layers.LeakyReLU(alpha=0.1))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2),
                                           padding="same"))
        self.model.add(layers.Dropout(rate=0.25))

        # Define the fifth wave of layers
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(units=128,
                                    kernel_regularizer=regularizers.l2(1e-6)))
        self.model.add(layers.LeakyReLU(alpha=0.1))
        self.model.add(layers.Dropout(rate=0.5))

        self.model.add(layers.Dense(units=2,
                                    kernel_regularizer=regularizers.l2(1e-6),
                                    activation="softmax"))

        self.generator_train = RoadSequenceGenerator(train_path, 'data', 'verify', 2, dim=dimension)
        self.generator_validation = RoadSequenceGenerator(validation_path, 'data', 'verify', 2, dim=dimension)

    @utility.overrides(model.Model)
    def train(self, verbosity, epochs=150, steps=5000, print_at_end=True):

        logger.info("Preparing training, compiling model ...")

        optimiser = optimizers.Adam()
        self.model.compile(loss=losses.categorical_crossentropy,
                           optimizer=optimiser,
                           metrics=["accuracy"])

        lr_callback = callbacks.ReduceLROnPlateau(monitor="acc",
                                                  factor=0.5,
                                                  patience=0.5,
                                                  verbose=0,
                                                  epsilon=0.0001,
                                                  cooldown=0,
                                                  min_lr=0)
        stop_callback = callbacks.EarlyStopping(monitor="acc",
                                                min_delta=0.0001,
                                                patience=11,
                                                verbose=0,
                                                mode="auto")

        logger.info("Starting training ...")

        hist = self.model.fit_generator(self.generator_train,
                                        steps_per_epoch=steps,
                                        epochs=epochs,
                                        callbacks=[lr_callback, stop_callback],
                                        validation_data=self.generator_validation)
        print(hist.history)

    @utility.overrides(model.Model)
    def save(self, filename):
        """Save the weights of the trained model.

        Args:
            filename (str): filename for the weights.
        """
        self.model.save_weights(os.path.join(file_path, "../../results/weights", filename))
        logger.info("Weights saved to results/weights/{}".format(filename))
