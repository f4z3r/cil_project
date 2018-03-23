#!/usr/bin/env python3

import os, sys, logging

# Silence import message
stderr = sys.stderr
sys.stderr = open('/dev/null', 'w')
import keras
sys.stderr = stderr


import utility

logger = logging.getLogger("cil_project.models.cnn_lr_d")

class Model:
    """CNN model implementing a classifier using leaky ReLU and dropouts."""

    def __init__(self, patch_size=16, context_padding=28):
        """Initialise the model."""
        logger.info("Generating model ...")

        self.patch_size = patch_size
        self.context_padding = context_padding

        # The following can be set using a config file in ~/.keras/keras.json
        if keras.backend.image_dim_ordering() == "tf":
            # Keras is using Tensorflow as backend
            input_dim = (self.patch_size, self.patch_size, 3)
        else:
            # Keras is using Theano as backend
            input_dim = (3, self.patch_size, self.patch_size)

        # Define the model
        self.model = keras.models.Sequential()

        # Define the first wave of layers
        self.model.add(keras.layers.Convolution2D(filters=64,
                                                  kernel_size=(5, 5),
                                                  padding="same",
                                                  input_shape=input_dim))
        self.model.add(keras.layers.LeakyReLU(alpha=0.1))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2,2),
                                                 padding="same"))
        self.model.add(keras.layers.Dropout(rate=0.25))

        # Define the second wave of layers
        self.model.add(keras.layers.Convolution2D(filters=128,
                                                  kernel_size=(3, 3),
                                                  padding="same"))
        self.model.add(keras.layers.LeakyReLU(alpha=0.1))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2,2),
                                                 padding="same"))
        self.model.add(keras.layers.Dropout(rate=0.25))

        # Define the third wave of layers
        self.model.add(keras.layers.Convolution2D(filters=256,
                                                  kernel_size=(3, 3),
                                                  padding="same"))
        self.model.add(keras.layers.LeakyReLU(alpha=0.1))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2,2),
                                                 padding="same"))
        self.model.add(keras.layers.Dropout(rate=0.25))

        # Define the fourth wave of layers
        self.model.add(keras.layers.Convolution2D(filters=256,
                                                  kernel_size=(3, 3),
                                                  padding="same"))
        self.model.add(keras.layers.LeakyReLU(alpha=0.1))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2,2),
                                                 padding="same"))
        self.model.add(keras.layers.Dropout(rate=0.25))

        # Define the fifth wave of layers
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(units=128,
                                          kernel_regularizer=keras.regularizers.l2(1e-6)))
        self.model.add(keras.layers.LeakyReLU(alpha=0.1))
        self.model.add(keras.layers.Dropout(rate=0.5))

        self.model.add(keras.layers.Dense(units=2,
                                          kernel_regularizer=keras.regularizers.l2(1e-6)))

        logger.info("Done")



