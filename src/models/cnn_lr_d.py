#!/usr/bin/env python3

import os, sys, logging
import numpy as np

# Silence import message
stderr = sys.stderr
#sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr


import utility
from models import cnn_base_model

logger = logging.getLogger("cil_project.models.cnn_lr_d")

file_path = os.path.dirname(os.path.abspath(__file__))

class CnnLrD(cnn_base_model.CnnBaseModel):
    """CNN model implementing a classifier using leaky ReLU and dropouts."""

    def __init__(self, train_generator, validation_generator, train_path, validation_path, patch_size=16, context_padding=28, load_images=True):
        """Initialise the model.

        Args:
            train_path (str): path to training data.
            patch_size (int): default=16 - the size of the patch to analyse.
            context_padding (int): default=28 - padding on each side of the analysed patch.
            load_images (bool): ONLY DISABLE FOR CODE CHECKS
        """
        self.train_generator = train_generator
        self.validation_generator = validation_generator

        # super().__init__(train_path, validation_path, patch_size, context_padding, load_images)

        logger.info("Generating CNN model with leaky ReLU and dropouts ...")

        # Define the model
        self.model = keras.models.Sequential()

        # Define the first wave of layers
        self.model.add(keras.layers.Convolution2D(filters=64,
                                                  kernel_size=(5, 5),
                                                  padding="same",
                                                  input_shape=train_generator.input_dim()))
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
                                          kernel_regularizer=keras.regularizers.l2(1e-6),
                                          activation="softmax"))

        logger.info("Done")

    @utility.overrides(cnn_base_model.CnnBaseModel)
    def train(self, verbosity, epochs=150, steps=5000, print_at_end=True):
        """Train the model.

        Args:
            verbosity (bool): if the training should be verbose.
            epochs (int): default: 150 - epochs to train.
            steps (int): default: 5000 - batches per epoch to train.
            print_at_end (bool): print history at the end of the training.
        """
        logger.info("Preparing training, compiling model ...")
        if verbosity:
            verbosity = 1
        else:
            verbosity = 0

        optimiser = keras.optimizers.Adam()
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=optimiser,
                           metrics=["accuracy"])

        lr_callback = keras.callbacks.ReduceLROnPlateau(monitor="acc",
                                                        factor=0.5,
                                                        patience=0.5,
                                                        verbose=0,
                                                        epsilon=0.0001,
                                                        cooldown=0,
                                                        min_lr=0)
        stop_callback = keras.callbacks.EarlyStopping(monitor="acc",
                                                      min_delta=0.0001,
                                                      patience=11,
                                                      verbose=0,
                                                      mode="auto")

        log_dir = os.path.join(os.path.dirname(file_path), os.path.normpath("..//data/logs/"))
        model_dir = os.path.join(os.path.dirname(file_path), os.path.normpath("..//data/models/"))
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=32,
                                                     write_graph=True,
                                                     write_grads=False, write_images=False, embeddings_freq=0,
                                                     embeddings_layer_names=None, embeddings_metadata=None)

        checkpoint_callback = keras.callbacks.ModelCheckpoint(model_dir + '\model_cnn_lr_d.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                        monitor='val_loss', verbose=0, save_best_only=False,
                                                        save_weights_only=False, mode='auto', period=1)

        logger.info("Starting training ...")

        try:
            hist = self.model.fit_generator(self.train_generator.generate_patch(),
                                            steps_per_epoch=5000,
                                            verbose=verbosity,
                                            epochs=epochs,
                                            callbacks=[lr_callback, stop_callback],
                                            validation_data=self.validation_generator.generate_patch(),
                                            validation_steps=100)
            if print_at_end:
                print(hist.history)
        except KeyboardInterrupt:
            logger.warning("\nTraining interrupted")
        else:
            logger.info("Training completed")

    @utility.overrides(cnn_base_model.CnnBaseModel)
    def save(self, filename):
        """Save the weights of the trained model.

        Args:
            filename (str): filename for the weights.
        """
        self.model.save_weights(os.path.join(file_path, "../../results/weights", filename))
        logger.info("Weights saved to results/weights/{}".format(filename))
