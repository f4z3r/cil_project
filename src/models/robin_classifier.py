#!/usr/bin/env python3

import os, logging

from keras import Input, Model, optimizers, losses, callbacks
from keras.layers import Convolution2D, LeakyReLU, MaxPooling2D, Dropout, Dense, regularizers, Flatten

import utility
from models import augmentation_model

logger = logging.getLogger("cil_project.models.RobinClassifier")

file_path = os.path.dirname(os.path.abspath(__file__))

class RobinClassifier(augmentation_model.AugmentationModel):
    """CNN model implementing a classifier using leaky ReLU and dropouts."""

    def __init__(self, train_path, validation_path, patch_size=16, context_padding=28, load_images=True):
        self.train_path = train_path
        self.validation_path = validation_path
        self.patch_size = patch_size

        dimension = 72

        inputs = Input(shape=(dimension, dimension, 3))
        x = Convolution2D(filters=64,
                          kernel_size=(5, 5),
                          padding="same")(inputs)

        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2),
                         padding="same")(x)
        x = Dropout(rate=0.25)(x)

        # Define the second wave of layers
        x = Convolution2D(filters=128,
                          kernel_size=(3, 3),
                          padding="same")(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2),
                         padding="same")(x)
        x = Dropout(rate=0.25)(x)

        # Define the third wave of layers
        x = Convolution2D(filters=256,
                          kernel_size=(3, 3),
                          padding="same")(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2),
                         padding="same")(x)
        x = Dropout(rate=0.25)(x)

        # Define the fourth wave of layers
        x = Convolution2D(filters=256,
                          kernel_size=(3, 3),
                          padding="same")(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2),
                         padding="same")(x)
        x = Dropout(rate=0.25)(x)

        # Define the fifth wave of layers
        x = Flatten()(x)
        x = Dense(units=128,
                  kernel_regularizer=regularizers.l2(1e-6))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(rate=0.5)(x)

        predictions = Dense(units=2,
                            kernel_regularizer=regularizers.l2(1e-6),
                            activation="softmax")(x)

        self.model = Model(inputs=inputs, outputs=predictions)

        optimiser = optimizers.Adam()
        self.model.compile(loss=losses.categorical_crossentropy,
                      optimizer=optimiser,
                      metrics=["accuracy"])

    @utility.overrides(augmentation_model.AugmentationModel)
    def train(self, verbosity, epochs=150, steps=5000, print_at_end=True):
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

        log_dir = os.path.join(os.path.dirname(file_path), os.path.normpath("..//data/logs/"))
        model_dir = os.path.join(os.path.dirname(file_path), os.path.normpath("..//data/models/"))
        tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=32,
                                                     write_graph=True,
                                                     write_grads=False, write_images=False, embeddings_freq=0,
                                                     embeddings_layer_names=None, embeddings_metadata=None)

        checkpoint_callback = callbacks.ModelCheckpoint(model_dir + '\model.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                        monitor='val_loss', verbose=0, save_best_only=False,
                                                        save_weights_only=False, mode='auto', period=1)

        self.model.fit_generator(
            self.create_train_batch(),
            epochs=150,
            steps_per_epoch=1000,
            verbose=1,
            validation_data=self.create_validation_batch(),
            validation_steps=1,
            callbacks=[lr_callback, stop_callback, checkpoint_callback, tensorboard_callback]
        )

