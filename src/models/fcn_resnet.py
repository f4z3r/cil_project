#!/usr/bin/env python3

import logging
import os, sys

import keras
from keras import backend as K

import numpy as np

import tensorflow as tf

from models.base_model import BaseModel
from utils.commons import *

logger = logging.getLogger("cil_project.src.models.fcn_resnet")

file_path = os.path.dirname(os.path.abspath(__file__))

class FCNResNet(BaseModel):
    """A Fully convolutional neural network using the ResNet-50 architecture to add resudial
    learning. Such learning is achieved using shortcut connections (directly connecting the input
    of nth layer to some later (n+x)th layer). The 50 in "ResNet-50" indicates that it is a 50 layer
    residual network.
    """

    def __init__(self, train_generator, validation_generator = [], weight_decay=0.00005, batch_momentum=0.99, path=None):
        super().__init__(train_generator, validation_generator)

        # Load the model if given in a path
        if path is not None:
            logger.info("Loading existing model from {}".format(path))
            self.load(path)
            logger.info("Finished loading the model")
            return

        logger.info("Generating FCN model with ResNet-50 architecture...")

        num_classes = 1
        image_size = [400, 400, 3]
        inputs = keras.Input(shape=image_size)

        bn_axis = 3

        x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1',
                                kernel_regularizer=keras.regularizers.l2(weight_decay))(inputs)
        x = keras.layers.BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self._conv_block(3, filters=[64, 64, 256], stage=2, block='a', weight_decay=weight_decay, strides=(1, 1),
                             batch_momentum=batch_momentum)(x)
        x = self._identity_block(3, filters=[64, 64, 256], stage=2, block='b', weight_decay=weight_decay,
                                 batch_momentum=batch_momentum)(x)
        x = self._identity_block(3, filters=[64, 64, 256], stage=2, block='c', weight_decay=weight_decay,
                                 batch_momentum=batch_momentum)(x)

        x = self._conv_block(3, filters=[128, 128, 512], stage=3, block='a', weight_decay=weight_decay,
                             batch_momentum=batch_momentum)(x)
        x = self._identity_block(3, filters=[128, 128, 512], stage=3, block='b', weight_decay=weight_decay,
                                 batch_momentum=batch_momentum)(x)
        x = self._identity_block(3, filters=[128, 128, 512], stage=3, block='c', weight_decay=weight_decay,
                                 batch_momentum=batch_momentum)(x)
        x = self._identity_block(3, filters=[128, 128, 512], stage=3, block='d', weight_decay=weight_decay,
                                 batch_momentum=batch_momentum)(x)

        x = self._conv_block(3, filters=[256, 256, 1024], stage=4, block='a', weight_decay=weight_decay,
                             batch_momentum=batch_momentum)(x)
        x = self._identity_block(3, filters=[256, 256, 1024], stage=4, block='b', weight_decay=weight_decay,
                                 batch_momentum=batch_momentum)(x)
        x = self._identity_block(3, filters=[256, 256, 1024], stage=4, block='c', weight_decay=weight_decay,
                                 batch_momentum=batch_momentum)(x)
        x = self._identity_block(3, filters=[256, 256, 1024], stage=4, block='d', weight_decay=weight_decay,
                                 batch_momentum=batch_momentum)(x)
        x = self._identity_block(3, filters=[256, 256, 1024], stage=4, block='e', weight_decay=weight_decay,
                                 batch_momentum=batch_momentum)(x)
        x = self._identity_block(3, filters=[256, 256, 1024], stage=4, block='f', weight_decay=weight_decay,
                                 batch_momentum=batch_momentum)(x)

        x = self._atrous_conv_block(3, filters=[512, 512, 2048], stage=5, block='a', weight_decay=weight_decay,
                                    atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
        x = self._atrous_identity_block(3, filters=[512, 512, 2048], stage=5, block='b', weight_decay=weight_decay,
                                        atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
        x = self._atrous_identity_block(3, filters=[512, 512, 2048], stage=5, block='c', weight_decay=weight_decay,
                                        atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)


        x = keras.layers.Conv2D(num_classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='same',
                                strides=(1, 1), kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
        x = BilinearUpSampling2D(target_size=tuple(image_size))(x)

        self.model = keras.models.Model(inputs, x)

        logger.info("Compiling model ...")

        lr_base = 0.01
        # optimiser = keras.optimizers.SGD(lr=lr_base, momentum=0.9)
        optimiser = keras.optimizers.Adam(lr=1e-4)

        self.model.compile(loss=self.bce_dice_loss, optimizer=optimiser, metrics=[self.dice_coef])



    def dice_coef(self, y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_coef_loss(self, y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

    def bce_dice_loss(self, y_true, y_pred):
        return K.binary_crossentropy(y_true, y_pred) + self.dice_coef_loss(y_true, y_pred)


    def train(self, verbosity, epochs=150, steps=1000):
        """Train the model.

        Args:
            verbosity (bool): if the training should be verbose.
            epochs (int): default: 150 - epochs to train.
            steps (int): default: 5000 - batches per epoch to train.
        """

        logger.info("Starting training ...")
        if verbosity:
            verbosity = 1
        else:
            verbosity = 0

        lr_callback = keras.callbacks.ReduceLROnPlateau(monitor="acc",
                                                        factor=0.5,
                                                        patience=0.5,
                                                        verbose=0,
                                                        min_delta=0.0001,
                                                        cooldown=0,
                                                        min_lr=0)
        stop_callback = keras.callbacks.EarlyStopping(monitor="acc",
                                                      min_delta=0.0001,
                                                      patience=11,
                                                      verbose=0,
                                                      mode="auto")


        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=properties["LOG_DIR"], histogram_freq=0, batch_size=32,
                                                           write_graph=True,
                                                           write_grads=False, write_images=False, embeddings_freq=0,
                                                           embeddings_layer_names=None, embeddings_metadata=None)

        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            os.path.join(properties["OUTPUT_DIR"], 'weights.h5'),
            monitor='val_loss', verbose=0, save_best_only=True,
            save_weights_only=False, mode='auto', period=1)

        self.model.fit_generator(
            self.train_generator.generate_patch(batch_size=8),
            steps_per_epoch=steps,
            epochs=epochs,
            callbacks=[lr_callback, stop_callback, tensorboard_callback, checkpoint_callback],
            verbose=1,
            validation_data=self.validation_generator.generate_patch(batch_size=8),
            validation_steps=5)

    def save(self, path):
        """Save the model of the trained model.

        Args:
            path (path): path for the model file.
        """
        self.model.save(path)
        logger.info("Model saved to {}".format(path))


    def _conv_block(self, kernel_size, filters, stage, block, weight_decay=0.0, strides=(2, 2), batch_momentum=0.99):
        """conv_block is the block that has a conv layer at shortcut

        # Arguments
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names

        Note that from stage 3, the first conv layer at main path is with strides=(2,2)
        And the shortcut should have strides=(2,2) as well
        """
        def f(input_tensor):
            nb_filter1, nb_filter2, nb_filter3 = filters
            if K.image_data_format() == "channels_last":
                bn_axis = 3
            else:
                bn_axis = 1

            conv_name_base = "res" + str(stage) + block + "_branch"
            bn_name_base = "bn" + str(stage) + block + "_branch"

            x = keras.layers.Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + "2a",
                                    kernel_regularizer=keras.regularizers.l2(weight_decay))(input_tensor)
            x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + "2a", momentum=batch_momentum)(x)
            x = keras.layers.Activation("relu")(x)

            x = keras.layers.Conv2D(nb_filter2, (kernel_size, kernel_size), padding="same",
                                   name=conv_name_base + "2b",
                                   kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
            x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + "2b", momentum=batch_momentum)(x)
            x = keras.layers.Activation("relu")(x)

            x = keras.layers.Conv2D(nb_filter3, (1, 1), name=conv_name_base + "2c",
                                   kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
            x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + "2c", momentum=batch_momentum)(x)

            shortcut = keras.layers.Conv2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + "1",
                                           kernel_regularizer=keras.regularizers.l2(weight_decay))(input_tensor)
            shortcut = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + "1",
                                                       momentum=batch_momentum)(shortcut)

            x = keras.layers.Add()([x, shortcut])
            x = keras.layers.Activation("relu")(x)

            return x
        return f


    def _identity_block(self, kernel_size, filters, stage, block, weight_decay=0., batch_momentum=0.99):
        """The identity_block is the block that has no conv layer at shortcut

        # Arguments
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        """
        def f(input_tensor):
            nb_filter1, nb_filter2, nb_filter3 = filters
            if K.image_data_format() == 'channels_last':
                bn_axis = 3
            else:
                bn_axis = 1

            conv_name_base = 'res' + str(stage) + block + '_branch'
            bn_name_base = 'bn' + str(stage) + block + '_branch'

            x = keras.layers.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                                    kernel_regularizer=keras.regularizers.l2(weight_decay))(input_tensor)
            x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
            x = keras.layers.Activation('relu')(x)

            x = keras.layers.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b',
                                    kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
            x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
            x = keras.layers.Activation('relu')(x)

            x = keras.layers.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                                    kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
            x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

            x = keras.layers.Add()([x, input_tensor])
            x = keras.layers.Activation('relu')(x)
            return x
        return f

    def _atrous_identity_block(self, kernel_size, filters, stage, block, weight_decay=0., atrous_rate=(2, 2),
                               batch_momentum=0.99):
        """The identity_block is the block that has no conv layer at shortcut

        # Arguments
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        """
        def f(input_tensor):
            nb_filter1, nb_filter2, nb_filter3 = filters
            if K.image_data_format() == 'channels_last':
                bn_axis = 3
            else:
                bn_axis = 1


            conv_name_base = 'res' + str(stage) + block + '_branch'
            bn_name_base = 'bn' + str(stage) + block + '_branch'

            x = keras.layers.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                                    kernel_regularizer=keras.regularizers.l2(weight_decay))(input_tensor)
            x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
            x = keras.layers.Activation('relu')(x)

            x = keras.layers.Conv2D(nb_filter2, (kernel_size, kernel_size), dilation_rate=atrous_rate,
                              padding='same', name=conv_name_base + '2b',
                              kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
            x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
            x = keras.layers.Activation('relu')(x)

            x = keras.layers.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                                    kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
            x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

            x = keras.layers.Add()([x, input_tensor])
            x = keras.layers.Activation('relu')(x)
            return x
        return f


    def _atrous_conv_block(self, kernel_size, filters, stage, block, weight_decay=0., strides=(1, 1), atrous_rate=(2, 2),
                           batch_momentum=0.99):
        """conv_block is the block that has a conv layer at shortcut

        # Arguments
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        """
        def f(input_tensor):
            nb_filter1, nb_filter2, nb_filter3 = filters
            if K.image_data_format() == 'channels_last':
                bn_axis = 3
            else:
                bn_axis = 1


            conv_name_base = 'res' + str(stage) + block + '_branch'
            bn_name_base = 'bn' + str(stage) + block + '_branch'

            x = keras.layers.Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a',
                              kernel_regularizer=keras.regularizers.l2(weight_decay))(input_tensor)
            x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
            x = keras.layers.Activation('relu')(x)

            x = keras.layers.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', dilation_rate=atrous_rate,
                              name=conv_name_base + '2b',
                              kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
            x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
            x = keras.layers.Activation('relu')(x)

            x = keras.layers.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                                    kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
            x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

            shortcut = keras.layers.Conv2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1',
                                     kernel_regularizer=keras.regularizers.l2(weight_decay))(input_tensor)
            shortcut = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1', momentum=batch_momentum)(shortcut)

            x = keras.layers.Add()([x, shortcut])
            x = keras.layers.Activation('relu')(x)
            return x
        return f


class BilinearUpSampling2D(keras.engine.topology.Layer):
    def __init__(self, size=(1, 1), target_size=None, data_format='default', **kwargs):
        if data_format == 'default':
            data_format = K.image_data_format()

        self.size = tuple(size)

        if target_size is not None:
            self.target_size = tuple(target_size)
        else:
            self.target_size = None

        assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {tf, th}'

        self.data_format = data_format
        self.input_spec = [keras.layers.InputSpec(ndim=4)]

        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            width = int(self.size[0] * input_shape[2] if input_shape[2] is not None else None)
            height = int(self.size[1] * input_shape[3] if input_shape[3] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]

            return (input_shape[0],
                    input_shape[1],
                    width,
                    height)

        elif self.data_format == 'channels_last':
            width = int(self.size[0] * input_shape[1] if input_shape[1] is not None else None)
            height = int(self.size[1] * input_shape[2] if input_shape[2] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]

            return (input_shape[0],
                    width,
                    height,
                    input_shape[3])

        else:
            raise Exception('Invalid data_format: ' + self.data_format)

    def call(self, x, mask=None):
        if self.target_size is not None:
            return self.resize_images_bilinear(x, target_height=self.target_size[0], target_width=self.target_size[1],
                                               data_format=self.data_format)
        else:
            return self.resize_images_bilinear(x, height_factor=self.size[0], width_factor=self.size[1],
                                               data_format=self.data_format)

    def get_config(self):
        config = {'size': self.size, 'target_size': self.target_size}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def resize_images_bilinear(self, X, height_factor=1, width_factor=1, target_height=None, target_width=None,
                               data_format='default'):
        """Resizes the images contained in a 4D tensor of shape

        - [batch, channels, height, width] (for 'channels_first' data_format)
        - [batch, height, width, channels] (for 'channels_last' data_format)

        by a factor of (height_factor, width_factor). Both factors should be
        positive integers.
        """
        if data_format == 'default':
            data_format = K.image_data_format()

        if data_format == 'channels_first':
            original_shape = K.int_shape(X)
            if target_height and target_width:
                new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
            else:
                new_shape = tf.shape(X)[2:]
                new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))

            X = permute_dimensions(X, [0, 2, 3, 1])
            X = tf.image.resize_bilinear(X, new_shape)
            X = permute_dimensions(X, [0, 3, 1, 2])

            if target_height and target_width:
                X.set_shape((None, None, target_height, target_width))
            else:
                X.set_shape((None, None, original_shape[2] * height_factor, original_shape[3] * width_factor))

            return X

        elif data_format == 'channels_last':
            original_shape = K.int_shape(X)
            if target_height and target_width:
                new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
            else:
                new_shape = tf.shape(X)[1:3]
                new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))

            X = tf.image.resize_bilinear(X, new_shape)

            if target_height and target_width:
                X.set_shape((None, target_height, target_width, None))
            else:
                X.set_shape((None, original_shape[1] * height_factor, original_shape[2] * width_factor, None))
            return X

        else:
            raise Exception('Invalid data_format: ' + data_format)

