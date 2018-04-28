#!/usr/bin/env python3

import logging
import os

from keras import callbacks

from models.base_model import BaseModel

logger = logging.getLogger("cil_project.models.cnn_model")

file_path = os.path.dirname(os.path.abspath(__file__))

import os

os.environ["MKL_THREADING_LAYER"] = "GNU"

import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers.core import Activation, Flatten, Reshape

"""Resources used :
   http://cs231n.github.io/convolutional-networks/

   This implementation plays around with the hidden dimesion of the filters applied to the image.
   The 3-rd dimension, originally colors, comes to be substituted after the very first 3-d convolution, which convolutes over
   the all colors at the same time, by the hidden dimensions of filters.
   Then other stacked triplets of layers expands the hidden dimension and restrict it. Since non lnearity is applied, each of the three stacked blocks
   learns its filters.
   """


class CNN_keras(BaseModel):
    def __init__(self, train_generator, validation_generator = []):

        super().__init__(train_generator, validation_generator)

        logger.info("Generating CNN model with leaky ReLU and dropouts ...")

        input_dim = self.train_generator.input_dim(four_dim=True)
        print("input DMENSIONS")
        print(self.train_generator.input_dim())

        """Applying conv3D focusing on the new 3-rd dimension (filters) of the previous conv3D which hopefully learns through time 
           to attribute distinctive values to roads and not roads sub-filtered images. A way to think of it is that the first conv3D layer
           learns to count roads sub image and not sub image looking at filters dimension |||||||||| -> layers depth (filters) -> what is road , what is not? """

        """Think if to add a stride in layers & model= BatchNormalization()(model) at some point in blocks of layers"""
        self.model = Sequential()

        """ max-pooling and monotonely increasing non-linearities commute, so the result is the same, but applying max pooing before, subsample stuff and reduce computation
            https://stackoverflow.com/questions/35543428/activation-function-after-pooling-layer-or-convolutional-layer"""

        """1-st block of layers"""

        """From 72x72x3 -> 18x18x32"""
        self.model.add(Conv3D(32, kernel_size=(4, 4, 3), strides=(4, 4, 3), input_shape=input_dim))
        self.model.add(Reshape((18, 18, 32, 1)))
        """From 18x18x32 -> 9x9x16"""
        self.model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
        self.model.add(Reshape((9, 9, 16, 1)))
        self.model.add(Activation('relu'))
        print("Gone through first block")

        """2-rd block of layers"""
        """From 9x9x16 -> 9x9x64   (3-rd, filters, dimension expanded)"""
        self.model.add(Conv3D(64, kernel_size=(1, 1, 16), strides=(1, 1, 16), input_shape=input_dim))
        self.model.add(Reshape((9, 9, 64, 1)))

        """From 9x9x64 -> 9x9x32 (focusing on hidden filters dimension -> learning hidden representation)"""
        self.model.add(MaxPooling3D(pool_size=(1, 1, 2), strides=(1, 1, 2)))
        self.model.add(Reshape((9, 9, 32, 1)))

        self.model.add(Activation('relu'))

        """Again expanding the 3-rd dimension -> hidden dimension (of filter results which this way with backprop are forced to learn)"""
        """3-rd block of layers"""
        """From 9x9x32 -> 9x9x100 -> maybe the higher the depth , the more accuracy on results we get (the more filters learn and share their opinion on the sub image)"""
        self.model.add(Conv3D(100, kernel_size=(1, 1, 32), strides=(1, 1, 32), input_shape=input_dim))
        self.model.add(Reshape((9, 9, 100, 1)))

        """From 9x9x100 -> 9x9x50 (here we could do even 1x1x50 -> better to expand dimension in dense layer -> expanding just hidden dimensions) """
        self.model.add(MaxPooling3D(pool_size=(1, 1, 2), strides=(1, 1, 2)))
        self.model.add(Reshape((9, 9, 50, 1)))

        self.model.add(Activation('relu'))

        """Please note that each filter on each block of layers learn from previous filters of previous block of layers"""

        """Dense layer -> out for prediction & groundtruth -> learning afterwards with backprop"""
        """Thinks if to decomment the dense layer before the output one with only 2 units,
           cause all the information can be kernelized expanding with a dense layer (from 
           9x9x32 = 2592 -> for instance W=(2592,10000) going to higher dimensions 
           (sort of SVM, which is luckily done implicitely by the Neural network + non linearity)
           and then last out layer (10000,2) all reduced to 2 dimensions for output """

        self.model.add(Flatten())
        self.model.add(keras.layers.Dense(units=10000,
                                          kernel_regularizer=keras.regularizers.l2(1e-6),
                                          activation="relu"))
        # self.model.add(Activation('relu'))

        self.model.add(keras.layers.Dense(units=2,
                                          kernel_regularizer=keras.regularizers.l2(1e-6),
                                          activation="softmax"))
        print(self.model.summary())

    def train(self, verbosity, epochs=150, steps=5000, print_at_end=True):
        optimiser = keras.optimizers.Adam()
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=optimiser,
                           metrics=["accuracy"])
        # training_set=self.create_batch()
        # print(training_set)
        # self.model.fit(training_set, epochs=10)

        log_dir = os.path.join(os.path.dirname(file_path), os.path.normpath("..//data/logs/"))
        model_dir = os.path.join(os.path.dirname(file_path), os.path.normpath("..//data/models/"))
        tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=32,
                                                     write_graph=True,
                                                     write_grads=False, write_images=False, embeddings_freq=0,
                                                     embeddings_layer_names=None, embeddings_metadata=None)

        checkpoint_callback = callbacks.ModelCheckpoint(model_dir + '\model_cnn_model.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                        monitor='val_loss', verbose=0, save_best_only=False,
                                                        save_weights_only=False, mode='auto', period=1)

        self.model.fit_generator(self.train_generator.generate_patch(four_dim=True),
                                 steps_per_epoch=steps,
                                 epochs=epochs,
                                 callbacks=[checkpoint_callback, tensorboard_callback],
                                 validation_data=self.validation_generator.generate_patch(four_dim=True),
                                 validation_steps=100
                                 )




"""from now on just copy pasted examples"""

"""#variable initialization 
nb_filters =100
kernel_size= {}
kernel_size[0]= [3,3]
kernel_size[1]= [4,4]
kernel_size[2]= [5,5]
input_shape=(32, 32, 3)
pool_size = (2,2)
nb_classes =2
no_parallel_filters = 3

# create seperate model graph for parallel processing with different filter sizes
# apply 'same' padding so that ll produce o/p tensor of same size for concatination
# cancat all paralle output

inp = Input(shape=input_shape)
convs = []
for k_no in range(len(kernel_size)):
    conv = Convolution2D(nb_filters, kernel_size[k_no][0], kernel_size[k_no][1],
                    border_mode='same',
                         activation='relu',
                    input_shape=input_shape)(inp)
    pool = MaxPooling2D(pool_size=pool_size)(conv)
    convs.append(pool)

if len(kernel_size) > 1:
    out = Merge(mode='concat')(convs)
else:
    out = convs[0]

conv_model = Model(input=inp, output=out)

# add created model grapg in sequential model

model = Sequential()
model.add(conv_model)        # add model just like layer
model.add(Convolution2D(nb_filters, kernel_size[1][0], kernel_size[1][0]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
model.add(Flatten(input_shape=input_shape))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('tanh'))
"""

"""rows, cols = 100, 15
def create_convnet(img_path='network_image.png'):
    input_shape = Input(shape=(rows, cols, 1))

    tower_1 = Conv2D(20, (100, 5), padding='same', activation='relu')(input_shape)
    tower_1 = MaxPooling2D((1, 11), strides=(1, 1), padding='same')(tower_1)

    tower_2 = Conv2D(20, (100, 7), padding='same', activation='relu')(input_shape)
    tower_2 = MaxPooling2D((1, 9), strides=(1, 1), padding='same')(tower_2)

    tower_3 = Conv2D(20, (100, 10), padding='same', activation='relu')(input_shape)
    tower_3 = MaxPooling2D((1, 6), strides=(1, 1), padding='same')(tower_3)

    merged = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
    merged = Flatten()(merged)

    out = Dense(200, activation='relu')(merged)
    out = Dense(num_classes, activation='softmax')(out)

    model = Model(input_shape, out)
    plot_model(model, to_file=img_path)
    return model"""
