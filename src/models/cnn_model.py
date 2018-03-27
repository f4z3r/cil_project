#!/usr/bin/env python3

import os, sys, logging
import numpy as np

# Silence import message
stderr = sys.stderr
#sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr


import utility
from models import model

logger = logging.getLogger("cil_project.models.cnn_lr_d")

file_path = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import os

os.environ["MKL_THREADING_LAYER"] = "GNU"

import keras
from keras.models import Sequential
from keras.layers.core import Dense,Dropout, Flatten
from keras.layers.convolutional import Conv2D,Conv3D, MaxPooling2D,MaxPooling3D, AveragePooling3D
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Embedding, LSTM, Bidirectional, Lambda, concatenate, add, Embedding, TimeDistributed
import matplotlib.image as img
import cv2
import csv
from models import model
import utility

class CNN_keras(model.Model):

    def __init__(self, train_path, patch_size=16, context_padding=28, load_images=True):

        super().__init__(train_path, patch_size, context_padding, load_images)
        logger.info("Generating CNN model with leaky ReLU and dropouts ...")

        # The following can be set using a config file in ~/.keras/keras.json
        if keras.backend.image_dim_ordering() == "tf":
            # Keras is using Tensorflow as backend
            input_dim = (self.window_size, self.window_size, 3)
        else:
             # Keras is using Theano as backend
            input_dim = (3, self.window_size, self.window_size)
        
        if load_images:
            # Preload the images
            self.load_images()
        else:
            raise ValueError("load_images must be set to True")

        logger.info("Done")


        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(10,10), input_shape=input_dim))
        #model= BatchNormalization()(model)
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(4,4)))
        self.model.add(Flatten())
        self.model.add(keras.layers.Dense(units=2,
                                          kernel_regularizer=keras.regularizers.l2(1e-6),
                                          activation="softmax"))
        print(self.model.summary())


    
    def train(self, verbosity, epochs=150, steps=5000, print_at_end=True):

        optimiser = keras.optimizers.Adam()
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=optimiser,
                           metrics=["accuracy"])
        #training_set=self.create_batch()
        #print(training_set)
        #self.model.fit(training_set, epochs=10)
        hist = self.model.fit_generator(self.create_batch(),
                                        steps_per_epoch=steps,
                                        epochs=epochs
                                        )


#model=CNN_keras("CNN")
#print(model.images)
#model.preprocess_data()
#model.model_setup()
#model.fit_model()


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