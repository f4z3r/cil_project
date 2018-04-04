#!usr/bin/env python3

import os, sys, logging
import numpy as np
import tensorflow as tf
# import keras
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from sklearn.preprocessing import LabelEncoder

import utility
from models import cnn_base_model

logger = logging.getLogger("cil_project.models.dnn_classifier")

file_path = os.path.dirname(os.path.abspath(__file__))

class DnnClassifier(cnn_base_model.CnnBaseModel):
    """Deep neural network classifier model. This is composed of several fully connected layers."""
    def __init__(self, train_path, patch_size=16, context_padding=28, load_images=True):
        """Initialise the model.

        Args:
            train_path (str): path to training data.
            patch_size (int): default=16 - the size of the patch to analyse.
            context_padding (int): default=28 - padding on each side of the analysed patch.
            load_images (bool): ONLY DISABLE FOR CODE CHECKS
        """
        super().__init__(train_path, patch_size, context_padding, load_images)
        logger.info("Generating DNN classifier ...")

        classifier_config = tf.contrib.learn.RunConfig(
            tf_random_seed=0,           # Make reproducible
            model_dir=os.path.join(file_path, os.path.normpath("../../results")),
            keep_checkpoint_max=3,
            save_checkpoints_secs=None,
            save_checkpoints_steps=1000)
        feature_columns = [tf.contrib.layers.real_valued_column("feature_col", dimension=75)]
        self.model = tf.estimator.DNNClassifier(
            hidden_units=[100, 150, 100, 50],
            feature_columns=feature_columns,
            n_classes=2,
            dropout=0.25,
            activation_fn=tf.nn.leaky_relu,
            config=classifier_config)

        # # The following can be set using a config file in ~/.keras/keras.json
        # if keras.backend.image_dim_ordering() == "tf":
        #     # Keras is using Tensorflow as backend
        #     input_dim = (self.window_size, self.window_size, 3)
        # else:
        #     # Keras is using Theano as backend
        #     input_dim = (3, self.window_size, self.window_size)

        # self.model = keras.Sequential()

        # self.model.add(keras.layers.Dense(units=512,
        #                                   input_shape=input_dim))
        # self.model.add(keras.layers.LeakyReLU(alpha=0.1))

        # self.model.add(keras.layers.Dense(units=1024))
        # self.model.add(keras.layers.LeakyReLU(alpha=0.1))

        # self.model.add(keras.layers.Dense(units=512))
        # self.model.add(keras.layers.LeakyReLU(alpha=0.1))

        # self.model.add(keras.layers.Dense(units=256))
        # self.model.add(keras.layers.LeakyReLU(alpha=0.1))

        # self.model.add(keras.layers.Dense(units=1,
        #                                   kernel_regularizer=keras.regularizers.l2(1e-6),
        #                                   activation="sigmoid"))



        if load_images:
            # Preload the images
            self.load_images()
        else:
            raise ValueError("load_images must be set to True")

        logger.info("Done")


    def _build_fn(self):
        """Return the model."""
        return self.model


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
            verbose = 1
        else:
            verbose = 0


        # self.model.compile(loss="binary_crossentropy",
        #                    optimizer="adam",
        #                    metrics=["accuracy"])

        # estimator = KerasClassifier(build_fn=self._build_fn(),
        #                             epochs=epochs,
        #                             batch_size=100,
        #                             verbose=verbose)

        # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

        logger.info("Starting training ...")
        # encoder = LabelEncoder()
        # for batch_data, verifier_data in self.create_batch(batch_size=100):
        #     encoder.fit(verifier_data[:,0])
        #     encoded_verifier = encoder.transform(verifier_data[:,0])
        #     res = cross_val_score(estimator, batch_data, encoded_verifier, cv=kfold)

        # print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

    @utility.overrides(cnn_base_model.CnnBaseModel)
    def evaluate(self):
        """Evaluate the efficiency of the model."""
        pass

    @utility.overrides(cnn_base_model.CnnBaseModel)
    def save(self, filename):
        pass
