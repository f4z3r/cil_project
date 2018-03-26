#!usr/bin/env python3

import os, sys, logging
import numpy as np
import tensorflow as tf

import utility
from models import model

logger = logging.getLogger("cil_project.models.dnn_classifier")

class DnnClassifier(model.Model):
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

        if load_images:
            # Preload the images
            self.load_images()
        else:
            raise ValueError("load_images must be set to True")

        logger.info("Done")

    @utility.overrides(model.Model)
    def train(self, verbosity):
        pass

    @utility.overrides(model.Model)
    def save(self, filename):
        pass
