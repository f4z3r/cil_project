#!/usr/bin/env python3

"""Tester module. This should implement unit tests for any important functionalities."""

import os
import unittest

from models import cnn_lr_d
from utils.commons import *

file_path = os.path.dirname(os.path.abspath(__file__))


class TestModels(unittest.TestCase):
    """Class testing the models."""

    @unittest.skip("Not properly implemented")
    def test_model_generation(self):
        """Test the model generations"""
        with self.assertRaises(ValueError):
            cnn_model1 = cnn_lr_d.CnnLrD(
                os.path.join(properties["TRAIN_DIR"],"data"),
                load_images=False)

    @unittest.skip("Not properly implemented")
    def test_cnn_lr_d_train(self):
        """Test the training function for CNN + LR + D model"""
        cnn_model1 = cnn_lr_d.CnnLrD(
            os.path.join(properties["TESTING_DIR"], os.path.normpath("training/data")))
        cnn_model1.train(True, epochs=1, steps=3, print_at_end=False)


def run():
    """Run the tests."""
    unittest.main(module="tests")
