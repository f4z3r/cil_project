#!/usr/bin/env python3

# Tester module. This should implement unit tests for any important functionalities.

import os, sys
import unittest

import utility

class TestUtilities(unittest.TestCase):
    """Class testing all utility functions."""

    # The following is an implementation example for a test
    def test_load_image(self):
        self.assertEqual(2, 2)


class TestCnnModel(unittest.TestCase):
    """Class testing the CNN model."""


if __name__ == "__main__":
    unittest.main()
