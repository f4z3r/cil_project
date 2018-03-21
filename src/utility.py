#!/usr/bin/env python3

# Utility module containing helper functions.
#
# Requires Pillow,

import numpy as np
import os
import sys
from PIL import Image

def load_image(filename):
    """Loads an image from disk and return it."""
    return Image.open(filename).load()



