#!/usr/bin/env python3

# Utility module containing helper functions.
#
# Requires Pillow,

import numpy as np
import os
import sys
import glob
import matplotlib.image as mpimg

def load_image(filepath):
    """Loads an image from disk and return it as a 2D pixel array.

    Args:
        filename (str): full path to image to be loaded

    Returns:
        numpy.ndarray : Array containing the pixel colour information.
    """
    return mpimg.imread(filepath)

def augment_img_set(path):
    """Augments the set of test images via rotations and reflections.

    Args:
        path (str): path to image folder.

    Returns:
        Nothing.
    """
    for file in glob.glob(os.path.join(path, "*.jpg")):
        filename_no_ext = "".join(os.path.basename(file).split(".")[:-1])
        filepath_no_ext = os.path.join(path, filename_no_ext)
        img = load_image(file)
        cpy_img = img

        # Generate 3 rotated copies
        for idx in range(3):
            cpy_img = np.rot90(cpy_img)
            mpimg.imsave("{}_{}.jpg".format(filepath_no_ext, idx), cpy_img)

        # Generate 4 rotated and flipped copies
        flip_img = np.flipud(img)
        for idx in range(3, 7):
            flip_img = np.rot90(flip_img)
            mpimg.imsave("{}_{}.jpg".format(filepath_no_ext, idx), flip_img)




