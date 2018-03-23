#!/usr/bin/env python3

"""Utility module containing helper functions."""

import numpy as np
import os, sys, glob, logging
import matplotlib.image as mpimg
from PIL import Image

logger = logging.getLogger("cil_project.utility")

def load_image(filepath):
    """Loads an image from disk and return it as a 2D pixel array.

    Args:
        filename (str): full path to image to be loaded.

    Returns:
        numpy.ndarray: Array containing the pixel colour information.
    """
    return mpimg.imread(filepath)

def augment_img_set(path):
    """Augments the set of test images via rotations and reflections.

    Args:
        path (str): path to image folder.

    Returns:
        Nothing.
    """
    for file in glob.glob(os.path.join(path, "*.png")):
        logger.info("Augmenting image: {}".format(file))

        filename_no_ext = "".join(os.path.basename(file).split(".")[:-1])
        filepath_no_ext = os.path.join(path, filename_no_ext)
        img = Image.open(file)
        cpy_img = img

        # Generate 3 rotated copies
        for idx in range(3):
            cpy_img = cpy_img.rotate(90)
            cpy_img.save("{}_{}.png".format(filepath_no_ext, idx))

        # Generate 4 rotated and flipped copies
        flip_img = img.transpose(Image.FLIP_TOP_BOTTOM)
        flip_img.save("{}_{}.png".format(filepath_no_ext, 3))
        for idx in range(4, 7):
            flip_img = flip_img.rotate(90)
            flip_img.save("{}_{}.png".format(filepath_no_ext, idx))

def generate_patches_with_pad(img, size, stride, pad):
    """Creates patches of size (size + pad, size+ pad) pixels from image img and shifting by stride. Note that the
    stride is not allowed to be larger than size as this would entail missing pixels.

    Args:
        img (numpy.ndarray): the image used to generate the patches.
        size (int): the size of interior window of the patches.
        stride (int): the amount of pixels to shift between patches.
        pad (int): padding to be added on each size of the patch.

    Returns:
        [numpy.ndarray]: a list of pixel matrices corresponding to the patches.
    """
    assert stride <= size, "Stride should not be larger than size."
    patch_list = []

    if len(img.shape) == 3:
        img_padded = np.pad(img, ((pad, pad), (pad, pad), (0,0)), mode="reflect")
    elif len(img.shape) == 2:
        img_padded = np.pad(img, ((pad, pad), (pad, pad)), mode="reflect")
    else:
        raise TypeError("The image provided should be a 2 or 3 dimensional array.")

    img_width = img.shape[0]
    img_height = img.shape[1]

    for h in range(pad, img_height + pad, stride):
        for w in range(pad, img_width + pad, stride):
            if len(img.shape) == 3:
                patch = img_padded[h - pad: h + size + pad, w - pad: w + size + pad, :]
            else:
                patch = img_padded[h - pad: h + size + pad, w - pad: w + size + pad]
            patch_list.append(patch)
    return patch_list
