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
    """
    for file in glob.glob(os.path.join(path, "*.png")):
        logger.debug("Augmenting image: {}".format(file))

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
    """Creates patches of size (size + 2 * pad, size + 2 * pad) pixels from image img and shifting
    by stride. Note that the stride is not allowed to be larger than size as this would entail
    missing pixels.

    Args:
        img (numpy.ndarray): the image used to generate the patches.
        size (int): the size of interior window of the patches.
        stride (int): the amount of pixels to shift between patches.
        pad (int): padding to be added on each size of the patch.

    Returns:
        numpy.ndarray: an array of pixel matrices corresponding to the patches.
    """
    assert stride <= size, "Stride should not be larger than size."
    patch_list = []

    img_padded = pad_image(img, pad)

    img_width = img.shape[0]
    img_height = img.shape[1]

    for h in range(pad, img_height + pad, stride):
        for w in range(pad, img_width + pad, stride):
            if len(img.shape) == 3:
                patch = img_padded[h - pad:h + size + pad, w - pad:w + size + pad, :]
            else:
                patch = img_padded[h - pad:h + size + pad, w - pad:w + size + pad]
            patch_list.append(patch)
    return np.asarray(patch_list)

def get_random_image_patch(data_img, verifier_img, size, stride, pad):
    """Creates a random patches of size (size + 2 * pad, size + 2 * pad) pixels from img and returns
    it with the corresponding verification patch from verifier.

    Args:
        img (numpy.ndarray): image from which to exctract patch.
        verifier (numpy.ndarray): image from which to exctract verifier patch.
        size (int): the size of interior window of the patches.
        stride (int): the amount of pixels to shift between patches.
        pad (int): padding to be added on each size of the patch.

    Returns:
        numpy.ndarray: the patch with padding.
        numpy.ndarray: the corresponding verifier patch (note this has total size (size x size).
    """
    img_padded = pad_image(data_img, pad)

    h = (np.random.choice(data_img.shape[1]) // stride) * stride + pad
    w = (np.random.choice(data_img.shape[0]) // stride) * stride + pad

    if len(data_img.shape) == 3:
        data_patch = img_padded[h - pad:h + size + pad, w - pad:w + size + pad, :]
    else:
        data_patch = img_padded[h - pad:h + size + pad, w - pad:w + size + pad]

    verifier_patch = verifier_img[h - pad:h + size - pad, w - pad:w + size - pad]

    return data_patch, verifier_patch


def load_training_set(img_path):
    """Load the training set into memory and pass it on.

    Args:
        path (str): path to the training set. Note that the folder structure must follow the
                    guidelines given in README.md.
    """
    logger.info("Loading image sets into memory ...")
    data_files = glob.glob(os.path.join(img_path, "*.png"))
    image_count = len(data_files)
    first = load_image(data_files[0])

    logger.info("Found {} images with {}x{} resolution.".format(image_count,
                                                                first.shape[1],
                                                                first.shape[0]))

    data_set = np.empty((image_count, first.shape[0], first.shape[1], first.shape[2]))
    verifier_set = np.empty((image_count, first.shape[0], first.shape[1], first.shape[2]))

    prog_bar = ProgressBar(image_count)
    prog_bar.print()

    for idx, file in enumerate(data_files):
        data_set[idx] = load_image(file)
        verifier_file = file.replace(os.path.normpath("assets/trainig/data"),
                                     os.path.normpath("assets/trainig/verify"))
        verifier_set[idx] = load_image(verifier_file)

        prog_bar.inc_and_print()

    return data_set, verifier_set


def pad_image(img, pad):
    """Pad image `img` by padding `pad`.

    Args:
        img (numpy.ndarray): the image to be padded.
        pad (int): the padding to be added on each side of the image.
    """
    assert pad >= 0, "Padding should be positive."

    if len(img.shape) == 3:
        img_padded = np.pad(img, ((pad, pad), (pad, pad), (0,0)), mode="reflect")
    elif len(img.shape) == 2:
        img_padded = np.pad(img, ((pad, pad), (pad, pad)), mode="reflect")
    else:
        raise TypeError("The image provided should be a 2 or 3 dimensional array.")

    return img_padded



####################################################################################################
# PROGRESS BAR
####################################################################################################
class ProgressBar:
    """
    Object that when printed to the terminal window prints a progress bar.

    Available methods:
        - __init__(total, prefix='', suffix='', decimals=1, length=50, fill='█')
        - print()
        - increment()
        - inc_and_print()
    """
    def __init__(self, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
        """
        Creates a ProgressBar object.

        Args:
            total (int): Total iterations.
            prefix (str): (default: '') Prefix string.
            suffix (str): (default: '') Suffix string
            decimals (int): (default: 1) Percentage precision parameter.
            length (int): (default: 50) Character length of bar.
            fill (str): (default: '█') Bar fill character.
        """
        self._iteration = 0
        self._total = total
        self._prefix = prefix
        self._suffix = suffix
        self._decimals = decimals
        self._length = length
        self._fill = fill

    def print(self):
        """
        Prints the progress bar. Note that the cursor is moved back to
        the beginning of the line to overwrite the progress bar next time it is
        printed.
        """
        percent = ("{0:." + str(self._decimals) + "f}").format(100 *
                  (self._iteration / float(self._total)))
        filled_length = int(self._length * self._iteration // self._total)
        bar = self._fill * filled_length + '-' * (self._length - filled_length)
        print('\r%s |%s| %s%% %s' % (self._prefix, bar, percent, self._suffix), end = '\r')

        # Print New Line on complete
        if self._iteration >= self._total:
            print()

    def increment(self):
        """
        Increments the iteration.
        """
        self._iteration += 1

    def inc_and_print(self):
        """
        Increments the iteration and prints the progress bar to the screen.
        """
        self.increment()
        self.print()
