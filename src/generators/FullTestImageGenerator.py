import glob
import os

import matplotlib.image as mpimg
import numpy as np


class FullTestImageGenerator:
    def __init__(self, path_images):
        x_files = sorted(glob.glob(os.path.join(path_images, "*.png")))
        self.file_iterator = iter(x_files)

    def generate_patch_sequential(self, batch_size=2, dimension=(400, 400)):
        while True:
            x_batch = np.empty((batch_size, dimension[0], dimension[1], 3))

            for idx in range(batch_size):
                x = mpimg.imread(next(self.file_iterator))
                x_batch[idx] = x

            yield x_batch
