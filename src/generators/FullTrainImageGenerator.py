import glob
import os
import random

import matplotlib.image as mpimg
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


class FullTrainImageGenerator:
    def __init__(self, path_images, path_labels, augment=True):
        x_files = sorted(glob.glob(os.path.join(path_images, "*.png")))
        y_files = sorted(glob.glob(os.path.join(path_labels, "*.png")))
        self.x_shape = mpimg.imread(x_files[0]).shape

        self.size = len(x_files)

        self.X = np.empty((self.size, self.x_shape[0], self.x_shape[1], self.x_shape[2]))
        self.Y = np.empty((self.size, self.x_shape[0], self.x_shape[1], 1))

        for idx, (x_file_path, y_file_path) in enumerate(zip(x_files, y_files)):
            self.X[idx] = mpimg.imread(x_file_path)
            self.Y[idx] = mpimg.imread(y_file_path).reshape(self.x_shape[0], self.x_shape[1], 1)

        data_gen_args = dict(rotation_range=360.,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             fill_mode="reflect")
        self.image_datagen = ImageDataGenerator(**data_gen_args)
        self.mask_datagen = ImageDataGenerator(**data_gen_args)

        # Provide the same seed and keyword arguments to the fit and flow methods
        # seed = random.randint(0, 1000)
        seed = 1  # TODO check if this should be static or random
        self.image_datagen.fit(self.X, augment=augment, seed=seed)
        self.mask_datagen.fit(self.Y, augment=augment, seed=seed)

        print('Generator initialized with {} pictures'.format(self.size))

    def generate_patch(self, batch_size=5):
        seed = random.randint(0, 100000)
        x = self.image_datagen.flow(self.X, batch_size=batch_size, seed=seed)
        y = self.mask_datagen.flow(self.Y, batch_size=batch_size, seed=seed)
        return zip(x, y)
