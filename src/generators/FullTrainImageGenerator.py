import glob
import os

import matplotlib.image as mpimg
import numpy as np


class FullTrainImageGenerator:
    def __init__(self, path_images, path_labels):
        x_files = sorted(glob.glob(os.path.join(path_images, "*.png")))
        y_files = sorted(glob.glob(os.path.join(path_labels, "*.png")))
        self.x_shape = mpimg.imread(x_files[0]).shape

        self.size = len(x_files)

        self.X = np.empty((self.size, self.x_shape[0], self.x_shape[1], self.x_shape[2]))
        self.Y = np.empty((self.size, self.x_shape[0], self.x_shape[1]))

        for idx, (x_file_path, y_file_path) in enumerate(zip(x_files, y_files)):
            self.X[idx] = mpimg.imread(x_file_path)
            self.Y[idx] = mpimg.imread(y_file_path)

        self.Y = self.Y / self.Y.max()  # normalize data
        print('Generator initialized with {} pictures'.format(self.size))

    def generate_patch(self, batch_size=2):
        while True:
            x_batch = np.empty((batch_size, self.x_shape[0], self.x_shape[1], self.x_shape[2]))
            y_batch = np.empty((batch_size, self.x_shape[0], self.x_shape[1]))

            for idx in range(batch_size):
                img_num = np.random.choice(self.X.shape[0])
                x = self.X[img_num]
                y = self.Y[img_num]

                x_batch[idx] = x
                y_batch[idx] = y

            y_batch = y_batch.reshape((batch_size, self.x_shape[0], self.x_shape[1], 1))

            yield (x_batch, y_batch)
