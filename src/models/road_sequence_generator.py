import glob

import keras
import numpy as np
from keras.utils import Sequence
from os import path
from scipy.misc import imread

'''
Runs on own thread because of keras.utils.Sequence
'''
class RoadSequenceGenerator(Sequence):
    def __init__(self, image_path, image_folder, mask_folder, batch_size, dim=72, patch=16, pad=28,
                 threshold_mask=0.25):
        print('initializing')
        self.batch_size = batch_size
        self.dim = dim
        self.patch = patch
        self.threshold_mask = threshold_mask

        self.pad = pad

        image_directory = glob.glob(path.join(path.join(path.normpath(image_path), image_folder), '*.png'))
        mask_directory = glob.glob(path.join(path.join(path.normpath(image_path), mask_folder), '*.png'))

        image_count = len(image_directory)
        first = imread(image_directory[0])

        self.image_set = np.empty((image_count,
                                   first.shape[0] + 2 * pad,
                                   first.shape[1] + 2 * pad,
                                   first.shape[2]))
        self.mask_set = np.empty((image_count, first.shape[0], first.shape[1]))

        for idx, (file, mask) in enumerate(zip(image_directory, mask_directory)):
            self.image_set[idx] = np.pad(imread(file), ((pad, pad), (pad, pad), (0, 0)), mode="reflect") / 255.
            self.mask_set[idx] = imread(mask) / 255.

    def __len__(self):
        return self.image_set.shape[0]

    def __getitem__(self, idx):
        batch_image = np.empty((self.batch_size, self.dim, self.dim, 3))
        batch_label = np.empty((self.batch_size, 2))

        for idx in range(self.batch_size):
            img_num = np.random.choice(self.image_set.shape[0])

            h = (np.random.choice(
                self.image_set[img_num].shape[1] - 2 * self.pad) // self.patch) * self.patch + self.pad
            w = (np.random.choice(
                self.image_set[img_num].shape[0] - 2 * self.pad) // self.patch) * self.patch + self.pad

            if len(self.image_set[img_num].shape) == 3:
                data_patch = self.image_set[img_num][h - self.pad:h + self.patch + self.pad,
                             w - self.pad:w + self.patch + self.pad, :]
            else:
                data_patch = self.image_set[img_num][h - self.pad:h + self.patch + self.pad,
                             w - self.pad:w + self.patch + self.pad]

            mask_patch = self.mask_set[img_num][h - self.pad:h + self.patch - self.pad,
                         w - self.pad:w + self.patch - self.pad]

            label = (np.mean(mask_patch) > 0.25) * 1
            label = keras.utils.to_categorical(label, num_classes=2)
            batch_image[idx] = data_patch
            batch_label[idx] = label

        return batch_image, batch_label
