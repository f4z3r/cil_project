import glob

import keras
import numpy as np
import scipy
from keras.utils import Sequence
from os.path import join, normpath

'''
Runs on own thread because of keras.utils.Sequence
'''


class RoadSequenceGenerator(Sequence):
    def __init__(self, image_path, image_folder, mask_folder, batch_size, dim=72, patch=16, threshold_mask=0.25):
        self.batch_size = batch_size
        self.dim = dim
        self.patch = patch
        self.threshold_mask = threshold_mask

        padding = dim // 2
        patch_padding = self.patch // 2

        image_directory = glob.glob(join(join(normpath(image_path), image_folder), '*.png'))
        mask_directory = glob.glob(join(join(normpath(image_path), mask_folder), '*.png'))

        first = scipy.ndimage.imread(image_directory[0])
        image_count = len(image_directory)

        self.image_set = np.empty((image_count, first.shape[0] + 2 * padding, first.shape[1] + 2 * padding, 3))
        self.mask_set = np.empty((image_count, first.shape[0] + 2 * patch_padding, first.shape[1] + 2 * patch_padding))

        for idx, (image_file, mask_file) in enumerate(zip(image_directory, mask_directory)):
            self.image_set[idx] = np.pad(scipy.ndimage.imread(image_file),
                                         ((padding, padding), (padding, padding), (0, 0)), mode="reflect")
            mask = np.pad(scipy.ndimage.imread(mask_file),
                          ((patch_padding, patch_padding), (patch_padding, patch_padding)),
                          mode="reflect")
            self.mask_set[idx] = mask / 255.

    def __len__(self):
        return self.image_set.shape[0] * 4

    def __getitem__(self, idx):
        images = np.empty((self.batch_size, self.dim, self.dim, 3))
        masks = np.empty((self.batch_size, 2))

        padding = self.dim // 2
        patch_padding = self.patch // 2

        for idx in range(self.batch_size):
            image_idx = np.random.randint(0, self.image_set.shape[0])
            image = self.image_set[image_idx]
            mask = self.mask_set[image_idx]

            # random center
            c = np.random.randint(0, image.shape[0] - self.dim, 2)
            c_image = c + [padding, padding]
            c_mask = c + [patch_padding, patch_padding]
            image_patch = image[(c_image[0] - padding):(c_image[0] + padding),
                          (c_image[1] - padding):(c_image[1] + padding), :]
            mask_patch = mask[(c_mask[0] - patch_padding):(c_mask[0] + patch_padding),
                         (c_mask[1] - patch_padding):(c_mask[1] + patch_padding)]

            label = np.mean(mask_patch) > self.threshold_mask
            label_categorized = keras.utils.to_categorical(label, num_classes=2)
            images[idx] = image_patch
            masks[idx] = label_categorized
        return images, masks
