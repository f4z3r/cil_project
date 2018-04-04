import random

import keras
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence


'''
Augments image data set and masks using Keras ImageDataGenerator.
Runs on own thread because of keras.utils.Sequence
'''
class RoadSequenceGenerator(Sequence):
    def __init__(self, image_path, image_folder, mask_folder, batch_size, seed=1, dim=72, patch=16,
                 threshold_mask=0.15, allow_rotation=True):
        self.image_path = image_path
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.batch_size = batch_size

        rotation = 0
        if allow_rotation:
            rotation = 180

        self.datagen = ImageDataGenerator(
            rotation_range=rotation,
            horizontal_flip=True,
            fill_mode='nearest',
            rescale=1. / 255,
        )

        self.seed = seed
        self.dim = dim
        self.patch = patch
        self.threshold_mask = threshold_mask

        # hack to get same seed for image and mask generator
        random.seed(seed)
        datagen_seed = random.randint(0, 1000000)

        self.image_generator = self.datagen.flow_from_directory(
            self.image_path,
            target_size=(400, 400),
            batch_size=self.batch_size,
            class_mode=None,
            classes=[self.image_folder],
            seed=datagen_seed,
        )

        self.mask_generator = self.datagen.flow_from_directory(
            self.image_path,
            target_size=(400, 400),
            batch_size=self.batch_size,
            class_mode=None,
            classes=[self.mask_folder],
            seed=datagen_seed,
        )

    def __len__(self):
        return len(self.image_generator)

    def __getitem__(self, idx):
        images = self.image_generator.next()
        masks = self.mask_generator.next()

        # find center
        n_pictures, height, width, _ = images.shape
        half_dim = self.dim // 2
        half_patch = self.patch // 2
        x = random.randint(half_dim, (height - half_dim))
        y = random.randint(half_dim, (width - half_dim))

        # crop the images using the dim size (e.g. 72 pixel)
        images_cropped = images[:, (x - half_dim): (x + half_dim), (y - half_dim): (y + half_dim), :]
        # crop the mask using the patch size (e.g. 16 pixel)
        masks_patch = masks[:, (x - half_patch): (x + half_patch), (y - half_patch): (y + half_patch), :]

        # calculate mean. If mean is over threshold mark as true
        mask_means = np.mean(masks_patch, axis=(1, 2, 3))
        y = mask_means > self.threshold_mask

        return images_cropped, keras.utils.to_categorical(y, num_classes=2)
