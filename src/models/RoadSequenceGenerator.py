import random

import keras
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence


class RoadSequenceGenerator(Sequence):
    def __init__(self, image_path, image_folder, mask_folder, batch_size, seed=1, dim=16, threshold_mask=0.25):
        self.image_path = image_path
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.batch_size = batch_size

        self.datagen = ImageDataGenerator(
            rotation_range=180,
            horizontal_flip=True,
            fill_mode='nearest',
            rescale=1. / 255,
        )

        self.seed = seed
        self.dim = dim
        self.threshold_mask = threshold_mask

        random.seed(seed)
        datagen_seed = random.randint(0, 1000000)

        self.image_generator = self.datagen.flow_from_directory(
            self.image_path,
            target_size=(400, 400),
            batch_size=self.batch_size,
            class_mode=None,
            classes=[self.image_folder],
            seed=datagen_seed,
            #save_to_dir='preview', save_prefix='image', save_format='png',
        )

        self.mask_generator = self.datagen.flow_from_directory(
            self.image_path,
            target_size=(400, 400),
            batch_size=self.batch_size,
            class_mode=None,
            classes=[self.mask_folder],
            seed=datagen_seed,
            #save_to_dir='preview', save_prefix='mask', save_format='png',
        )

    def __len__(self):
        return len(self.image_generator)

    def __getitem__(self, idx):
        images = self.image_generator.next()
        masks = self.mask_generator.next()

        # crop image and mask
        n_pictures, height, width, _ = images.shape
        x = random.randint(0, height - self.dim)
        y = random.randint(0, width - self.dim)
        images_cropped = images[:, x:(x + self.dim), y:(y + self.dim), :]
        masks_cropped = masks[:, x:(x + self.dim), y:(y + self.dim), :]

        # calculate mean. If mean is over threshold mark as true
        mask_means = np.mean(masks_cropped, axis=(1, 2, 3))
        y = mask_means > self.threshold_mask

        return images_cropped, keras.utils.to_categorical(y, num_classes=2)
