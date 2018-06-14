import os
import sys

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage import imread

class ImageToPatchGenerator(object):
    def __init__(self, path_to_data, path_to_test, steps_per_epoch, validation_steps, augmentation, train_valid_split=0.92):
        self.input_size = 608
        self.patch_size = 16
        self.foreground_threshold = 0.25

        # path_train = os.path.join(path_to_data, train_valid_split, "train")
        # path_valid = os.path.join(path_to_data, train_valid_split, "valid")
        # path_test = os.path.join(path_to_test, "test")

        self.images_train, self.masks_train = self.load_images(path_to_data, self.input_size, "train", train_valid_split)
        self.images_valid, self.masks_valid = self.load_images(path_to_data, self.input_size, "valid", train_valid_split)
        self.images_test, _ = self.load_images(path_to_test, self.input_size, "test", 0)
        self.images_test_names = sorted(os.listdir(os.path.join(path_to_test, 'data')))

        if augmentation:
            self.images_train_pre, self.masks_train_pre = self.augment(self.images_train, self.masks_train, augmentation)
            self.images_valid_pre, self.masks_valid_pre = self.augment(self.images_valid, self.masks_valid, augmentation)
        else:
            self.images_train_pre, self.masks_train_pre = self.images_train, self.masks_train
            self.images_valid_pre, self.masks_valid_pre = self.images_valid, self.masks_valid

        self.train_counter = 0
        self.valid_counter = 0

        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps

        self.augmentation = augmentation

    def next_batch(self, type="train", batch_size=8):
        while True:
            if type == "valid":
                image_set = self.images_valid_pre
                masks_set = self.masks_valid_pre
                if self.valid_counter == self.validation_steps - 1:
                    self.images_valid_pre, self.masks_valid_pre = self.augment(self.images_valid, self.masks_valid, self.augmentation)
                    self.valid_counter = 0
                else:
                    self.valid_counter = self.valid_counter + 1

            else:
                image_set = self.images_train_pre
                masks_set = self.masks_train_pre
                if self.train_counter == self.steps_per_epoch - 1:
                    self.images_train_pre, self.masks_train_pre = self.augment(self.images_train, self.masks_train, self.augmentation)
                    self.train_counter = 0
                else:
                    self.train_counter = self.train_counter + 1

            images = np.empty((batch_size, self.input_size, self.input_size, 3))
            masks = np.empty((batch_size, self.input_size // self.patch_size, self.input_size // self.patch_size))

            random_indices = np.random.choice(len(image_set), batch_size)
            random_rotation = np.random.choice(4, batch_size)

            random_flip = np.random.choice(2, batch_size)

            for i, idx in enumerate(random_indices):
                image = image_set[idx]
                mask = masks_set[idx]

                image = np.rot90(image, random_rotation[i])
                mask = np.rot90(mask, random_rotation[i])

                if random_flip[i] == 1:
                    image = np.fliplr(image)
                    mask = np.fliplr(mask)

                patches = self.extract_blocks(mask, (self.patch_size, self.patch_size))
                mask = (patches.mean(axis=(1, 2)) > self.foreground_threshold)
                mask = mask.reshape((608 // 16, 608 // 16))

                images[i] = image
                masks[i] = mask

            masks = masks.reshape(
                (batch_size, self.input_size // self.patch_size, self.input_size // self.patch_size, 1))

            yield images, masks

    @staticmethod
    def extract_blocks(a, blocksize):
        M, N = a.shape
        b0, b1 = blocksize
        return a.reshape(M // b0, b0, N // b1, b1).swapaxes(1, 2).reshape(-1, b0, b1)

    @staticmethod
    def augment(images, masks, augmentation):
        if augmentation:
            augmentation_size = len(images)
            datagen = ImageDataGenerator(
                width_shift_range=0.1,
                height_shift_range=0.1,
                rotation_range=360,
                shear_range=0.1,
                zoom_range=0.2,
                fill_mode="reflect"
            )

            seed = np.random.choice(100000, 1)
            d_images = datagen.flow(images, batch_size=augmentation_size, shuffle=False, seed=seed)
            d_masks = datagen.flow(masks.reshape((augmentation_size, 608, 608, 1)), batch_size=augmentation_size, shuffle=False, seed=seed)

            images_augmented = next(d_images)
            masks_augmented = next(d_masks).reshape((augmentation_size, 608, 608))

            return images_augmented, masks_augmented
        else:
            return images, masks

    @staticmethod
    def load_images(path, input_size, type, train_valid_split):
        image_files = sorted(os.listdir(os.path.join(path, 'data')))
        mask_files = sorted(
            os.listdir(os.path.join(path, 'verify')) if os.path.exists(os.path.join(path, 'verify')) else [])


        if type == "train":
            image_files = image_files[0:int(train_valid_split * 100)]
            mask_files = mask_files[0:int(train_valid_split * 100)]
        elif type == "valid":
            image_files = image_files[int(train_valid_split * 100):]
            mask_files = mask_files[int(train_valid_split * 100):]

        images = np.zeros((len(image_files), input_size, input_size, 3))
        for idx, image_file in enumerate(image_files):
            image = imread(os.path.join(path, 'data', image_file))
            images[idx, :, :, :] = image / 255.

        if len(mask_files) > 0:
            masks = np.zeros((len(image_files), input_size, input_size))
            for idx, mask_file in enumerate(mask_files):
                mask = imread(os.path.join(path, 'verify', mask_file))
                masks[idx, :, :] = mask / 255.

        else:
            masks = []

        print("Loaded {} images and {} masks from {}".format(len(images), len(masks), path))
        return images, masks


if __name__ == "__main__":
    generator = ImageToPatchGenerator("../assets/608/training", "../assets/tests", 200, 50, True)
    print(next(generator.next_batch("train")))
