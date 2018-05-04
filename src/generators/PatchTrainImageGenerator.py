import glob
import logging
import os

import keras
import matplotlib.image as mpimg
import numpy as np
import scipy

logger = logging.getLogger("cil_project.src.generators.PatchTrainImageGenerator")


class PatchTrainImageGenerator:
    def __init__(self, path_to_images, path_to_groundtruth, pad=28, patch_size=16):
        data_files = glob.glob(os.path.join(path_to_images, "*.png"))
        mask_files = glob.glob(os.path.join(path_to_groundtruth, "*.png"))
        image_count = len(data_files)
        first = mpimg.imread(data_files[0])

        data_set = np.empty((image_count,
                             first.shape[0] + 2 * pad,
                             first.shape[1] + 2 * pad,
                             first.shape[2]))
        verifier_set = np.empty((image_count, first.shape[0], first.shape[1]))

        for idx, (file, mask_file) in enumerate(zip(data_files, mask_files)):
            data_set[idx] = np.pad(mpimg.imread(file), ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
            verifier_set[idx] = mpimg.imread(mask_file)

        verifier_set = verifier_set / verifier_set.max()  # normalize data
        self.data_set = data_set
        self.verifier_set = verifier_set

        self.window_size = patch_size + 2 * pad
        self.patch_size = patch_size

        logger.info('PatchImageGenerator initialized with {} pictures'.format(image_count))

    def get_random_image_patch(self, data_img, verifier_img, size, stride):
        h = (np.random.choice(verifier_img.shape[1]) // stride) * stride
        w = (np.random.choice(verifier_img.shape[0]) // stride) * stride

        data_patch = data_img[h:h + size, w:w + size, :]
        verifier_patch = verifier_img[h:h + stride, w:w + stride]

        return data_patch, verifier_patch

    def generate_patch(self, batch_size=100, four_dim=False, augmentation=False):
        window_size = self.window_size
        patch_size = self.patch_size
        while True:
            batch_data = np.empty((batch_size, window_size, window_size, 3))
            batch_verifier = np.empty((batch_size, 2))

            for idx in range(batch_size):
                img_num = np.random.choice(self.data_set.shape[0])
                image = self.data_set[img_num]
                ground_truth = self.verifier_set[img_num]

                if augmentation:
                    if np.random.choice([False, True]):
                        # randomly flip
                        image = image[:, ::-1]
                        ground_truth = ground_truth[:, ::-1]

                    rotation = np.random.random_sample() * 360
                    image = scipy.ndimage.interpolation.rotate(image, rotation, mode="reflect", reshape=False)
                    ground_truth = scipy.ndimage.interpolation.rotate(ground_truth, rotation, mode="reflect",
                                                                      reshape=False)

                data_patch, veri_patch = self.get_random_image_patch(image, ground_truth, window_size, patch_size)

                label = (np.mean(veri_patch) > 0.25) * 1
                label = keras.utils.to_categorical(label, num_classes=2)
                batch_data[idx] = data_patch
                batch_verifier[idx] = label

            if four_dim:
                batch_data = batch_data.reshape(batch_size, self.window_size, self.window_size, 3, 1)

            yield (batch_data, batch_verifier)

    def input_dim(self, four_dim=False):
        if four_dim:
            return self.window_size, self.window_size, 3, 1
        return self.window_size, self.window_size, 3
