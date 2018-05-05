import glob
import logging
import os

import keras
import matplotlib.image as mpimg
import numpy as np

logger = logging.getLogger("cil_project.src.generators.PatchTrainImageGenerator")


class PatchTrainImageGenerator:
    def __init__(self, path_to_images, path_to_groundtruth, window_size=72, patch_size=16, threshold=0.25):
        padding = (window_size - patch_size) // 2

        data_files = glob.glob(os.path.join(path_to_images, "*.png"))
        mask_files = glob.glob(os.path.join(path_to_groundtruth, "*.png"))
        image_count = len(data_files)
        first = mpimg.imread(data_files[0])

        data_set = np.empty((image_count,
                             first.shape[0] + 2 * padding,
                             first.shape[1] + 2 * padding,
                             first.shape[2]))
        verifier_set = np.empty((image_count, first.shape[0] + 2 * padding, first.shape[1] + 2 * padding))

        for idx, (file, mask_file) in enumerate(zip(data_files, mask_files)):
            data_set[idx] = np.pad(mpimg.imread(file), ((padding, padding), (padding, padding), (0, 0)), mode="reflect")
            verifier_set[idx] = np.pad(mpimg.imread(mask_file), ((padding, padding), (padding, padding)),
                                       mode="reflect")

        verifier_set = verifier_set / verifier_set.max()  # normalize data

        self.data_set = data_set
        self.verifier_set = verifier_set
        self.patch_size = patch_size
        self.window_size = window_size
        self.padding = padding
        self.threshold = threshold

        logger.info('PatchImageGenerator initialized with {} pictures'.format(image_count))

    def get_random_image_patch(self, data_img, verifier_img, size, stride):
        h = (np.random.choice(verifier_img.shape[1]) // stride) * stride
        w = (np.random.choice(verifier_img.shape[0]) // stride) * stride

        data_patch = data_img[h:h + size, w:w + size, :]
        verifier_patch = verifier_img[h:h + stride, w:w + stride]

        return data_patch, verifier_patch

    def generate_patch(self, batch_size=100, four_dim=False, augmentation=True):
        window_size = self.window_size
        patch_size = self.patch_size
        while True:
            batch_data = np.empty((batch_size, window_size, window_size, 3))
            batch_verifier = np.empty((batch_size, 2))

            for idx in range(batch_size):
                img_num = np.random.choice(self.data_set.shape[0])
                image = self.data_set[img_num]
                ground_truth = self.verifier_set[img_num]

                # Sample a random window from the image
                center = np.random.randint(window_size // 2, image.shape[0] - window_size // 2, 2)
                sub_image = image[center[0] - window_size // 2:center[0] + window_size // 2,
                            center[1] - window_size // 2:center[1] + window_size // 2]
                gt_sub_image = ground_truth[center[0] - patch_size // 2:center[0] + patch_size // 2,
                               center[1] - patch_size // 2:center[1] + patch_size // 2]

                label = (np.array([np.mean(gt_sub_image)]) > self.threshold) * 1


                if augmentation:
                    # Image augmentation
                    # Random flip
                    if np.random.choice(2) == 0:
                        # Flip vertically
                        sub_image = np.flipud(sub_image)
                    if np.random.choice(2) == 0:
                        # Flip horizontally
                        sub_image = np.fliplr(sub_image)

                    # Random rotation in steps of 90°
                    num_rot = np.random.choice(4)
                    sub_image = np.rot90(sub_image, num_rot)
                    image = sub_image

                label = keras.utils.to_categorical(label, num_classes=2)
                batch_data[idx] = image
                batch_verifier[idx] = label

            if four_dim:
                batch_data = batch_data.reshape(batch_size, self.window_size, self.window_size, 3, 1)

            yield (batch_data, batch_verifier)

    def input_dim(self, four_dim=False):
        if four_dim:
            return self.window_size, self.window_size, 3, 1
        return self.window_size, self.window_size, 3
