import glob
import os

import keras
import matplotlib.image as mpimg
import numpy as np


class PatchImageGenerator:
    def __init__(self, path_to_images, path_to_groundtruth, pad=28, patch_size=16, context_padding=28):
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

        self.data_set = data_set
        self.verifier_set = verifier_set

        self.window_size = patch_size + 2 * context_padding
        self.patch_size = patch_size
        self.context_padding = context_padding

        print('PatchImageGenerator initialized with {} pictures'.format(image_count))

    def get_random_image_patch(self, data_img, verifier_img, size, stride, pad):
        h = (np.random.choice(data_img.shape[1] - 2 * pad) // stride) * stride + pad
        w = (np.random.choice(data_img.shape[0] - 2 * pad) // stride) * stride + pad

        if len(data_img.shape) == 3:
            data_patch = data_img[h - pad:h + size + pad, w - pad:w + size + pad, :]
        else:
            data_patch = data_img[h - pad:h + size + pad, w - pad:w + size + pad]

        verifier_patch = verifier_img[h - pad:h + size - pad, w - pad:w + size - pad]

        return data_patch, verifier_patch

    def generate_patch(self, batch_size=100, allow_augmentation=True):
        window_size = self.window_size
        patch_size = self.patch_size
        context_padding = self.context_padding
        while True:
            batch_data = np.empty((batch_size, window_size, window_size, 3))
            batch_verifier = np.empty((batch_size, 2))

            for idx in range(batch_size):
                img_num = np.random.choice(self.data_set.shape[0])
                data_patch, veri_patch = self.get_random_image_patch(self.data_set[img_num],
                                                                     self.verifier_set[img_num],
                                                                     patch_size,
                                                                     patch_size,
                                                                     context_padding)
                label = (np.mean(veri_patch) > 0.25) * 1
                label = keras.utils.to_categorical(label, num_classes=2)
                batch_data[idx] = data_patch
                batch_verifier[idx] = label
            # batch_data = np.rollaxis(batch_data, 3, 1)

            yield (batch_data, batch_verifier)

    def input_dim(self):
        return self.window_size, self.window_size, 3
