import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def drop_channels(image, R=True, G=True, B=True):
    drop_R = 0
    drop_G = 1
    drop_B = 2

    if (not R) & G & B:

        image[:, :, drop_R] = 0

    elif R & (not G) & B:

        image[:, :, drop_G] = 0

    elif R & G & (not B):

        image[:, :, drop_B] = 0

    elif not R & (not G) & B:

        image[:, :, drop_R] = 0
        image[:, :, drop_G] = 0

    elif (not R) & G & (not B):

        image[:, :, drop_R] = 0
        image[:, :, drop_B] = 0

    elif R & (not G) & (not B):

        image[:, :, drop_G] = 0
        image[:, :, drop_B] = 0

    elif (not R) & (not G) & B:

        image[:, :, drop_R] = 0
        image[:, :, drop_G] = 0

    return image


def image_filtering(image, filtering_kernel):
    """input : array_like
       Input array to filter.
       weights : array_like
       Array of weights, same number of dimensions as input
    """
    filtered_image = ndimage.convolve(image, filtering_kernel)

    return filtered_image


def filter_and_normalize(image, kernel):
    image = image_filtering(image, filtering_kernel=kernel)
    image = np.absolute(image)

    normalizer = np.amax(image)
    image[:, :, 0] = image[:, :, 0] / normalizer
    image[:, :, 1] = image[:, :, 1] / normalizer
    image[:, :, 2] = image[:, :, 2] / normalizer
    return image


def augment_constrast(image_path):
    img = cv2.imread(image_path, 1)
    cv2.imshow("Original image", img)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2, a, b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    cv2.imshow(image_path, img2)

    return img2


def main():
    # A very simple and very narrow highpass filter
    kernel = np.array([[[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]],
                       [[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]],
                       [[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]]])

    # 400x400x3 training images size
    image_training = mpimg.imread('../../assets/training/data/satImage_003.png')
    # 608x608x3
    image_testing = mpimg.imread('../../assets/tests/data/test_50.png')
    augment_constrast('../../assets/tests/data/test_50.png')
    augment_constrast('../../assets/training/data/satImage_003.png')

    print(kernel)
    print(kernel.shape)
    print(image_training.shape)

    plt.imshow(image_training)
    plt.show()

    image_training = filter_and_normalize(image=image_training, kernel=kernel)

    image_testing = filter_and_normalize(image=image_testing, kernel=kernel)

    plt.imshow(image_training)
    plt.show()
    plt.imshow(image_testing)
    plt.show()

    print("Done")


if __name__ == "__main__":
    main()
