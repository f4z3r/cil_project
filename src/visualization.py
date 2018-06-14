import csv
import glob
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from utils.commons import *


def show_image(image):
    plt.imshow(image)
    plt.show()


def filter_id_entries(id_img, csv_file):
    x_entries_roads = []
    y_entries_roads = []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:

            if int(id_img) == int(row['Id'][0:row['Id'].index("_")]):
                if int(row["Prediction"]) == 1:
                    id_y_x = row["Id"].split("_")
                    # print(id_y_x)
                    y_entries_roads.append(int(id_y_x[1]))
                    x_entries_roads.append(int(id_y_x[2]))

    if len(x_entries_roads) == 0:
        print("Sorry the provided id of the image has not been found in the csv")
        sys.exit(1)

    return x_entries_roads, y_entries_roads


def get_image_ids(data_files):
    images_ids = []
    for image_file in data_files:
        image_file = image_file[image_file.index("/"):-1]
        start_index = image_file.index("_") + 1
        end_index = image_file.index(".")

        images_ids.append(image_file[start_index:end_index])
    return images_ids


def visualize(id_img, csv_file, path_to_images, patch_size):
    x, y = filter_id_entries(id_img=int(id_img), csv_file=csv_file)

    images = glob.glob(os.path.join(path_to_images, "*.png"))
    images_ids = get_image_ids(images)

    img = mpimg.imread(images[images_ids.index(id_img)])
    img1 = mpimg.imread(images[images_ids.index(id_img)])

    total_entries = len(x)

    print("[INFO] Total predicted road patches: ", total_entries)
    for idx in range(total_entries):
        img[x[idx]:x[idx] + patch_size, y[idx]:y[idx] + patch_size, 0] = 0
    show_image(img)

# def compare_csvs(csv_file1, csv_file2):
#    x1, y1 = filter_id_entries(id_img, csv_file1)
