import glob
import os

import keras
import matplotlib.image as mpimg
import numpy as np
import sys, traceback

class PatchTestImageGenerator:
    def __init__(self, path_to_images, save_predictions_path, pad = 28, patch_size = 16, context_padding = 28, four_dim = False):
        data_files = glob.glob(os.path.join(path_to_images, "*.png"))
        self.extract_image_ids(data_files=data_files)
        image_count = len(data_files)
        first = mpimg.imread(data_files[0])
        """Define the 72x72x3 single patch -> 16x16 patch with context"""
        data_set = np.empty((image_count,
                             first.shape[0] + 2 * pad,
                             first.shape[1] + 2 * pad,
                             first.shape[2]))

        for idx, file in enumerate(data_files):
            data_set[idx] = np.pad(mpimg.imread(file), ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        self.path_to_images = path_to_images
        #TODO ubcomment the following line for short testing purposes
        #self.data_set = data_set[0:10]
        self.data_set = data_set
        print("TOTAL TESTING SET ",data_set.shape)
        self.four_dim = four_dim
        self.window_size = patch_size + 2 * context_padding
        self.patch_size = patch_size
        self.context_padding = context_padding

        print('PatchImageGenerator initialized with {} pictures'.format(image_count))

    def extract_image_ids(self, data_files):

        images_ids = []
        for image_file in data_files:
            image_file = image_file[image_file.index("/"):-1]
            start_index = image_file.index("_")+1
            end_index = image_file.index(".")

            images_ids.append(image_file[start_index:end_index])

        self.images_ids = images_ids




    def check_dimensions_patch_with_img(self, w_img, h_img, padding):
        patch_size = self.patch_size
        check_result = (w_img-2*padding)/patch_size == 0 and (h_img-2*padding)/patch_size
        return check_result


    def get_test_patches_from_image(self, data_img):
        window_size = self.window_size
        patch_size = self.patch_size
        context_padding = self.context_padding

        dimensions = list(data_img.shape)
        width_image = dimensions[0] 
        height_image = dimensions[1]

        """check_result = self.check_dimensions_patch_with_img(width_image, height_image, context_padding)
        
        if not check_result:
            print("The width of the image ",width_image," and the height of the image", height_image,
                   " are incompatible with patch size ", patch_size)
            traceback.print_exc(file=sys.stdout)
            sys.exit(0)"""
        print("Parameters' checks:")
        print("Width padded image ",width_image)
        print("Height padded image ",height_image)
        patches_over_width = int((width_image-2*context_padding)/patch_size)
        print("Patches over width ",patches_over_width)
        patches_over_height = int((height_image-2*context_padding)/patch_size)
        print("Patches over height ",patches_over_height)
        total_patches = patches_over_height*patches_over_width
        print("Patch size ",patch_size)
        
        """Test patches:
            from the left to the right w.r.t the image width,
            from the bottom to top w.r.t the image height (bottom is supposed to be the top of the image)
        """
        img_patches = []

        for patch_h_idx in range(patches_over_height):
            
            total_height_patch_context = patch_size + context_padding*2
            for patch_w_idx in range(patches_over_width):

                total_width_patch_context = patch_size + context_padding*2

                start_w = patch_w_idx * patch_size
                end_w = patch_w_idx * patch_size + patch_size + context_padding*2
                start_h = patch_h_idx * patch_size
                end_h = patch_h_idx * patch_size + patch_size + context_padding*2

                img_patches.append(data_img[start_w:end_w,start_h:end_h,:])
        print("Total patches: ",total_patches)
        return total_patches, np.asarray(img_patches)


    def generate_test_patches(self, batch_size=100):
        window_size = self.window_size
        patch_size = self.patch_size
        context_padding = self.context_padding
        while True:
            img_number = 1
            for img in self.data_set:
                total_patches, img_patches = self.get_test_patches_from_image(img)

                if self.four_dim:
                    patch_idx = 0;
                    for patch in img_patches:
                        img_patches[patch_idx] = patch.reshape(total_patches, self.window_size, self.window_size, 3, 1)
                        patch_idx = patch_idx + 1
                print("Patches of the image number ",img_number," created..")
                img_number = img_number+1

                yield img_patches
            break
    def input_dim(self):
        if self.four_dim:
            return self.window_size, self.window_size, 3, 1
        return self.window_size, self.window_size, 3