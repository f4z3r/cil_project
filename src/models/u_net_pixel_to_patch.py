import logging
import os
import re

import keras
from keras import Input, Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, concatenate, Dropout, \
    LeakyReLU
from keras.losses import binary_crossentropy
from keras.optimizers import Adam, RMSprop
from models.base_model import BaseModel
from scipy import ndimage
from utils.commons import properties

HEIGHT = 400
WIDTH = 400
CHANNELS = 3

logger = logging.getLogger("cil_project.src.models.u_net_pixel_to_patch")

file_path = os.path.dirname(os.path.abspath(__file__))


class UNet(BaseModel):
    def __init__(self, train_generator, _, path=None):
        super().__init__(train_generator, _)

        checkpoint_loc = os.path.join(properties["OUTPUT_DIR"], 'weights.h5')

        self.validation_steps = 50
        self.batch_size = 4
        self.model = UNet.get_unet_downsampling(activation="leakyrelu", regularizer=1e-6, dropout_rate=0.25)
        self.model.summary()

        self.callbacks_list = [EarlyStopping(monitor='val_loss',
                                             patience=16,
                                             verbose=1,
                                             min_delta=1e-4),
                               ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.1,
                                                 patience=8,
                                                 verbose=1,
                                                 epsilon=1e-4),
                               ModelCheckpoint(checkpoint_loc,
                                               monitor='val_loss',
                                               save_best_only=True,
                                               save_weights_only=True),
                               TensorBoard(log_dir=properties["OUTPUT_DIR"])]

        if path:
            logger.info("Loading weights from {}".format(path))
            self.load(path)
            logger.info("Finished loading weights")

    def train(self, epochs=100, steps=500, print_at_end=True):
        self.model.fit_generator(generator=self.train_generator.next_batch("train", batch_size=self.batch_size),
                                 steps_per_epoch=steps,
                                 epochs=epochs,
                                 callbacks=self.callbacks_list,
                                 validation_data=self.train_generator.next_batch("valid", batch_size=self.batch_size),
                                 validation_steps=self.validation_steps)

    def predict(self, path_images, submission_path_filename):
        images_files = sorted(os.listdir(os.path.join(path_images, "data")))
        with open(submission_path_filename, 'w') as f:
            f.write('Id,Prediction\n')
            for idx, name in enumerate(images_files):
                image = ndimage.imread(os.path.join(path_images, "data", name)).reshape((1, 608, 608, 3)) / 255.0
                # image = test_images[idx].reshape((1, 608, 608, 3)) / 255.
                predicted_mask = self.model.predict(image)
                predicted_mask = predicted_mask.reshape((38, 38))
                submission_string = self.mask_to_submission_strings(name, predicted_mask)
                f.writelines('{}\n'.format(s) for s in submission_string)

    def load(self, filename):
        self.model.load_weights(filename)

    @staticmethod
    def mask_to_submission_strings(image_filename, predicted_mask):
        """Reads a single image and outputs the strings that should go into the submission file"""
        img_number = int(re.search(r"\d+", image_filename).group(0))
        for j in range(38):
            for i in range(38):
                patch = predicted_mask[i, j]
                label = int(patch > 0.5)
                yield ("{:03d}_{}_{},{}".format(img_number, j * 16, i * 16, label))

    @staticmethod
    def get_unet_layer_down(size, depth, inputs, activation="relu", dropout_rate=0.0):
        down = inputs
        for _ in range(depth):
            down = Conv2D(size, (3, 3), padding='same')(down)
            down = BatchNormalization()(down)
            if activation == "relu":
                down = Activation('relu')(down)
            elif activation == "leakyrelu":
                down = LeakyReLU(alpha=0.1)(down)
        down_pool = MaxPooling2D((2, 2), strides=(2, 2))(down)
        if dropout_rate > 0:
            down_pool = Dropout(rate=dropout_rate)(down_pool)
        return down_pool, down

    @staticmethod
    def get_unet_layer_up(size, depth, inputs, inputs_2, activation="relu", dropout_rate=0.0):
        up = UpSampling2D((2, 2))(inputs)
        up = concatenate([inputs_2, up], axis=3)
        for _ in range(depth):
            up = Conv2D(size, (3, 3), padding='same')(up)
            up = BatchNormalization()(up)
            if activation == "relu":
                up = Activation('relu')(up)
            elif activation == "leakyrelu":
                up = LeakyReLU(alpha=0.1)(up)
            if dropout_rate > 0:
                up = Dropout(rate=dropout_rate)(up)
        return up

    @staticmethod
    def get_unet_downsampling(input_shape=(608, 608, 3), optimizer="Adam", activation="relu", dropout_rate=0.0,
                              regularizer=0.0):
        inputs = Input(shape=input_shape)
        num_classes = 1

        down0a_pool, down0a = UNet.get_unet_layer_down(16, 2, inputs, activation, dropout_rate)
        down0_pool, down0 = UNet.get_unet_layer_down(32, 2, down0a_pool, activation, dropout_rate)
        down1_pool, down1 = UNet.get_unet_layer_down(64, 2, down0_pool, activation, dropout_rate)
        down2_pool, down2 = UNet.get_unet_layer_down(128, 2, down1_pool, activation, dropout_rate)
        down3_pool, down3 = UNet.get_unet_layer_down(256, 2, down2_pool, activation, dropout_rate)
        center_pool, center = UNet.get_unet_layer_down(512, 2, down3_pool, activation, dropout_rate)
        up3 = UNet.get_unet_layer_up(256, 3, center, down3, activation, dropout_rate)

        if regularizer > 0.0:
            # e.g. regularizer = 1e-6
            classify = Conv2D(num_classes, (1, 1), activation='sigmoid',
                              kernel_regularizer=keras.regularizers.l2(regularizer))(
                up3)
        else:
            classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up3)

        model = Model(inputs=inputs, outputs=classify)

        if optimizer == "Adam":
            opt = Adam()
        else:
            opt = RMSprop(lr=0.0001)

        model.compile(optimizer=opt, loss=UNet.bce_dice_loss, metrics=[UNet.dice_coeff])

        return model

    @staticmethod
    def dice_coeff(y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return score

    @staticmethod
    def dice_loss(y_true, y_pred):
        loss = 1 - UNet.dice_coeff(y_true, y_pred)
        return loss

    @staticmethod
    def bce_dice_loss(y_true, y_pred):
        loss = binary_crossentropy(y_true, y_pred) + UNet.dice_loss(y_true, y_pred)
        return loss

    def save(self, path):
        """Save the model of the trained model.

        Args:
            path (path): path for the model file.
        """
        self.model.save(path)
        logger.info("Model saved to {}".format(path))
