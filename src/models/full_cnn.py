import os
import time
import logging

import matplotlib.pyplot as plt
from keras import Input, Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, concatenate
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

from models.base_model import BaseModel
from utils.commons import properties

HEIGHT = 400
WIDTH = 400
CHANNELS = 3

logger = logging.getLogger("cil_project.src.models.full_cnn")

file_path = os.path.dirname(os.path.abspath(__file__))


class FullCNN(BaseModel):
    def __init__(self, train_generator, validation_generator, path=None):
        super().__init__(train_generator, validation_generator)

        checkpoint_loc = os.path.join(properties["OUTPUT_DIR"], 'weights.h5')

        # earlyStopping = EarlyStopping(monitor='val_loss',
        #                               patience=5,
        #                               verbose=1,
        #                               min_delta=0.0001,
        #                               mode='min')

        modelCheckpoint = ModelCheckpoint(checkpoint_loc,
                                          monitor='val_loss',
                                          verbose=1,
                                          save_weights_only=False,
                                          save_best_only=True)

        self.callbacks_list = [modelCheckpoint]

        self.model = self.get_unet_400()
        self.model.compile(loss=self.bce_dice_loss, optimizer=Adam(lr=1e-4), metrics=[self.dice_coef])

        if path:
            logger.info("Loading weights from {}".format(path))
            self.load(path)
            logger.info("Finished loading weights")
            return

    def train(self, verbosity=None, epochs=150, steps=1000, print_at_end=True):
        self.model.fit_generator(
            self.train_generator.generate_patch(batch_size=8),
            steps_per_epoch=steps,
            epochs=epochs,
            callbacks=self.callbacks_list,
            verbose=1,
            validation_data=self.validation_generator.generate_patch(batch_size=8),
            validation_steps=5)

    def predict(self, test_generator):
        x_batch = next(test_generator.generate_patch_sequential())
        y_batch = self.model.predict(x_batch, batch_size=2)

        img = y_batch[0, :, :].reshape(400, 400)

        plt.imshow(img)
        plt.show()
        print()

    def load(self, filename):
        self.model.load_weights(filename)

    def dice_coef(self, y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_coef_loss(self, y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

    def bce_dice_loss(self, y_true, y_pred):
        return binary_crossentropy(y_true, y_pred) + self.dice_coef_loss(y_true, y_pred)

    def down(self, filters, input_):
        down_ = Conv2D(filters, (3, 3), padding='same')(input_)
        down_ = BatchNormalization(epsilon=1e-4)(down_)
        down_ = Activation('relu')(down_)
        down_ = Conv2D(filters, (3, 3), padding='same')(down_)
        down_ = BatchNormalization(epsilon=1e-4)(down_)
        down_res = Activation('relu')(down_)
        down_pool = MaxPooling2D((2, 2), strides=(2, 2))(down_)
        return down_pool, down_res

    def up(self, filters, input_, down_):
        up_ = UpSampling2D((2, 2))(input_)
        up_ = concatenate([down_, up_], axis=3)
        up_ = Conv2D(filters, (3, 3), padding='same')(up_)
        up_ = BatchNormalization(epsilon=1e-4)(up_)
        up_ = Activation('relu')(up_)
        up_ = Conv2D(filters, (3, 3), padding='same')(up_)
        up_ = BatchNormalization(epsilon=1e-4)(up_)
        up_ = Activation('relu')(up_)
        up_ = Conv2D(filters, (3, 3), padding='same')(up_)
        up_ = BatchNormalization(epsilon=1e-4)(up_)
        up_ = Activation('relu')(up_)
        return up_

    def get_unet_400(self, input_shape=(HEIGHT, WIDTH, CHANNELS), num_classes=1):
        inputs = Input(shape=input_shape)

        down1, down1_res = self.down(25, inputs)
        down2, down2_res = self.down(50, down1)
        down3, down3_res = self.down(100, down2)
        down4, down4_res = self.down(200, down3)

        center = Conv2D(200, (3, 3), padding='same')(down4)
        center = BatchNormalization(epsilon=1e-4)(center)
        center = Activation('relu')(center)

        center = Conv2D(200, (3, 3), padding='same')(center)
        center = BatchNormalization(epsilon=1e-4)(center)
        center = Activation('relu')(center)

        up4 = self.up(200, center, down4_res)
        up3 = self.up(100, up4, down3_res)
        up2 = self.up(50, up3, down2_res)
        up1 = self.up(25, up2, down1_res)

        classify = Conv2D(num_classes, (1, 1), activation='sigmoid', name='final_layer')(up1)

        model = Model(inputs=inputs, outputs=classify)

        return model

    def save(self, path):
        """Save the model of the trained model.

        Args:
            path (path): path for the model file.
        """
        self.model.save(path)
        logger.info("Model saved to {}".format(path))
