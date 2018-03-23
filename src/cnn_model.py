from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Dropout, Flatten
import matplotlib.image as img
import cv2
from utility import *
class CNN_keras:

    def __init__(self,network):
        self.net=network
	


    def preprocess_data(self):

        self.images_filtered=[]

        for image in self.images:

            filtered_image=generate_patches_with_pad(image, 16, 16, 0)
            self.images_filtered.append(filtered_image)

    def load_data(self):

        """ Load data into array"""
        self.images=[]
        image= img.imread("../assets/test/test_img.jpg")	
        print ("Original images are shaped ",image.shape)
        self.images.append(image)
        """for i=0;i<=6;i++:
            image= img.imread("../assets/test/test_img"+i".jpg")
            self.images.append(image)"""

    def model_setup(self):

        num_classes=2
        input_data=self.images_filtered[0][0]
        input_shape=input_data.shape

        self.filters=[]
        self.filters[0]=[16,16,3]
        self.filters[1]=[20,20,3]


        self.model= Sequential()
        #Stacking first convolutional layer"""
        #self.model.add(convolution2D(64, self.filters[0], border_mode='same', input_shape=input_shape))
        #self.model.add(LeakyReLU(alpha=0.1))
        input_shape = Input(shape=(rows, cols, 1))
        #filter of different sizes to have higher final accuracy in classification
        #kernel_size=[]
        #kernel_size.add([16, 16, 3])
        #kernel_size.add([16,10,3])
        #kernel_size.add([16,20,3])

        nb_filters_convl_1=30
        nb_filters_convl_2=40
        nb_filters_convl_3=20

        max_pooling_window=[1,8,1]
        #keras.layers.Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
        channel_1 = Conv3D(nb_filters_convl_1, kernel_size[0], padding='same', activation='relu', strides=1)(input_shape) 
        #z1_1=[[(800-kernel_size(0,0)]/stride)+1,([800-kernel_size(0,1)]/stride)+1,([3-kernel_size(0,2)]/stride)+1]
        channel_1 = MaxPooling3D(max_pooling_window, strides=(1, 1), padding='same')(channel_1)

        max_pooling_window2=max_pooling_window
        max_pooling_window2[1]=max_pooling_window+kernel_size[0][1]-kernel_size[1][1]
        channel_2 = Conv3D(nb_filters_convl_1, kernel_size[1], padding='same', activation='relu', strides=1)(input_shape)
        channel_2 = MaxPooling3D((1, max_pooling_window2), strides=(1, 1), padding='same')(channel_2)

        max_pooling_window3=max_pooling_window
        max_pooling_window3[1]=max_pooling_window[0][1]+kernel_size[0][1]-kernel_size[2][1]
        channel_3 = Conv3D(nb_filters_convl_1, kernel_size[2], padding='same', activation='relu',strides=1)(input_shape)
        channel_3 = MaxPooling3D((1, max_pooling_window3), strides=(1, 1), padding='same')(channel_3)

        #the three differently filtered inputs concatenated
        merged = keras.layers.concatenate([channel_1, channel_2, channel_3], axis=1)
        print("merged shape ",merged.shape)

        self.model.add(merged)
        self.model.add(Dropout(0.25))
        self.model.add(LeakyReLU(alpha=0.1))

        #z1_dimensions=?
        self.model.add(Convolution3D(nb_filters_convl_2,kernel_size[1],border_mode='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=pool_size, border_mode='same'))
        self.model.add(Dropout(0.25))

        #vectorize all the previous multidimensional output to input everythong nto a dense final layer
        self.model.add(Flatten())

        out = Dense(input_to_be_determined, activation='relu')
        out = Dense(num_classes, activation='softmax')(out)

        model = Model(input_shape, out)
        plot_model(model, to_file=img_path)
        return model



model=CNN_keras("CNN")
model.load_data()
#print(model.images)
model.preprocess_data()
print(model.images_filtered[0][0].shape)
model.model_setup()



"""from now on just copy pasted examples"""

"""#variable initialization 
nb_filters =100
kernel_size= {}
kernel_size[0]= [3,3]
kernel_size[1]= [4,4]
kernel_size[2]= [5,5]
input_shape=(32, 32, 3)
pool_size = (2,2)
nb_classes =2
no_parallel_filters = 3

# create seperate model graph for parallel processing with different filter sizes
# apply 'same' padding so that ll produce o/p tensor of same size for concatination
# cancat all paralle output

inp = Input(shape=input_shape)
convs = []
for k_no in range(len(kernel_size)):
    conv = Convolution2D(nb_filters, kernel_size[k_no][0], kernel_size[k_no][1],
                    border_mode='same',
                         activation='relu',
                    input_shape=input_shape)(inp)
    pool = MaxPooling2D(pool_size=pool_size)(conv)
    convs.append(pool)

if len(kernel_size) > 1:
    out = Merge(mode='concat')(convs)
else:
    out = convs[0]

conv_model = Model(input=inp, output=out)

# add created model grapg in sequential model

model = Sequential()
model.add(conv_model)        # add model just like layer
model.add(Convolution2D(nb_filters, kernel_size[1][0], kernel_size[1][0]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
model.add(Flatten(input_shape=input_shape))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('tanh'))
"""

"""rows, cols = 100, 15
def create_convnet(img_path='network_image.png'):
    input_shape = Input(shape=(rows, cols, 1))

    tower_1 = Conv2D(20, (100, 5), padding='same', activation='relu')(input_shape)
    tower_1 = MaxPooling2D((1, 11), strides=(1, 1), padding='same')(tower_1)

    tower_2 = Conv2D(20, (100, 7), padding='same', activation='relu')(input_shape)
    tower_2 = MaxPooling2D((1, 9), strides=(1, 1), padding='same')(tower_2)

    tower_3 = Conv2D(20, (100, 10), padding='same', activation='relu')(input_shape)
    tower_3 = MaxPooling2D((1, 6), strides=(1, 1), padding='same')(tower_3)

    merged = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
    merged = Flatten()(merged)

    out = Dense(200, activation='relu')(merged)
    out = Dense(num_classes, activation='softmax')(out)

    model = Model(input_shape, out)
    plot_model(model, to_file=img_path)
    return model"""