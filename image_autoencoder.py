import tensorflow as tf
from keras.layers import Input, Dense, MaxPooling2D, Flatten, Conv2D, Dropout, Reshape, UpSampling2D, Conv2DTranspose
from keras.models import Model, Sequential
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from os.path import join

all_data = np.load(join('petfinder-adoption-prediction','all_images.npy'))

#csv file has 14994 rows - break up train/test based on that
num_train = 48899
num_test = 58283 - 48899
X_train = all_data[0:num_train,:,:,:]
X_test = all_data[num_train:,:,:,:]
num_epochs = 10

model = Sequential()
model.add(Conv2D(32, (5,5), activation='relu', input_shape=(50,50,3)))
model.add(MaxPooling2D((2,2), name='max_pool'))

# model.add(Conv2D(5, (10,10)))
# model.add(MaxPooling2D((2,2)))

# model.add(Flatten())
# model.add(Dense(75, activation='relu', kernel_regularizer='l2'))
# model.add(Dense(245, activation='relu', kernel_regularizer='l2'))
# model.add(Reshape((7,7,5)))

# model.add(Conv2DTranspose(5, (6,6), activation='relu'))
# model.add(UpSampling2D((2,2), interpolation='bilinear'))
model.add(Conv2DTranspose(32, (5,5), activation='relu'))
model.add(UpSampling2D((2,2), interpolation='bilinear'))
# model.add(Conv2DTranspose(3, (10,10), activation='relu'))
model.add(Conv2D(3, (5,5), activation='sigmoid'))

model.summary()
encoder = Model(inputs=model.input, outputs=model.get_layer('max_pool').output)
encoder.summary()

model.compile(optimizer='adam', loss='mean_absolute_error')

model.fit(X_train, X_train, batch_size=200, epochs=num_epochs, validation_split=.1)

test_reconstruct = model.predict(X_test)
np.save('reconstructed_imgs.npy', test_reconstruct)

all_pooled = encoder.predict(all_data)
np.save('all_pooled32.npy', all_pooled)
#need to either average or max or norm this for each data point then