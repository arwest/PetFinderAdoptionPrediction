import tensorflow as tf
from keras.layers import Input, Dense, MaxPooling2D, Flatten, Conv2D, Dropout, Reshape, UpSampling2D, Conv2DTranspose, BatchNormalization
from keras.models import Model, Sequential
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from os.path import join

all_data = np.load(join('petfinder-adoption-prediction','all_gray_128.npy'))

#csv file has 14994 rows - break up train/test based on that
num_train = 48926
num_test = 58311 - 48926
X_train = all_data[0:num_train,:,:]
X_train = X_train.reshape(X_train.shape[0], 128, 128, 1).astype('float32')
X_test = all_data[num_train:,:,:]
X_test = X_test.reshape(X_test.shape[0], 128, 128, 1).astype('float32')
num_epochs = 3

model = Sequential()
model.add(Conv2D(32, (4,4), activation='relu', input_shape=(128,128,1)))
BatchNormalization()
model.add(MaxPooling2D((2,2), name='max_pool'))

model.add(Conv2D(64, (4,4)))
BatchNormalization()
model.add(MaxPooling2D((2,2)))


model.add(Conv2D(32, (4,4)))
BatchNormalization()
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(16, (3,3)))
BatchNormalization()
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_regularizer='l2'))
model.add(Dense(64, activation='relu', kernel_regularizer='l2', name='latent'))
model.add(Dense(256, activation='relu', kernel_regularizer='l2'))
model.add(Reshape((8,8,4)))

model.add(Conv2DTranspose(16, (5,5), activation='relu'))
model.add(UpSampling2D((2,2), interpolation='bilinear'))
model.add(Conv2DTranspose(32, (6,6), activation='relu'))
model.add(UpSampling2D((2,2), interpolation='bilinear'))
model.add(Conv2DTranspose(32, (3,3), activation='relu'))
model.add(UpSampling2D((2,2), interpolation='bilinear'))
model.add(Conv2DTranspose(1, (9,9), activation='sigmoid'))

model.summary()
encoder = Model(inputs=model.input, outputs=model.get_layer('latent').output)
encoder.summary()

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, X_train, batch_size=200, epochs=num_epochs, validation_split=.1)

test_reconstruct = model.predict(X_test)
np.save('reconstructed_gray.npy', test_reconstruct)

# all_pooled = encoder.predict(all_data)
# np.save('gray_latent.npy', all_pooled)
#need to either average or max or norm this for each data point then