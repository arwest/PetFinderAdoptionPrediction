import tensorflow as tf
from keras.layers import Input, Dense, MaxPooling2D, Flatten, Conv2D, Dropout
from keras.models import Sequential
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from os.path import join

# X_data_dir = join('petfinder-adoption-prediction','train_images_npy')
# all_data = np.zeros((58283,50,50,3)) # we have 58283 images
# file_names = []
# for i, file_name in enumerate(sorted(os.listdir(X_data_dir))):
# 	file_names.append(file_name) #= file_name#.split('-')[0]
# file_names.sort()
# print(file_names[0:20])
# for i, file_name in enumerate(file_names):
# 	if i % 5000 == 0: print(i)
# 	array = np.load(join(X_data_dir,file_name))
# 	all_data[i,:,:,:] = array
# np.save(join('petfinder-adoption-prediction','all_images.npy'), all_data)

all_data = np.load(join('petfinder-adoption-prediction','all_images.npy'))
all_labels = np.load(join('petfinder-adoption-prediction','train_images_results','matrix.npy'))[:,1]
all_targets = np.zeros((len(all_labels), 5)) # len = 58283
for i in range(len(all_labels)):
	idx = int(all_labels[i])
	all_targets[i, idx] = 1
#csv file has 14994 rows - break up train/test based on that
num_train = 50000
num_test = 8283
X_train = all_data[0:num_train,:,:,:]
y_train = all_targets[0:num_train,:]
X_test = all_data[num_train:,:,:,:]
y_test = all_targets[num_train:,:]
num_epochs = 5

# Build CNN 
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(50,50,3)))
# model.add(Conv2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(.2))

# model.add(Conv2D(64,(3,3), activation='relu'))
# # model.add(Conv2D(32,(3,3), activation='relu'))
# model.add(MaxPooling2D((2,2)))
# model.add(Dropout(.2))

model.add(Conv2D(32,(3,3), activation='relu'))
# model.add(Conv2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(.2))

model.add(Conv2D(32,(3,3), activation='relu'))
# model.add(Conv2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(.2))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer='l2'))
model.add(Dropout(.2))
model.add(Dense(64, activation='relu', kernel_regularizer='l2'))
model.add(Dropout(.2))
# model.add(Dense(32, activation='relu', kernel_regularizer='l2'))
model.add(Dense(5, activation='softmax')) # five categories

model.compile(optimizer='adam', loss='categorical_crossentropy',
	metrics=['accuracy'])
model.summary()

# Train
model.fit(X_train, y_train, epochs=num_epochs, validation_split=.1)

# Not a null model, but basically just gives data likelihood I think

# Test
score = model.evaluate(X_test, y_test)
print('Loss, Accuracy = ', score)
predictions = model.predict(X_test)
print(predictions)
# temp = np.zeros(8283)
# #fix this
# for i in range(8283):
# 	temp[i] = np.argmax(predictions[i,:])
# print(temp)
# print("number misclassifications = ", np.linalg.norm(temp-y_test[:,0], 1))

