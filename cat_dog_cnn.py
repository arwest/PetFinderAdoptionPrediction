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

X_data_dir = 'petfinder-adoption-prediction\\train_images_npy\\'
# all_data = np.zeros((58283,50,50,3)) # we have 58283 images
# file_names = []
# for i, file_name in enumerate(sorted(os.listdir(X_data_dir))):
# 	file_names.append(file_name) #= file_name#.split('-')[0]
# file_names.sort()
# print(file_names[0:20])
# for i, file_name in enumerate(file_names):
# 	if i % 5000 == 0: print(i)
# 	array = np.load(X_data_dir+file_name)
# 	all_data[i,:,:,:] = array
# np.save('all_images.npy', all_data)

all_data = np.load('petfinder-adoption-prediction\\all_images.npy')
all_labels = np.load('petfinder-adoption-prediction\\train_images_results\\matrix.npy')[:,0] # check filename and index
print(all_labels[0:20])
all_labels = (all_labels * -1) + 2 # dogs are 1, cats are 0
all_targets = np.zeros((58283,2))
all_targets[:,0] = all_labels # dog indicator
all_targets[:,1] = (-1*all_labels) + 1 # cat indicator
#csv file has 14994 rows - break up train/test based on that
num_train = 50000
num_test = 8283
X_train = all_data[0:num_train,:,:,:]
y_train = all_targets[0:num_train,:]
X_test = all_data[num_train:,:,:,:]
y_test = all_targets[num_train:,:]
num_epochs = 5

print(y_train[0:20,:])
# print(all_labels[0:20])

# Build CNN 
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(50,50,3)))
# model.add(Conv2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(.2))

model.add(Conv2D(64,(3,3), activation='relu'))
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
model.add(Dense(32, activation='relu', kernel_regularizer='l2'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='binary_crossentropy',
	metrics=['accuracy'])
model.summary()

# Train
model.fit(X_train, y_train, epochs=num_epochs, validation_split=.1)

# Test
score = model.evaluate(X_test, y_test)
print('Loss, Accuracy = ', score)
predictions = model.predict(X_test)
print(predictions)
temp = np.zeros(8283)
for i in range(8283):
	if predictions[i,0] > predictions[i,1]:
		temp[i] = 1
print(temp)
print("number misclassifications = ", np.linalg.norm(temp-y_test[:,0], 1))

# This model is giving 74% test accuracy

