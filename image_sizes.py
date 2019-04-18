import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import os

train_img_path = 'petfinder-adoption-prediction\\train_images'

# This is to decide what size we should rescale the images to
# heights = []
# widths = []
# print(len(os.listdir(train_img_path))) #58311 training images
# for filename in sorted(os.listdir(train_img_path)):
# 	img = Image.open(train_img_path+'\\'+filename)
# 	w, h = img.size
# 	widths.append(w)
# 	heights.append(h)
# print(widths)
# print(heights)
# print(np.mean(widths)) # 400
# print(np.mean(heights)) # 390

img_size = 100 # for now, let's just make it a square
def tf_resize_images(X_img_file_paths):
    X_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, None, 3))
    tf_img = tf.image.resize_images(X, (img_size, img_size), 
                                    tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Each image is resized individually as different image may be of different size.
        for index, file_name in enumerate(sorted(os.listdir(X_img_file_paths))):
            try: 
            	img = mpimg.imread(X_img_file_paths+'\\'+file_name)[:, :, :3] # Do not read alpha channel.
            	resized_img = sess.run(tf_img, feed_dict = {X: img})
            	# X_data.append(resized_img)
            	nparray = np.array(resized_img, dtype=np.float32)/255
            	np.save('petfinder-adoption-prediction\\train_images100_npy\\'+file_name[:-4]+'.npy', nparray)
            except: 
            	print(file_name) #these images are black and white - throw away
    # X_data = np.array(X_data, dtype = np.float32) # Convert to numpy
    # return X_data

tf_resize_images(train_img_path)
# resized_imgs = tf_resize_images(train_img_path)
# print(resized_imgs.shape)
# np.save('train_img_resize.npy', resized_imgs)
