import time
import os

from medpy.io import load, save
import numpy as np

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d

import tensorflow as tf

# Network Parameters
tf.reset_default_graph()
width = 256
height = 256
n_channels = 1
n_classes = 2 # total classes (brain, non-brain)
x = tf.placeholder(tf.float32, [None, width, height, n_channels])

################Create Model######################
conv1 = conv_2d(x, 32, 3, activation='relu', padding='same', regularizer="L2")
conv1 = conv_2d(conv1, 32, 3, activation='relu', padding='same', regularizer="L2")
pool1 = max_pool_2d(conv1, 2)

conv2 = conv_2d(pool1, 64, 3, activation='relu', padding='same', regularizer="L2")
conv2 = conv_2d(conv2, 64, 3, activation='relu', padding='same', regularizer="L2")
pool2 = max_pool_2d(conv2, 2)

conv3 = conv_2d(pool2, 128, 3, activation='relu', padding='same', regularizer="L2")
conv3 = conv_2d(conv3, 128, 3, activation='relu', padding='same', regularizer="L2")
pool3 = max_pool_2d(conv3, 2)

conv4 = conv_2d(pool3, 256, 3, activation='relu', padding='same', regularizer="L2")
conv4 = conv_2d(conv4, 256, 3, activation='relu', padding='same', regularizer="L2")
pool4 = max_pool_2d(conv4, 2)

conv5 = conv_2d(pool4, 512, 3, activation='relu', padding='same', regularizer="L2")
conv5 = conv_2d(conv5, 512, 3, activation='relu', padding='same', regularizer="L2")

up6 = upsample_2d(conv5,2)
up6 = tflearn.layers.merge_ops.merge([up6, conv4], 'concat',axis=3)
conv6 = conv_2d(up6, 256, 3, activation='relu', padding='same', regularizer="L2")
conv6 = conv_2d(conv6, 256, 3, activation='relu', padding='same', regularizer="L2")

up7 = upsample_2d(conv6,2)
up7 = tflearn.layers.merge_ops.merge([up7, conv3],'concat', axis=3)
conv7 = conv_2d(up7, 128, 3, activation='relu', padding='same', regularizer="L2")
conv7 = conv_2d(conv7, 128, 3, activation='relu', padding='same', regularizer="L2")

up8 = upsample_2d(conv7,2)
up8 = tflearn.layers.merge_ops.merge([up8, conv2],'concat', axis=3)
conv8 = conv_2d(up8, 64, 3, activation='relu', padding='same', regularizer="L2")
conv8 = conv_2d(conv8, 64, 3, activation='relu', padding='same', regularizer="L2")

up9 = upsample_2d(conv8,2)
up9 = tflearn.layers.merge_ops.merge([up9, conv1],'concat', axis=3)
conv9 = conv_2d(up9, 32, 3, activation='relu', padding='same', regularizer="L2")
conv9 = conv_2d(conv9, 32, 3, activation='relu', padding='same', regularizer="L2")

pred = conv_2d(conv9, 2, 1,  activation='linear', padding='valid')

pred_reshape = tf.reshape(pred, [-1, n_classes])

###############Initialize Model#######################
init = tf.initialize_all_variables()
    
sess = tf.Session()
sess.run(init)

saver = tf.train.Saver()
#Load model
model_path = "../Model/model_epoch830.ckpt"
saver.restore(sess, model_path)

##############Evaluate Model on data#############
inputCounter = 0
for f in os.walk('../InputData'):
    inputCounter += 1
    if inputCounter > 1: 
        print(f[0])
        for f2 in os.listdir(f[0]): 
            if not("mask") in f2 and "nii" in f2 and "SE" in f[0]:
                t = time.time()
                image_data, image_header = load(f[0]+'/'+f2) # Load data
                imageDim = np.shape(image_data)   
                image_data = np.moveaxis(image_data, -1, 0) # Bring the last dim to the first
                input_data = image_data[..., np.newaxis] # Add one axis to the end

                out = sess.run(tf.nn.softmax(pred_reshape), feed_dict={x: input_data}) # Find probabilities
                _out = np.reshape(out, (imageDim[2], imageDim[0], imageDim[1], n_classes)) # Reshape to input shape
                
                mask = 1 - np.argmax(np.asarray(_out), axis=3).astype(float) # Find mask
                mask = np.moveaxis(mask, 0, -1) # Bring the first dim to the last

                elapsed = time.time() - t 
                save(mask, f[0]+'/ft_mask'+f2, image_header) # Save the mask
                print(elapsed)
