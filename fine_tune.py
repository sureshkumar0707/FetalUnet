# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 14:09:10 2018

@author: zfq
"""
from datetime import datetime
import os

from medpy.io import load, save
import numpy as np

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d

import tensorflow as tf

tf.reset_default_graph()

# Network Parameters
checkpoint_path = "../Model"
batch_size = 8
learning_rate = 0.005
display_step = 10
num_epochs = 1000
width = 256
height = 256
n_channels = 1
n_classes = 2 # total classes (brain, non-brain)

x = tf.placeholder(tf.float32, [None, width, height, n_channels])
y = tf.placeholder(tf.float32, [None, n_classes])

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


#########configure train process#######
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred_reshape, labels = y))

fine_tune_var_list = [i for i in tf.trainable_variables()]
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, fine_tune_var_list)
    gradients = list(zip(gradients, fine_tune_var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in fine_tune_var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(pred_reshape, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter("./tensorboard_log")

saver = tf.train.Saver()

image = []
label = []
index = 0
for f in os.walk('../InputData'):
    for f2 in os.listdir(f[0]): 
        if not("Net_mask") in f2 and not("gt_mask") in f2 and "nii" in f2 and "SE" in f[0]:
            print(index)            
            print(f2)
#            if index == 12:
#                break
           
            image_data, _ = load(f[0]+'/'+f2) # Load data
            image_data = image_data.astype(np.float)
            image_data = np.moveaxis(image_data, -1, 0) # Bring the last dim to the first    
            print(image_data.shape)
          
            label_data, _ = load(f[0] + '/' + 'gt_mask' + f[0][36:] + '.nii')
            label_data = label_data.astype(np.float)
            label_data = np.moveaxis(label_data, -1, 0) # Bring the last dim to the first
            print(label_data.shape)

            if index == 0:
                image = image_data
                label = label_data
            else:
                image = np.concatenate((image, image_data), axis=0)
                label = np.concatenate((label, label_data), axis=0)
            index += 1                

train_dataset = tf.data.Dataset.from_tensor_slices((image[0:196,:,:], label[0:196,:,:]))
train_dataset = train_dataset.repeat()
train_batched_dataset = train_dataset.batch(1)

train_iterator = train_batched_dataset.make_one_shot_iterator()
train_next_element = train_iterator.get_next()

val_dataset = tf.data.Dataset.from_tensor_slices((image[196:,:,:], label[196:,:,:]))
val_dataset = val_dataset.repeat()
val_batched_dataset = val_dataset.batch(1)

val_iterator = val_batched_dataset.make_one_shot_iterator()
val_next_element = val_iterator.get_next()

with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())
   
    # Add the model graph to TensorBoard
    #writer.add_graph(sess.graph)

    #Load model
    model_path = "../Model/Fetal_2D_Ref_6980_norm0.ckpt"
    saver.restore(sess, model_path)
    
    print("{} Start training...".format(datetime.now()))
    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        step = 1
        while step < 49:
            d, l = sess.run(train_next_element)
            d = d[..., np.newaxis] 
            l = l.flatten()
            l2 = 1 - l
            l = l[:,np.newaxis]
            l2 = l2[:,np.newaxis]
            l = np.concatenate((l2,l), axis=1)

            sess.run(train_op, feed_dict={x: d,
                                          y: l})

            # Generate summary with the current batch of data and write to file
#            if step % display_step == 0:
#                s = sess.run(merged_summary, feed_dict={x: d,
#                                                        y: l})
#                writer.add_summary(s, epoch * 49 + step)

            step += 1
            
        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        test_acc = 0.
        test_count = 0
        for _ in range(86):
            d, l = sess.run(val_next_element)
            d = d[..., np.newaxis] 
            l = l.flatten()
            l2 = 1 - l
            l = l[:,np.newaxis]
            l2 = l2[:,np.newaxis]
            l = np.concatenate((l2,l), axis=1)
            
            acc = sess.run(accuracy, feed_dict={x: d,
                                                y: l})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("Validation Accuracy = {:.4f}".format(test_acc))
   
        if epoch % 10 == 0:
            print("{} Saving checkpoint of model...".format(datetime.now()))
    
            #save checkpoint of the model
            checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch)+'.ckpt')
            save_path = saver.save(sess, checkpoint_name)
    
            print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))

#
##%%
#sess = tf.Session()
#a = np.array([[1,2],[3,4],[5,6]])
#b = np.array([[5,2],[8,4],[9,6]])
#dataset = tf.data.Dataset.from_tensor_slices((a, b))
#dataset = dataset.repeat()
#batched_dataset = dataset.batch(1)
#
#iterator = batched_dataset.make_one_shot_iterator()
#next_element = iterator.get_next()
#
#a = sess.run(next_element)
#
##%%
#image_data, image_header = load("../InputData/skullStrippingImages/SE4/BH_Loc_T2SSFSE.nii") # Load data  
#image_data = np.moveaxis(image_data, -1, 0) # Bring the last dim to the first
#input_data = image_data[..., np.newaxis] # Add one axis to the end
#a = input_data
#

