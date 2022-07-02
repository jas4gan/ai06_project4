# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 10:24:57 2022

@author: jases

https://www.kaggle.com/competitions/data-science-bowl-2018/overview

https://shrdc-my.sharepoint.com/:u:/g/personal/kong_kah_chun_shrdc_org_my/Eezac_jW3b9Fiazf0q3cDN4BEUIcOyzln9y95V7ER-orMg?e=qj3pZM
"""
#%%
#1. Import packages

import matplotlib.pyplot as plt
import numpy as np
from scipy import io
import glob, os
import tensorflow as tf
from tensorflow import keras
import datetime
from tensorflow_examples.models.pix2pix import pix2pix
import cv2
from IPython.display import clear_output

#2. Load the data
#prepare an empty list for image and mask
train_images=[]
train_masks=[]
test_images=[]
test_masks=[]
X_test=[]
train_file_path=r"C:\Users\jases\Desktop\AI-06\DL\data\data-science-bowl-2018-2\train"
test_file_path=r"C:\Users\jases\Desktop\AI-06\DL\data\data-science-bowl-2018-2\test"

#Load the train images
train_image_dir= os.path.join(train_file_path, "inputs")
for train_image_file in os.listdir(train_image_dir):
    img= cv2.imread(os.path.join(train_image_dir, train_image_file))
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img= cv2.resize(img, (128,128))
    train_images.append(img)
    
#Load the train mask
train_mask_dir= os.path.join(train_file_path, "masks")
for train_mask_file in os.listdir(train_mask_dir):
    mask= cv2.imread(os.path.join(train_mask_dir, train_mask_file), cv2.IMREAD_GRAYSCALE)
    mask= cv2.resize(mask, (128,128))
    train_masks.append(mask)

#load the test images
test_image_dir= os.path.join(test_file_path, "inputs")
for test_image_file in os.listdir(test_image_dir):
    img= cv2.imread(os.path.join(test_image_dir, test_image_file))
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img= cv2.resize(img, (128,128))
    test_images.append(img)
   
#load the test masks
test_mask_dir= os.path.join(test_file_path, "masks")
for test_mask_file in os.listdir(test_mask_dir):
    mask= cv2.imread(os.path.join(test_mask_dir, test_mask_file), cv2.IMREAD_GRAYSCALE)
    mask= cv2.resize(mask, (128,128))
    test_masks.append(mask)
    
#%%
#convert the lists into numpy array
train_images_np= np.array(train_images)
train_masks_np= np.array(train_masks)

#%%
#check some examples
#for images
plt.figure(figsize= (10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    img_plot= train_images[i]
    plt.imshow(img_plot)
    plt.axis('off')
    
plt.show()

#%%
#for masks
plt.figure(figsize= (10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    mask_plot= train_masks[i]
    plt.imshow(mask_plot)
    plt.axis('off')

plt.show()

#%%
#3. Data preprocessing
#expand the mask dimension
train_masks_np_exp= np.expand_dims(train_masks_np, axis=-1)

#check the mask output
print(np.unique(train_masks[0])) # on mask 0

#%%
#change the mask value
train_converted_masks= np.round(train_masks_np_exp/255)
print(train_converted_masks)

#%%
#normalize the images
train_converted_images= train_images_np / 255.0
print(train_converted_images)

#%%
#do train test split
from sklearn.model_selection import train_test_split
SEED=12345
x_train, x_test, y_train, y_test= train_test_split(train_converted_images, train_converted_masks, test_size=0.2, random_state=SEED)

#%%
#convert numpy array into tensor slice
x_train_tensor= tf.data.Dataset.from_tensor_slices(x_train)
x_test_tensor= tf.data.Dataset.from_tensor_slices(x_test)
y_train_tensor= tf.data.Dataset.from_tensor_slices(y_train)
y_test_tensor= tf.data.Dataset.from_tensor_slices(y_test)

#%%
#zip the tensor slice into ZipDataset
train= tf.data.Dataset.zip((x_train_tensor, y_train_tensor))
test= tf.data.Dataset.zip((x_test_tensor, y_test_tensor))

#%%
#convert into PrefetchDataset
BATCH_SIZE= 16
AUTOTUNE= tf.data.AUTOTUNE
BUFFER_SIZE= 1000
STEPS_PER_EPOCH= 800//BATCH_SIZE
VALIDATION_STEPS= 200//BATCH_SIZE

train= train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train= train.prefetch(buffer_size= AUTOTUNE)

test= test.batch(BATCH_SIZE).prefetch(buffer_size= AUTOTUNE)

#%%
#4. Create the model
#use a pretrained as feature extractor
base_model = tf.keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)

#Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
    ]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

#Define the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

#Define the upsampling path
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]


def unet_model(output_channels:int):
    inputs = tf.keras.layers.Input(shape=[128,128,3])
    #Applying functional API
    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    
    #Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x,skip])
    
    #This is the last layer of the model (output layer)
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2, padding='same') #64x64 --> 128x128
    
    x = last(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)

#%%
OUTPUT_CLASSES = 3

model = unet_model(output_channels=OUTPUT_CLASSES)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])

tf.keras.utils.plot_model(model, show_shapes=True)

#%%
#create functions to show predictions
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]
    return pred_mask[0]

def display(display_list):
    plt.figure(figsize=(15,15))
    title = ['Input Image','True Mask','Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
        
    plt.show()

#%%    
for images, masks in train.take(2):# to see how the image and mask display
    sample_image,sample_mask = images[0], masks[0]
    display([sample_image,sample_mask])
    
def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)])
            
    else:
        display([img,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))])
     
show_predictions()
    
#%%
#Create a callback to help to display results during training
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))

#%%
#hyperparameters for model training
EPOCHS = 20
model_history = model.fit(train, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test,
                          callbacks=[DisplayCallback()])

#%%
#deploy model
show_predictions(test,3)
