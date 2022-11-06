import os
import numpy as np
import random
import pandas as pd 
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from numpy.random import seed 

#Fixing seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(37)
random.seed(1254)
tf.random.set_seed(89)

# Path for the images and number check. Check ReadMe to download image data.
a_dir = os.path.join('../input/training_set/C')
b_dir = os.path.join('../input/training_set/D')
c_dir = os.path.join('../input/training_set/SD')

print('Total training C images:', len(os.listdir(a_dir)))
print('Total training D images:', len(os.listdir(b_dir)))
print('Total training SD images:', len(os.listdir(c_dir)))

a_files = os.listdir(a_dir)
b_files = os.listdir(b_dir)
c_files = os.listdir(c_dir)

# Size of images
img_width, img_height = 256, 256 


#-------------- filters of the conv. layers and the learning rate
l1 = 32
l2 = 32
l3 = 64
l4 = 128
l5 = 256

lrate = 1e-4

#------------------------

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

    
def rand(a, b):
    return np.random.rand() * (b - a) + a
      
from skimage.filters import gaussian 
def Gaussian(im):
    return gaussian(im, sigma=rand(0,5))    

#Defining image data directories and data generation
TRAINING_DIR = "../input/training_set"
training_datagen = ImageDataGenerator(rescale = 1./255,preprocessing_function=Gaussian)

VALIDATION_DIR = "../input/validation_set"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(128, 128),
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False)


validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(128, 128),
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False)


# Architecture of the model
model = tf.keras.models.Sequential([
    # First convolution
    tf.keras.layers.Conv2D(l1, (3,3), activation='relu', 
                           kernel_regularizer=regularizers.l2(0.001),
                           input_shape=(128, 128,1), padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Second convolution
    tf.keras.layers.Conv2D(l2, (3,3), activation='relu', padding='same',
                           kernel_regularizer=regularizers.l2(0.001)
                           ),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Third convolution
    tf.keras.layers.Conv2D(l3, (3,3), activation='relu', padding='same',
                           kernel_regularizer=regularizers.l2(0.001)
                           ),
    tf.keras.layers.MaxPooling2D(2,2),

    # Fourth convolution
    tf.keras.layers.Conv2D(l4, (3,3), activation='relu', padding='same',
                           kernel_regularizer=regularizers.l2(0.001)
                           ),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Fifth convolution
    tf.keras.layers.Conv2D(l5, (3,3), activation='relu', padding='same',
                           kernel_regularizer=regularizers.l2(0.001)
                           ),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Flatten and Dropout
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    
    # Hidden layer
    tf.keras.layers.Dense(l5, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


opt = keras.optimizers.Adam(learning_rate=lrate)

# Set the name of the log file (optional)
fvr0 = str(l1)+"_"+str(l2)+"_"+str(l3)+"_"+str(l4)+"_"+str(l5)+"_"+str(lrate)
filename0='_hist.csv'
filename=fvr0+filename0

history_logger=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

# Stop based on minimum validation loss with 20 step patience.
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

# Save checkpoint based on maximum validation accuracy (edit for appropriate path).
fvr = "_{epoch:04d}.hd5"
f_path = fvr0+fvr

mc = ModelCheckpoint(filepath=f_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')

# Compile and fit the model
model.compile(loss = 'categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(train_generator, epochs=500, batch_size=32, validation_data = validation_generator, verbose = 1, callbacks=[es,mc,history_logger])
