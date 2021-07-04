

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from tensorflow import keras
from tqdm import tqdm
import tensorflow
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
K.clear_session()
import itertools
import matplotlib.pyplot as plt
import cv2
import matplotlib.cm as cm

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.applications import *
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder


#Data to be loaded as X_train,Y_train and X_test,Y_test with onehot encoded labels

image_width=224
image_height=224
no_of_channels=3
input_shape=(image_width,image_height,no_of_channels)



reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.1,
                                         patience=5,
                                         cooldown=2,
                                         min_lr=1e-8,
                                         verbose=1)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
)

EPOCHS = 100

"""# **Xception**"""


tmodel_base = Xception(input_shape = input_shape, 
                                include_top = False, 
                                weights = 'imagenet')
for layer in tmodel_base.layers:
    layer.trainable = False

#Getting desired layer output
# Modification of pretrained model
last_layer = tmodel_base.get_layer('block14_sepconv2_act')
last_output = last_layer.output

x = MaxPooling2D(strides=(2,2))(last_output)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.15)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(2, activation='softmax')(x)
#Compiling model
model2 = Model(inputs = tmodel_base.input, outputs = x, name = 'Our_Xception')
model2.summary()

opt1 = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999)
xception_checkpoint = ModelCheckpoint("xception_best.h5", monitor='val_accuracy', verbose=1,
    save_best_only=True, mode='auto', period=1)

model2.compile(optimizer = opt1 , loss = 'categorical_crossentropy', metrics = ['accuracy'])

history2 = model2.fit(X_train, Y_train, epochs=EPOCHS, validation_data = (X_test,Y_test)
                     ,class_weight=class_weight ,callbacks=[ xception_checkpoint])


"""# **ResNet**"""

from tensorflow.keras.applications.resnet50 import ResNet50
num_classes= 3
tmodel_base = ResNet50(input_shape = input_shape, 
                                include_top = False, 
                                weights = 'imagenet')
for layer in tmodel_base.layers:
    layer.trainable = False

last = tmodel_base.output


x = Conv2D(1024,(2,2),strides=(1,1))(last)
x = Flatten()(x)
x = Dense(512, activation = 'relu')(x)
x = Dropout(rate = 0.15)(x)
x = Dense(256, activation = 'relu')(x)
x = Dropout(rate = 0.25)(x)
x = Dense(2, activation = 'softmax')(x)

#Compiling model
model3 = Model(inputs = tmodel_base.input, outputs = x)
model3.summary()

opt1 = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999)
resnet_checkpoint = ModelCheckpoint("resnetbest.h5", monitor='val_accuracy', verbose=1,
    save_best_only=True, mode='auto', period=1)

model3.compile(optimizer = opt1 , loss = 'categorical_crossentropy', metrics = ['accuracy'])
history3 = model3.fit(X_train, Y_train,epochs=EPOCHS, validation_data = (X_test, Y_test)
                      ,class_weight=class_weight ,callbacks=[resnet_checkpoint])


"""# **VGG16**"""

from tensorflow.keras.applications import VGG16
tmodel_base = VGG16(input_shape = input_shape, 
                                include_top = False, 
                                weights = 'imagenet')
for layer in tmodel_base.layers:
    layer.trainable = False

#Getting desired layer output
last_layer = tmodel_base.get_layer('block5_pool')
last = last_layer.output

x = Flatten()(last)
x = Dense(512, activation = 'relu')(x)
x = Dropout(rate = 0.25)(x)
x = Dense(2, activation = 'softmax')(x)
#Compiling model
model1 = Model(inputs = tmodel_base.input, outputs = x, name = 'VGG16')
opt1 = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999)


model1.compile(optimizer = opt1 , loss = 'categorical_crossentropy', metrics = ['accuracy'])
model1.summary()

vgg_checkpoint = ModelCheckpoint("vgg_best.h5", monitor='val_loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

history1 = model1.fit(X_train, Y_train,  epochs=EPOCHS, validation_data = (X_test, Y_test)
                       ,class_weight=class_weight ,callbacks=[vgg_checkpoint])



