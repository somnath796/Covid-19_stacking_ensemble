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
n_classes=2

"""# **Evaluation**"""

our_vgg = tf.keras.models.load_model('./vgg_best.h5')
our_xcep = tf.keras.models.load_model('./xception_best.h5')
our_res = tf.keras.models.load_model('./resnetbest.h5')

print("-"*30)
print("VGG16")
print(our_vgg.evaluate(X_test,Y_test))
print("-"*30)
print("Xception")
print(our_xcep.evaluate(X_test,Y_test))
print("-"*30)
print("ResNet50")
print(our_res.evaluate(X_test,Y_test))
print("-"*30)

our_vgg.trainable = False
our_xcep.trainable = False
our_res.trainable = False

#Delete unused variables and clear garbage values
#del history1, model1, hist_df
import gc

gc.collect()

"""# **Ensemble**"""

def stacking_ensemble(members,input_shape,n_classes):
    commonInput = Input(shape=input_shape)
    out=[]

    for model in members:    
        model._name= model.get_layer(index = 0)._name +"-test"+ str(members.index(model)+1)
        out.append(model(commonInput))

    modeltmp = concatenate(out,axis=-1)
    #modeltmp = Dense(256, activation='relu')(modeltmp)
    modeltmp = Dense(128, activation='relu')(modeltmp)
    modeltmp = Dropout(0.1)(modeltmp)
    modeltmp = Dense(n_classes, activation='softmax')(modeltmp)
    stacked_model = Model(commonInput,modeltmp)
    stacked_model.compile( loss='categorical_crossentropy',optimizer= optimizer, metrics=['accuracy'])

    return stacked_model

members =[our_vgg, our_xcep, our_res]

batch=16
optimizer= Adam(lr=5e-5, beta_1=0.9, beta_2=0.999)

stacked = stacking_ensemble(members,(image_height,image_width,3),n_classes)
print(stacked.summary())

stacked_checkpoint = ModelCheckpoint("stacked_best.h5", monitor='val_accuracy', verbose=1,
    save_best_only=True, mode='auto', period=1)

stacked_hist = stacked.fit(X_train,Y_train,
                            epochs=EPOCHS, #epochs,
                            verbose=1,
                            batch_size = 16,
                            validation_data= (X_test,Y_test),
                            class_weight = class_weight,
                            callbacks=[reduce_learning_rate, stacked_checkpoint]) 



stacked.evaluate(X_test, Y_test)

del stacked_hist, stacked
gc.collect()

stacked = tf.keras.models.load_model("/content/stacked_best.h5")
print("-.-"*30)
print('Model Name: Stacked')
print(stacked.evaluate(X_test, Y_test))
print("-.-"*30)

"""# **Finding F1 score etc.**"""

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import classification_report, confusion_matrix

# True values
y_true = Y_test.argmax(axis=-1)

def confusion(model):
  #Predicted values
  prob = model.predict(X_test)
  y_pred = prob.argmax(axis= -1)
  # Print the confusion matrix
  print("--"*30)
  print(confusion_matrix(y_true, y_pred))
  print("--"*30)
  # Print the precision and recall, among other metrics
  print(classification_report(y_true, y_pred, digits=6))
  print("--"*30)

our_models =  {"VGG16": our_vgg, "Xception": our_xcep, "ResNet50": our_res, "Stacked": stacked}
for m in our_models:
  print("--"*30)
  print(m)
  print("--"*30)
  confusion(our_models[m])
