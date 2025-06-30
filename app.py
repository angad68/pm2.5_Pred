#!/usr/bin/env python
# coding: utf-8

# # Citation Request
# If you find this notebook and dataset helpful for your research, we kindly request you to cite our paper related to this work. The citation details are as follows:
# Paper 1: https://ieeexplore.ieee.org/document/10086917
# "Utomo, S., John, A., Pratap, A., Jiang, Z. S., Karthikeyan, P., & Hsiung, P. A. (2023, February). AIX implementation in image-based PM2. 5 estimation: Toward an AI model for better understanding. In 2023 15th International Conference on Knowledge and Smart Technology (KST) (pp. 1-6). IEEE. DOI: https://doi.org/10.1109/KST57286.2023.10086917"
# 
# Paper 2: https://dl.acm.org/doi/abs/10.1145/3582515.3609531
# "Utomo, S., Rouniyar, A., Jiang, G. H., Chang, C. H., Tang, K. C., Hsu, H. C., & Hsiung, P. A. (2023, September). Eff-AQI: An Efficient CNN-Based Model for Air Pollution Estimation: A Study Case in India. In Proceedings of the 2023 ACM Conference on Information Technology for Social Good (pp. 165-172). DOI: https://doi.org/10.1145/3582515.3609531"
# 
# Paper 3: https://www.mdpi.com/1999-5903/15/11/371
# "Utomo, S., Rouniyar, A., Hsu, H. C., & Hsiung, P. A. (2023). Federated Adversarial Training Strategies for Achieving Privacy and Security in Sustainable Smart City Applications. Future Internet, 15(11), 371. DOI: https://doi.org/10.3390/fi15110371"

# ### nvidia-smi is script to check the available GPU in our machine

# In[ ]:


get_ipython().system('nvidia-smi')


# ### opendatasets
# Opendatasets is a Python library designed to facilitate the downloading of openly accessible datasets from Kaggle directly into the Jupyter Notebook environment. This tool streamlines the process of accessing and integrating Kaggle datasets, enriching research endeavors or educational pursuits. For detailed instructions, refer to the following link: https://www.geeksforgeeks.org/how-to-download-kaggle-datasets-into-jupyter-notebook/

# In[3]:


#"We have commented out this section as the dataset has already been downloaded."
'''
import opendatasets as od
 
od.download(
    "https://www.kaggle.com/datasets/adarshrouniyar/air-pollution-image-dataset-from-india-and-nepal")
'''


# In[1]:


import os

current_directory = os.getcwd()
print("Current Working Directory:", current_directory)


# ## Dataset Exploration
# ### Training Data

# In[2]:


import pandas as pd
df_train = pd.read_csv('/kaggle/input/air-pollution-image-dataset-from-india-and-nepal/Dataset_for_AQI_Classification/Dataset_for_AQI_Classification/train_data.csv')
df_train


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns
# Assuming AQI_Class column contains categories like 'a_Good', 'b_Moderate', etc.
# You can create a new column with the modified category labels

# Define a mapping dictionary to map the old labels to the new labels
category_mapping = {
    'a_Good': 'Good',
    'b_Moderate': 'Moderate',
    'c_Unhealthy_for_Sensitive_Groups': 'USG', 
    'd_Unhealthy' : 'Unhealthy',
    'e_Very_Unhealthy' : 'Very Unhealthy',
    'f_Severe' : 'Severe'
}

# Apply the mapping to create a new column with modified category labels
df_train['Modified_AQI_Class'] = df_train['AQI_Class'].map(category_mapping)

# Now, you can plot the count of modified categories
plt.figure(figsize=(12,6))
plt.title('Class Distribution of Training Dataset')
custom_order = ['Good', 'Moderate', 'USG', 'Unhealthy', 'Very Unhealthy', 'Severe']
sns.countplot(data=df_train,x='Modified_AQI_Class', order=custom_order, palette='Set2')


# In[4]:


import numpy as np
min_pm2_lable = np.min(df_train['PM2.5'])
max_pm2_lable = np.max(df_train['PM2.5'])
mean_pm2_lable = np.mean(df_train['PM2.5'])
stdev_pm2_lable = np.std(df_train['PM2.5'])
severe = np.count_nonzero(df_train['PM2.5'] > 250.5)
good = np.count_nonzero(df_train['PM2.5'] < 12.1)
moderate = np.count_nonzero((df_train['PM2.5'] > 12) & (df_train['PM2.5'] < 35.5))
sensitive = np.count_nonzero((df_train['PM2.5'] > 35.4) & (df_train['PM2.5'] < 55.5))
unhealthy = np.count_nonzero((df_train['PM2.5'] > 55.4) & (df_train['PM2.5'] < 150.5))
vunhealthy = np.count_nonzero((df_train['PM2.5'] > 150.4) & (df_train['PM2.5'] < 250.5))
print('Minimum label value for PM2.5 :', min_pm2_lable)
print('Maximum label value for PM2.5 :', max_pm2_lable)
print('Average label value for PM2.5 :', mean_pm2_lable)
print('Standard Deviation label value for PM2.5 :', stdev_pm2_lable)
print('Severe class based on PM2.5 value :', severe)
print('Very Unhealthy class based on PM2.5 value :', vunhealthy)
print('Unhealthy class based on PM2.5 value :', unhealthy)
print('Sensitive class based on PM2.5 value :', sensitive)
print('Moderate class based on PM2.5 value :', moderate)
print('Good class based on PM2.5 value :', good)


# In[8]:


'''
Statsmodels is a Python library that provides classes and functions for estimating and interpreting various statistical models. 
It offers a wide range of tools for statistical analysis, hypothesis testing, and data exploration. 
'''
#!pip install statsmodels


# #### In this experiment, where the focus is on estimating PM2.5 values from input images, it's essential to explore the statistical characteristics and distribution of PM2.5 values in your dataset. By conducting thorough data exploration of PM2.5 values in your dataset, you can gain valuable insights into the characteristics and distribution of air quality measurements. This understanding will inform the development and evaluation of your image-based PM2.5 estimation model.

# In[5]:


import statsmodels.api as sm
df_pm25 = df_train['PM2.5'].sort_values()
df_pm25 = df_pm25.reset_index(drop=True)
x = df_pm25.index
y = df_pm25
plt.xlabel('Index')
plt.ylabel('PM2.5 Value')
plt.title('Data Linearity of PM2.5 for Training')
plt.scatter(x, y)
plt.show()
# Fit the linear regression model
cX = sm.add_constant(x)
model = sm.OLS(y, cX).fit()
print(model.summary())


# ### Validation Data 

# In[6]:


df_val = pd.read_csv('/kaggle/input/air-pollution-image-dataset-from-india-and-nepal/Dataset_for_AQI_Classification/Dataset_for_AQI_Classification/val_data.csv')
df_val


# In[7]:


category_mapping = {
    'a_Good': 'Good',
    'b_Moderate': 'Moderate',
    'c_Unhealthy_for_Sensitive_Groups': 'USG', 
    'd_Unhealthy' : 'Unhealthy',
    'e_Very_Unhealthy' : 'Very Unhealthy',
    'f_Severe' : 'Severe'
}

# Apply the mapping to create a new column with modified category labels
df_val['Modified_AQI_Class'] = df_val['AQI_Class'].map(category_mapping)

# Now, you can plot the count of modified categories
plt.figure(figsize=(12,6))
plt.title('Class Distribution of Validation Dataset')
custom_order = ['Good', 'Moderate', 'USG', 'Unhealthy', 'Very Unhealthy', 'Severe']
sns.countplot(data=df_val,x='Modified_AQI_Class', order=custom_order, palette='Set2')


# In[8]:


min_pm2_lable = np.min(df_val['PM2.5'])
max_pm2_lable = np.max(df_val['PM2.5'])
mean_pm2_lable = np.mean(df_val['PM2.5'])
stdev_pm2_lable = np.std(df_val['PM2.5'])
severe = np.count_nonzero(df_val['PM2.5'] > 250.5)
good = np.count_nonzero(df_val['PM2.5'] < 12.1)
moderate = np.count_nonzero((df_val['PM2.5'] > 12) & (df_val['PM2.5'] < 35.5))
sensitive = np.count_nonzero((df_val['PM2.5'] > 35.4) & (df_val['PM2.5'] < 55.5))
unhealthy = np.count_nonzero((df_val['PM2.5'] > 55.4) & (df_val['PM2.5'] < 150.5))
vunhealthy = np.count_nonzero((df_val['PM2.5'] > 150.4) & (df_val['PM2.5'] < 250.5))
print('Minimum label value for PM2.5 :', min_pm2_lable)
print('Maximum label value for PM2.5 :', max_pm2_lable)
print('Average label value for PM2.5 :', mean_pm2_lable)
print('Standard Deviation label value for PM2.5 :', stdev_pm2_lable)
print('Severe class based on PM2.5 value :', severe)
print('Very Unhealthy class based on PM2.5 value :', vunhealthy)
print('Unhealthy class based on PM2.5 value :', unhealthy)
print('Sensitive class based on PM2.5 value :', sensitive)
print('Moderate class based on PM2.5 value :', moderate)
print('Good class based on PM2.5 value :', good)


# In[9]:


df_pm25_val = df_val['PM2.5'].sort_values()
df_pm25_val = df_pm25_val.reset_index(drop=True)
x = df_pm25_val.index
y = df_pm25_val
plt.xlabel('Index')
plt.ylabel('PM2.5 Value')
plt.title('Data Linearity of PM2.5 for Validation')
plt.scatter(x, y)
plt.show()
# Fit the linear regression model
cX = sm.add_constant(x)
model = sm.OLS(y, cX).fit()
print(model.summary())


# ### Testing Data 

# In[10]:


df_test = pd.read_csv('/kaggle/input/air-pollution-image-dataset-from-india-and-nepal/Dataset_for_AQI_Classification/Dataset_for_AQI_Classification/testing_data.csv')
df_test = df_test.sample(frac=1).reset_index(drop=True)
df_test


# In[11]:


category_mapping = {
    'a_Good': 'Good',
    'b_Moderate': 'Moderate',
    'c_Unhealthy_for_Sensitive_Groups': 'USG', 
    'd_Unhealthy' : 'Unhealthy',
    'e_Very_Unhealthy' : 'Very Unhealthy',
    'f_Severe' : 'Severe'
}

# Apply the mapping to create a new column with modified category labels
df_test['Modified_AQI_Class'] = df_test['AQI_Class'].map(category_mapping)

# Now, you can plot the count of modified categories
plt.figure(figsize=(12,6))
plt.title('Class Distribution of Testing Dataset')
custom_order = ['Good', 'Moderate', 'USG', 'Unhealthy', 'Very Unhealthy', 'Severe']
sns.countplot(data=df_test,x='Modified_AQI_Class', order=custom_order, palette='Set2')


# In[12]:


min_pm2_lable = np.min(df_test['PM2.5'])
max_pm2_lable = np.max(df_test['PM2.5'])
mean_pm2_lable = np.mean(df_test['PM2.5'])
stdev_pm2_lable = np.std(df_test['PM2.5'])
severe = np.count_nonzero(df_test['PM2.5'] > 250.5)
good = np.count_nonzero(df_test['PM2.5'] < 12.1)
moderate = np.count_nonzero((df_test['PM2.5'] > 12) & (df_test['PM2.5'] < 35.5))
sensitive = np.count_nonzero((df_test['PM2.5'] > 35.4) & (df_test['PM2.5'] < 55.5))
unhealthy = np.count_nonzero((df_test['PM2.5'] > 55.4) & (df_test['PM2.5'] < 150.5))
vunhealthy = np.count_nonzero((df_test['PM2.5'] > 150.4) & (df_test['PM2.5'] < 250.5))
print('Minimum label value for PM2.5 :', min_pm2_lable)
print('Maximum label value for PM2.5 :', max_pm2_lable)
print('Average label value for PM2.5 :', mean_pm2_lable)
print('Standard Deviation label value for PM2.5 :', stdev_pm2_lable)
print('Severe class based on PM2.5 value :', severe)
print('Very Unhealthy class based on PM2.5 value :', vunhealthy)
print('Unhealthy class based on PM2.5 value :', unhealthy)
print('Sensitive class based on PM2.5 value :', sensitive)
print('Moderate class based on PM2.5 value :', moderate)
print('Good class based on PM2.5 value :', good)


# In[13]:


df_pm25_test = df_test['PM2.5'].sort_values()
df_pm25_test = df_pm25_test.reset_index(drop=True)
x = df_pm25_test.index
y = df_pm25_test
plt.xlabel('Index')
plt.ylabel('PM2.5 Value')
plt.title('Data Linearity of PM2.5 for Testing')
plt.scatter(x, y)
plt.show()
# Fit the linear regression model
cX = sm.add_constant(x)
model = sm.OLS(y, cX).fit()
print(model.summary())


# # Model Development and Training Process

# In[14]:


#Import all necessary library
import sys
import numpy as np

from typing import Dict, Optional, Tuple
from pathlib import Path

import math
import pandas as pd

import tensorflow as tf
from tensorflow import keras

import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from tensorflow.keras import backend #Keras version 2.1.6
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Concatenate, LeakyReLU, Input, Conv2D, MaxPooling2D, BatchNormalization, Add 

from tensorflow.keras import layers

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image
#from PIL import Image

from sklearn.metrics import r2_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


get_ipython().run_line_magic('matplotlib', 'inline')

#os.environ["CUDA_VISIBLE_DEVICES"] = '1'


# In[15]:


#Preparing label data for Training
y_train = df_train[['AQI','PM2.5','PM10','O3','CO','SO2','NO2']].copy()
y_train


# In[16]:


#Preparing label data for Validation
y_val = df_val[['AQI','PM2.5','PM10','O3','CO','SO2','NO2']].copy()
y_val


# In[17]:


#Preparing label data for Testing
y_test = df_test[['AQI','PM2.5','PM10','O3','CO','SO2','NO2']].copy()
y_test


# In[18]:


#This function takes the path to an RGB image file as input, reads the image using Keras library and converts it to a NumPy array. 
#You can then use this array as input to your machine learning model.

def build_x(path, y, df):
    train_img = []
    for i in range(len(y)):
        img = image.load_img(path + df['Filename'][i])
        img = image.img_to_array(img)
        img = img / 255
        train_img.append(img)

    x = np.array(train_img)
    return x


# #### Converting Training Images to Array

# In[19]:


train_img = build_x('/kaggle/input/air-pollution-image-dataset-from-india-and-nepal/Air Pollution Image Dataset/Air Pollution Image Dataset/Combined_Dataset/All_img/',y_train, df_train)
train_img.shape


# #### Converting Validation Images to Array

# In[20]:


val_img = build_x('/kaggle/input/air-pollution-image-dataset-from-india-and-nepal/Air Pollution Image Dataset/Air Pollution Image Dataset/Combined_Dataset/All_img/',y_val, df_val)
val_img.shape


# #### Converting Testing Images to Array

# In[21]:


test_img = build_x('/kaggle/input/air-pollution-image-dataset-from-india-and-nepal/Air Pollution Image Dataset/Air Pollution Image Dataset/Combined_Dataset/All_img/',y_test, df_test)
test_img.shape


# In[25]:


plt.imshow(test_img[1])


# In[26]:


y_test['PM2.5'][1]


# ### Model Development
# For a detailed explanation of our model architecture, please refer to our paper available at https://dl.acm.org/doi/abs/10.1145/3582515.3609531. Our paper provides comprehensive information about the design choices, network architecture, and training procedures employed in our model for PM2.5 estimation from input images. We believe that a thorough understanding of the model architecture presented in the paper will enhance your comprehension and facilitate the replication of our results. Thank you for your interest in our work!

# In[27]:


# Create the input tensor
inputs = Input(shape=(224, 224, 3))

# Define the VGG16 convolutional layers
conv1 = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(inputs)
leak1 = LeakyReLU(alpha=0.1)(conv1)
conv2 = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(leak1)
leak2 = LeakyReLU(alpha=0.1)(conv2)
pool1 = MaxPooling2D((3, 3), strides=(2, 2), name='block1_pool')(leak2)

conv3 = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(pool1)
leak3 = LeakyReLU(alpha=0.1)(conv3)
conv4 = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(leak3)
leak4 = LeakyReLU(alpha=0.1)(conv4)
pool2 = MaxPooling2D((3, 3), strides=(2, 2), name='block2_pool')(leak4)

conv5 = Conv2D(128, (3, 3), padding='same', name='block3_conv1')(pool2)
leak5 = LeakyReLU(alpha=0.1)(conv5)
res2 = Add()([leak5,pool2])
pool3 = MaxPooling2D((3, 3), strides=(2, 2), name='block3_pool')(res2)

conv7 = Conv2D(128, (3, 3), padding='same', name='block4_conv1')(pool3)
leak7 = LeakyReLU(alpha=0.1)(conv7)
res3 = Add()([leak7,pool3])
pool4 = MaxPooling2D((3, 3), strides=(2, 2), name='block4_pool')(res3)

conv9 = Conv2D(128, (3, 3), padding='same', name='block5_conv1')(pool4)
leak9 = LeakyReLU(alpha=0.1)(conv9)
res4 = Add()([leak9,pool4])
pool5 = MaxPooling2D((3, 3), strides=(2, 2), name='block5_pool')(res4)

conv11 = Conv2D(256, (3, 3), padding='same', name='block6_conv1')(pool5)
leak11 = LeakyReLU(alpha=0.1)(conv11)
conv12 = Conv2D(256, (3, 3), padding='same', name='block6_conv2')(leak11)
leak12 = LeakyReLU(alpha=0.1)(conv12)
pool6 = MaxPooling2D((3, 3), strides=(2, 2), name='block6_pool')(leak12)

# Define the final fully-connected layers
flatten = Flatten()(pool6)
dense1 = Dense(1024)(flatten)
fcLeak1 = LeakyReLU(alpha=0.1)(dense1)
dense2 = Dense(1024)(fcLeak1)
fcLeak2 = LeakyReLU(alpha=0.1)(dense2)
#aqi = Dense(1, activation='linear', name="AQI_output")(fcLeak2)
pm25 = Dense(1, activation='linear', name="PM2.5_output")(fcLeak2)
#pm10 = Dense(1, activation='linear', name="PM10_output")(fcLeak2)
# Create the hybrid model
#model = Model(inputs=inputs, outputs=[aqi,pm25,pm10])
model = Model(inputs=inputs, outputs=pm25)

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='mae', optimizer=opt)
model.summary()


# In[29]:


#Uncomment code below if you want to load your pre-trained model for testing
#model.load_weights('./AI_fGood_20230608.best.hdf5')
#but if you don't have the pre-trained model, run the following next cells to train the model


# In[28]:


#Specify the file directory where you want to save the weights of your trained model.
weight_path="{}_20240506.weights.h5".format('LIME')


# ### Start the model training process

# In[ ]:


callback = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto'),
    ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                    save_best_only=True, mode='min', save_weights_only = True)]
#history = model.fit(x_origin_train, [y_train['AQI'],y_train['PM2.5'],y_train['PM10']], validation_data=(x_origin_valid, [y_valid['AQI'],y_valid['PM2.5'],y_valid['PM10']]), batch_size=16, epochs=150, callbacks=callback)
history = model.fit(train_img, y_train['PM2.5'], 
                    validation_data=(val_img, y_val['PM2.5']), 
                    batch_size=16, epochs=150, callbacks=callback)


# *InceptionV3*****

# In[ ]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('PM2.5 Model loss (MAE)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


y_predict = inception_model.predict(test_img).flatten()
mean_prediction = np.mean(y_predict)
print('Mean PM2.5 Prediction:', mean_prediction)


# Model evaluation
loss = inception_model.evaluate(test_img, y_test['PM2.5'], batch_size=16)
print('MAE of PM2.5 is :', loss)

y_predict = inception_model.predict(test_img)
print('R2 for PM2.5 is :', r2_score(y_test['PM2.5'], y_predict))



# In[35]:


_y_test = y_test['PM2.5'].reset_index(drop=True)
plt.plot(_y_test[:100], label='True Label')
plt.plot(y_predict[:100].tolist(), label='Estimation Value')

# set the x-axis label
plt.xlabel('Index')

# set the y-axis label
plt.ylabel('Value')

# set the plot title
plt.title('True vs Estimation (PM2.5)')

# Adding a legend
plt.legend()

# display the plot
plt.show()


# In[36]:


index = y_test['PM2.5'].index
print(index)


# In[37]:


for i in range(len(index[:10])):
    print(df_test['Filename'][index[i]])


# In[38]:


'''
The purpose of the following code is to convert the PM2.5 value to a class, 
as the model is attempting to estimate the PM2.5 value. 
Although we can obtain the MAE value after training, it is important to perform this conversion.
It is useful when calculating accuracy, F1 score, and drawing the confusion matrix.
'''


# In[39]:


#Classify the predict PM2.5 concentration to the air quality levels

y_predict_pm25 = np.zeros(len(y_predict))

for i in range(len(y_predict)):
    if y_predict[i] <= 12:
        y_predict_pm25[i] = 0
    elif y_predict[i] >= 12.1 and y_predict[i] <= 35.4:
        y_predict_pm25[i] = 1
    elif y_predict[i] >= 35.5 and y_predict[i] <= 55.4:
        y_predict_pm25[i] = 2
    elif y_predict[i] >= 55.5 and y_predict[i] <= 150.4:
        y_predict_pm25[i] = 3
    elif y_predict[i] >= 150.5 and y_predict[i] <= 250.4:
        y_predict_pm25[i] = 4
    elif y_predict[i] > 250.4:
        y_predict_pm25[i] = 5
    else:
        print('Exception Occured!')
    
y_predict_pm25 = y_predict_pm25.astype(int)
    
    
y_predict_pm25


# In[40]:


#Classify the Ground Truth PM2.5 concentration to the air quality levels

y_test_pm25 = np.zeros(len(_y_test))

for i in range(len(_y_test)):
    if _y_test[i]  <= 12:
        y_test_pm25[i] = 0
    elif _y_test[i] >= 12.1 and _y_test[i] <= 35.4:
        y_test_pm25[i] = 1
    elif _y_test[i] >= 35.5 and _y_test[i] <= 55.4:
        y_test_pm25[i] = 2
    elif _y_test[i] >= 55.5 and _y_test[i] <= 150.4:
        y_test_pm25[i] = 3
    elif _y_test[i] >= 150.5 and _y_test[i] <= 250.4:
        y_test_pm25[i] = 4
    elif _y_test[i] > 250.4:
        y_test_pm25[i] = 5
    else:
        print('Exception Occured!')

y_test_pm25 = y_test_pm25.astype(int)
        
        
y_test_pm25


# In[41]:


#---Balanced Accuracy for PM2.5---------------

balanced_accuracy_score(y_test_pm25, y_predict_pm25)


# In[42]:


#---Classification Accuracy for PM2.5---------

t = 0
n = 0

for i in range(len(y_predict_pm25)):
    if y_predict_pm25[i] == y_test_pm25[i]:
        t = t + 1
    else:
        n = n + 1
        
acc = t / len(y_predict_pm25)

print('Acc: ', acc, ' True: ', t, ' False: ', n)


# In[43]:


#---Macro F1 Score for PM2.5------------------

f1_score(y_test_pm25, y_predict_pm25, average='macro')


# In[44]:


Y_pred_classes = y_predict_pm25
Y_true = y_test_pm25
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax, annot_kws={'fontsize': 14})
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix of PM2.5")
plt.show()


# In[45]:


# Row-wise normalization
normalized_matrix_row = confusion_mtx.astype('float') / confusion_mtx.sum(axis=1)[:, np.newaxis]

# Overall normalization
normalized_matrix_overall = confusion_mtx.astype('float') / confusion_mtx.sum()

# Plot the original confusion matrix
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
sns.heatmap(confusion_mtx, annot=True, cmap='Blues', cbar=False, square=True)
plt.title('Original Confusion Matrix')

# Plot the row-wise normalized confusion matrix
plt.subplot(1, 3, 2)
sns.heatmap(normalized_matrix_row, annot=True, cmap='Blues', cbar=False, square=True)
plt.title('Row-wise Normalized')

# Plot the overall normalized confusion matrix
#plt.subplot(1, 3, 3)
#sns.heatmap(normalized_matrix_overall, annot=True, cmap='Blues', cbar=False, square=True)
#plt.title('Overall Normalized')

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()


# In[46]:


'''
In order to demonstrate how our model determines the output based on the relevant feature from the input photos, 
we make use of LIME. For more infromation please go to https://github.com/marcotcr/lime
Please uncomment code below to install LIME library
'''
#!pip install lime


# In[ ]:





# In[47]:


from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from PIL import Image


# In[48]:


'''
#This code uses the LIME image explainer to explain all of the images that are being tested, 
#and then it saves the explanation to a specified folder.
explainer = lime_image.LimeImageExplainer()
for i in range(len(index)):
    explanation = explainer.explain_instance(test_img[i], model.predict, top_labels=1, hide_color=0, num_samples=500)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False)
    explained_image = mark_boundaries(temp / 2 + 0.5, mask)
    
    # Save the explained image
    image_path = df_test['Filename'][index[i]]
    explained_image_path = "./LIME_Explanation/lime_" + image_path
    explained_image_pil = Image.fromarray((explained_image * 255).astype(np.uint8))
    explained_image_pil.save(explained_image_path)

'''
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(test_img[2], model.predict, top_labels=1, hide_color=0, num_samples=500)
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False)
explained_image = mark_boundaries(temp / 2 + 0.5, mask)
plt.imshow(explained_image)
plt.show()
    


# In[49]:


explanation = explainer.explain_instance(test_img[25], model.predict, top_labels=1, hide_color=0, num_samples=500)
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False)
explained_image = mark_boundaries(temp / 2 + 0.5, mask)
plt.imshow(explained_image)
plt.show()


# In[50]:


'''
import matplotlib.image as mpimg

# Specify the folder containing the images
folder_path = "./LIME_Explanation/"

# Get a list of files in the folder
files = os.listdir(folder_path)

# Display the first 16 images of LIME explanation 
num_images_to_display = 16
fig, axes = plt.subplots(4, 4, figsize=(12, 12))

for i, ax in enumerate(axes.flat):
    if i < num_images_to_display:
        img = mpimg.imread(os.path.join(folder_path, files[i]))
        ax.imshow(img)
        ax.axis('off')

plt.tight_layout()
plt.show()
'''

