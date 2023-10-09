#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[2]:


mag_data = pd.read_csv('data_participant_SBQuantum.csv')
mag_data


# In[3]:


# step 1: clean data
# df = mag_data.drop(columns=['experience_key'])
df_filled = mag_data.fillna(0) #fill NaN with 0's
df_filled


# In[4]:


# get predicting set and dataset
prediction_data = df_filled[df_filled['C12_magnetometer_Bx'] == 0]
dataset = df_filled[df_filled['C12_magnetometer_Bx'] != 0]

prediction_data


# In[5]:


# step 3: split into training and testing sets
X = dataset.drop(columns=['experience_key','C12_magnetometer_Bx', 'C12_magnetometer_By', 'C12_magnetometer_Bz', 'C12_magnetometer_Bnorm'])
X

y = dataset[['C12_magnetometer_Bx', 'C12_magnetometer_By', 'C12_magnetometer_Bz']]
y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[6]:


# step 4 & 5: create and train the model
model = LinearRegression()
model.fit(X_train, y_train)


# In[7]:


# test the model
test_predictions = model.predict(X_test)
test_predictions

r_squared = r2_score(y_test, test_predictions)
r_squared


# In[8]:


# step 6: make predictions
predictions = model.predict(prediction_data.drop(columns=['experience_key', 'C12_magnetometer_Bx', 'C12_magnetometer_By', 'C12_magnetometer_Bz', 'C12_magnetometer_Bnorm']))
predictions


# In[9]:


# add results to prediction data
prediction_data.loc[:, ['C12_magnetometer_Bx', 'C12_magnetometer_By', 'C12_magnetometer_Bz']] = predictions

# calculate Bnorm
prediction_data.loc[:, 'C12_magnetometer_Bnorm'] = \
np.linalg.norm(prediction_data[['C12_magnetometer_Bx', 'C12_magnetometer_By', 'C12_magnetometer_Bz']], axis=1)

prediction_data


# In[10]:


# step 7: aggregate the data with the predictions
final_predictions = pd.concat([dataset, prediction_data])

# save to output csv file
final_predictions.to_csv('output_SBQuantum.csv', index=False)
final_predictions

