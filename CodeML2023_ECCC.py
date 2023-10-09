#!/usr/bin/env python
# coding: utf-8

# Import necessary packages and modules
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Gather data
df = pd.read_csv('data_participant.csv')

# Split data into two (one with missing y values and one without)
missing_y = df['y_thunderstorm'].isna() & df['y_hail'].isna() & df['y_severe'].isna()
df_predict = df[missing_y]
df_remaining = df[~missing_y]

# Split df_remaining into data frames that are/are not missing hail_size values
missing_hail_size = df_remaining['hail_size'].isna()
df_wo_hail_size = df_remaining[missing_hail_size]
df_cleaned = df_remaining[~missing_hail_size]

# Isolate outputs with values
y1 = df_cleaned['y_thunderstorm']
y2 = df_cleaned['y_hail']
y3 = df_cleaned['y_severe']

# Train models for y_thunderstorm, y_hail, y_severe
model1 = DecisionTreeClassifier()
X_train, X_test, y1_train, y1_test = train_test_split(df_cleaned.drop(columns=['ID', 'year', 'month', 'day', 'hour', 'start_time', 'latitude', 'longitude', 'event', 'y_thunderstorm', 'y_hail', 'y_severe']), y1.astype(int), test_size=0.2)
model1.fit(X_train, y1_train)

# predictions1 = model1.predict(X_test)
# score1 = accuracy_score(y1_test, predictions1)
# score1

model2 = DecisionTreeClassifier()
X_train, X_test, y2_train, y2_test = train_test_split(df_cleaned.drop(columns=['ID', 'year', 'month', 'day', 'hour', 'start_time', 'latitude', 'longitude', 'event', 'y_thunderstorm', 'y_hail', 'y_severe']), y2.astype(int), test_size=0.2)
model2.fit(X_train, y2_train)

# predictions2 = model2.predict(X_test)
# score2 = accuracy_score(y2_test, predictions2)
# score2

model3 = DecisionTreeClassifier()
X_train, X_test, y3_train, y3_test = train_test_split(df_cleaned.drop(columns=['ID', 'year', 'month', 'day', 'hour', 'start_time', 'latitude', 'longitude', 'event', 'y_thunderstorm', 'y_hail', 'y_severe']), y3.astype(int), test_size=0.2)
model3.fit(X_train, y3_train)

# predictions3 = model3.predict(X_test)
# score3 = accuracy_score(y3_test, predictions3)
# score3

# Predict missing values of y_thunderstorm, y_hail, y_severe
y1_predict = model1.predict(df_predict.drop(columns=['ID', 'year', 'month', 'day', 'hour', 'start_time', 'latitude', 'longitude', 'event', 'y_thunderstorm', 'y_hail', 'y_severe'])).astype(bool)
y2_predict = model2.predict(df_predict.drop(columns=['ID', 'year', 'month', 'day', 'hour', 'start_time', 'latitude', 'longitude', 'event', 'y_thunderstorm', 'y_hail', 'y_severe'])).astype(bool)
y3_predict = model3.predict(df_predict.drop(columns=['ID', 'year', 'month', 'day', 'hour', 'start_time', 'latitude', 'longitude', 'event', 'y_thunderstorm', 'y_hail', 'y_severe'])).astype(bool)

# Replace missing y-values with predictions
df_predict_new = df_predict.drop(columns=['y_thunderstorm', 'y_hail', 'y_severe'])
df_predict_new['y_thunderstorm'] = y1_predict
df_predict_new['y_hail'] = y2_predict
df_predict_new['y_severe'] = y3_predict

# Combine all sub data frames into complete final data frame
df_final = pd.concat([df_predict_new, df_wo_hail_size, df_cleaned], ignore_index=True)

# Save .csv file
df_final.to_csv('output.csv', index=False)