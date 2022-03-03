'''
Name: Ethan Santos
Course: CPSC 483 - Intro to Machine Learning
File: knncode_ethansantos.py
Purpose: Implements KNN Model by scratch to predict whether Fullerton residents are Unhappy or Happy.
'''
# Importing necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

# Finding the distances from 
def findDistances(training_set, x, y, xName, yName):
    distArr = []

    for i in range(0, len(training_set)):
        distance = math.sqrt((x - train_set[xName].iloc[i]) ** 2 + (y - train_set[yName].iloc[i]) ** 2)
        distArr.append(distance)

    training_set['Distance'] = distArr


# Importing the csv file and moving first column to the last
df = pd.read_csv(r'HappinessData-1.csv')
df = df[['City Services Availability', 'Housing Cost', 'Quality of schools', 'Community trust in local police', 'Community Maintenance', 'Availability of Community Room ', 'Unhappy/Happy']]

# Finding NAs and deleting the rows which contain them
df = df.dropna()

# Test Splitting our data 80:20 and shuffling
df = df.sample(frac = 1)
train_set_size = int(0.8 * len(df))

# Splitting data set
train_set = df[:train_set_size]
test_set = df[train_set_size:]

# Getting covariance matrix from train_set
covMatrix = train_set.corr()
pearsonMatrix = covMatrix.unstack()
pearsonMatrix = pearsonMatrix.sort_values(kind="quicksort")

# Availability of Community Room and City Services Availability have the highest correlation, 
# we can drop the rest of the variables
train_set = train_set.drop(['Housing Cost', 'Quality of schools', 'Community trust in local police', 'Community Maintenance'], axis = 1)

# Getting the distances from each point/row and then getting the K-Nearest Neighbors
# and then printing out predictions and results
sumCorrect = 0
for i in range(0, len(test_set)):
    x = test_set['City Services Availability'].iloc[i]
    y = test_set['Availability of Community Room '].iloc[i]
    print("Point: " + str(x) + ", " + str(y))
    findDistances(train_set, x, y, 'City Services Availability', 'Availability of Community Room ')
    
    k = 11
    k_result = train_set.nsmallest(k, ['Distance'])
    prediction = k_result['Unhappy/Happy'].value_counts()[:1]

    # Printing out predictions
    if prediction.index[0] == test_set['Unhappy/Happy'].iloc[i]:
        sumCorrect += 1
    if prediction.index[0] == 0:
        print("Prediction: Unhappy")
    elif prediction.index[0] == 1:
        print("Prediction: Happy")

    # Printing out actual test set values in Unhappy/Happy
    if test_set['Unhappy/Happy'].iloc[i] == 0:
        print("Result: Unhappy")
    elif test_set['Unhappy/Happy'].iloc[i] == 1:
        print("Result: Happy")

# Printing out accuracy
print(sumCorrect/len(test_set))