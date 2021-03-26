'''1. Question or problem definition
   2. Acquire training data and testing data
   3. Wrangle, prepare and cleanse the data
   4. Analyze, identify patterns and explore the data
   5. Visualize, report, present the problem solving steps and final solution
   6. Supply and submit results
'''
'''Classifying, Correlating, Converting, Completing, Correcting, Creating,
Charting'''

import pandas as pd
import numpy as np
import random as rnd

import matplotlib.pyplot as lt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

test_data = pd.read_csv('/home/robincoal/projects/titanic/test.csv')
train_data = pd.read_csv('/home/robincoal/projects/titanic/train.csv')

print(train_data.columns.values)
print(train_data.tail())

train_data.info()
print('_'*40)
print(test_data.info())

train_data.describe()
train_data.describe(include=['O'])

