# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 18:07:32 2019

@author: Blisston Kirubha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Life Expectancy Data.csv')
dataset = dataset.iloc[:,:].values

#Preprocessing

from sklearn.preprocessing import Imputer

#Missing values
imputer = Imputer(missing_values ="NaN", strategy="mean",axis = 0 )
dataset[:,3:] = imputer.fit_transform(dataset[:,3:])

#Feature scaling

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
dataset[:,0] = le.fit_transform(dataset[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
dataset = onehotencoder.fit_transform(dataset).toarray()