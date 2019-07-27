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
dataset[:,2] = le.fit_transform(dataset[:,2])
onehotencoder = OneHotEncoder(categorical_features=[0])
dataset = onehotencoder.fit_transform(dataset).toarray()
x = dataset[:,:195]
x = np.append(x,dataset[:,196:215],axis = 1)
y = dataset[:,195]
#Training set test set
x=x[:, 1:]


import statsmodels.formula.api as sm

np.append(np.ones((2938,1)).astype(int),values = x,axis = 1)

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print(regressor_OLS.summary())
    return x
SL = 0.05
x = x[:, :]
X_Modeled = backwardElimination(x, SL)
x= X_Modeled


from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression

lr = LinearRegression();
lr.fit(x_train,y_train)


y_pred = lr.predict(x_test)
accuracy = lr.score(x_test,y_test)
print(accuracy)
