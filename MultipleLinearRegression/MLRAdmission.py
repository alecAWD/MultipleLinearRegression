# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:26:45 2021

@author: Alec
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D

admission = pd.read_csv('Admission.csv')
admission = admission.drop(['Serial No.'], axis=1)
print(admission.describe())
#sns.heatmap(admission.corr(), annot= True)
#sns.pairplot(admission)

X = admission.drop(['Admission Chance'], axis=1)
Y = admission[['Admission Chance']]

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2)

regressor = LinearRegression(fit_intercept=(True))
regressor.fit(X_train, y_train)

print('Coefficient (m): ', regressor.coef_)
print('Coefficient (b): ', regressor.intercept_)

y_predict = regressor.predict(X_test)

plt.scatter(y_test,y_predict, color='Red')
plt.ylabel('Model Predictions')
plt.xlabel('True Values')

k = X_test.shape[1]
n = len(X_test)
RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)), '.3f'))
print('RMSE: ',RMSE)
MSE = mean_squared_error(y_test, y_predict)
print('MSE: ',MSE)
MAE = mean_absolute_error(y_test, y_predict)
print('MAE: ',MAE)
r2 = r2_score(y_test, y_predict)
print('R2: ', r2)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
print('ADJ_R2: ',adj_r2)
MAPE = np.mean(np.abs((y_test-y_predict)/y_test)) * 100
print('MAPE: ', MAPE)