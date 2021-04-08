# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 19:55:18 2021

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

stock = pd.read_csv('S&P500_Stock_Data.csv')

#print(stock.describe())

#sns.pairplot(stock)

X = stock[['Interest Rates','Employment']]
Y = stock[['S&P 500 Price']] 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2)

regressor = LinearRegression(fit_intercept=(True))
regressor.fit(X_train, Y_train)

print('Linear Model Coeff (m)', regressor.coef_)
print('Linear Model Coeff (b)', regressor.intercept_)

y_predict = regressor.predict(X_test)


plt.scatter(Y_test,y_predict, color='Red')
plt.xlabel('True Values')
plt.ylabel('Model Predictions')
plt.title('Multiple Linear Regression')

k = X_test.shape[1]
n = len(X_test)

RMSE = float(format(np.sqrt(mean_squared_error(Y_test, y_predict)), '.3f'))
print('RMSE: ',RMSE)
MSE = mean_squared_error(Y_test, y_predict)
print('MSE: ',MSE)
MAE = mean_absolute_error(Y_test, y_predict)
print('MAE: ',MAE)
r2 = r2_score(Y_test, y_predict)
print('R2: ', r2)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
print('ADJ_R2: ',adj_r2)
MAPE = np.mean(np.abs((Y_test-y_predict)/Y_test)) * 100
print('MAPE: ', MAPE)

x_surf, y_surf = np.meshgrid(np.linspace(stock['Interest Rates'].min(), stock['Interest Rates'].max(), 100), np.linspace(stock['Employment'].min(), stock['Employment'].max(), 100))
onlyX = pd.DataFrame({'Interest Rates' : x_surf.ravel(), 'Employment' : y_surf.ravel()})

fittedY = regressor.predict(onlyX)
fittedY = fittedY.reshape(x_surf.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection= '3d')
ax.scatter(stock['Interest Rates'], stock['Employment'], stock['S&P 500 Price'], color='Blue', marker='x')
ax.plot_surface(x_surf, y_surf, fittedY, color='Red', alpha=0.3)
ax.set_xlabel('Interest Rates')
ax.set_ylabel('Unemployment Rates')
ax.set_zlabel('Stock Index Price')
ax.view_init(20,30)

