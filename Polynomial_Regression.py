import matplotlib.pyplot as plt 
import pandas as pd 
import pylab as pl 
import numpy as np 
df = pd.read_csv("FuelConsumption.csv") 
df.head()
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
#splitting data
msk  = np.random.rand(len(df)) < 0.8
train = cdf[msk] 
test = cdf[~msk] 
#creating a regr model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model 
train_X = np.asanyarray(train[['ENGINESIZE']])
train_Y = np.asanyarray(train[['CO2EMISSIONS']])
test_X = np.asanyarray(test[['ENGINESIZE']]) 
test_Y = np.asanyarray(test[['CO2EMISSIONS']]) 
poly = PolynomialFeatures(degree = 2)
train_x_poly = poly.fit_transform(train_X) 
clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly,train_Y) 
print("Coefficients:" , clf.coef_)
print("Intercepts: ", clf.intercept_) 
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color = 'blue')
XX = np.arange(0.0,10.0,0.1) 
YY = clf.intercept_[0]+clf.coef_[0][0]*XX+ clf.coef_[0][1]*pow(XX,2)
plt.plot(XX,YY,'-r')
plt.xlabel("Engine Size")
plt.ylabel("Emissions")
#testing the model
from sklearn.metrics import r2_score 
test_x_poly = poly.fit_transform(test_X) 
test_y_ = clf.predict(test_x_poly) 
print("Mean Absolute Error: %.2f" % np.mean(np.absolute(test_y_-test_Y))) 
print("Residual Sum of Sqaures: %.2f" % np.mean((test_y_ - test_Y)**2)) 
print("R2-score: %.2f" % r2_score(test_y_ , test_Y))