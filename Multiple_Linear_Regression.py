import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import pylab as pl 
df = pd.read_csv("FuelConsumption.csv")
df.head()
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color = 'blue')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()
#data distribution
msk = np.random.rand(len(df))<0.8
train = cdf[msk]
test = cdf[~msk]
#creating regr model by doing train test method
from sklearn import linear_model
regr = linear_model.LinearRegression()
X = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
Y = np.asanyarray(train[['CO2EMISSIONS']]) 
regr.fit(X,Y) 
print('Coefficients: ',regr.coef_)
#predicting data
y_hat = regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
X_test = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]) 
Y_test = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual Sum Of Squares: %.2f" % np.mean((y_hat-Y_test)**2)) 
print('Variance Score: %0.2f' % regr.score(X_test,Y_test))