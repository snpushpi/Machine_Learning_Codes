import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl 
import numpy as np 
#reading CSV File
df = pd.read_csv("FuelConsumption.csv")
#showing a csv file
print(df.head())
df.describe()#Summarize the data 
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)
viz = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz.hist() #SHows the histogram
plt.show() #Plot things
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("FUELCONSUMPTIONS_COMB")
plt.ylabel('EMISSIONS')
plt.show()
plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS) 
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()
#Now train test split ,method for simple linear regression :)
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
#creating a linear model
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_X = np.asanyarray(train[['ENGINESIZE']])
train_Y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_X,train_Y)
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
#now plotting data :)
plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS, color='blue')
plt.plot(train_X,regr.coef_[0][0]*train_X + regr.intercept_[0],'-r')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
#now using the trained data to predict another data set
from sklearn.metrics import r2_score
test_X = np.asanyarray(test[['ENGINESIZE']])
test_Y = np.asanyarray(test[['CO2EMISSIONS']])
y_hat = regr.predict(test_X)
print("Mean Absolute Error: %.2f" % np.mean(np.absolute(y_hat-test_Y)))
print("Residual Sum of Squares: %.2f" % np.mean((test_Y-y_hat)**2)) 
print("R_2 Score: %.2f" % r2_score(y_hat , test_Y))