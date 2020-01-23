import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
#Example of how linear regression modeled a dataset, linear function graph
x = np.arange(-5.0, 5.0, 0.1) 
y = 2*(x)+3
y_noise = 2*np.random.normal(size=x.size) 
y_data = y + y_noise 
plt.figure(figsize=(8,6)) 
plt.plot(x, y_data,'bo')
plt.plot(x,y,'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable') 
plt.show()
# 𝑦=log(𝑎𝑥3+𝑏𝑥2+𝑐𝑥+𝑑)
#Cubic function graph
x = np.arange(-5.0, 5.0, 0.1) 
y = 1*(x**3) + 1*(x**2) + 1*x + 3
y_noise = 20*np.random.normal(size =x.size) 
y_data = y + y_noise 
plt.plot(x, y_data, 'bo') 
plt.plot(x,y,'r')
plt.ylabel('Dependent Variable') 
plt.xlabel('Independent Variable')
plt.show()
#Plotting exp function
X = np.arange(-5.0, 5.0, 0.1)
Y = np.exp(X) 
plt.plot(X,Y)
plt.ylabel('Dependent Variable') 
plt.xlabel('Independent Variable') 
plt.show() 
X = np.arange(-5.0, 5.0, 0.1) 
Y = 1 - 4/(1+np.power(3,X-2)) 
plt.plot(X,Y) 
plt.ylabel('Dependent Variable') 
plt.xlabel('Independent Variable') 
plt.show() 
# we will first plot data to see which function it fits
df = pd.read_csv("china_gdp.csv") 
cdf = df[['Year','Value']] 
plt.figure(figsize = (8,5)) 
x_data, y_data = (cdf['Year'],cdf['Value']) 
plt.plot(x_data, y_data, 'ro') 
plt.ylabel('GDP') 
plt.xlabel('Year')
plt.show()
# It looks like logistic function, so we will try plotting that

#𝑌̂ =11+𝑒𝛽1(𝑋−𝛽2), logistic functio looks like this -
X = np.arange(-5.0,5.0,0.1)
Y = 1.0/(1.0+np.exp(-X)) 
plt.plot(X,Y) 
plt.xlabel("Dependent Variable") 
plt.ylabel("Independent Variable")
plt.show() 
# Now a sample sigmoid funtion which might fit this
def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
    return y
Beta_1 = 0.10
Beta_2 = 1990.0
Y_pred = sigmoid(x_data,Beta_1,Beta_2) 
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro') 
#This is a bad model, let's first normalize our data to find out the best parameters
xdata = x_data/max(x_data)
ydata = y_data/max(y_data) 
#Now we can use curve_fit function which uses non-linear least square method to fit the
#best sigmoid function
from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata) 
print("Beta_1 = %f, Beta_2 = %f" % (popt[0], popt[1]))
#Now let's plot the function using our regression model
x = np.linspace(1960, 2015, 55) 
x = x/max(x) 
plt.figure(figsize=(8,5)) 
y = sigmoid(x, *popt) 
plt.plot(x,y,label = 'fit') 
plt.plot(xdata,ydata,label ='data') 
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()  