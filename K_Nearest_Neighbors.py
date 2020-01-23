import itertools
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.ticker import NullFormatter 
import pandas as pd 
import matplotlib.ticker as ticker 
from sklearn import preprocessing 
df = pd.read_csv('teleCust1000t.csv') 
df.head()  
df['custcat'].value_counts() #Just counting the values
df.hist(column='income', bins=50) #
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
y = df['custcat'].values
#Now we will normalize the data, data standardization gives zero mean and unit variance
#It is a good practice, specially for good algorithms like KNN
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
#splitting the data for training and then testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4) 
from sklearn.neighbors import KNeighborsClassifier 
neigh = KNeighborsClassifier(n_neighbors=4).fit(X_train, y_train) 
yhat = neigh.predict(X_test) 
from sklearn import metrics 
print("Train Set Accuracy:", metrics.accuracy_score(y_train, neigh.predict(X_train))) 
print("Test Set Accuracy:", metrics.accuracy_score(y_test,yhat)) 
#Now we will more values of k to get the most correct value
Ks = 10 
mean_acc = np.zeros((Ks-1)) 
std_acc = np.zeros((Ks-1)) 
ConfustionMx = []; 
for n in range(1,Ks):
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train) 
    yhat = neigh.predict(X_test) 
    mean_acc[n-1]=metrics.accuracy_score(y_test,yhat) 
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show() 