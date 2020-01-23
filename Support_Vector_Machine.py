import pandas as pd 
import pylab as pl 
import numpy as np 
import scipy.optimize as opt 
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 
cell_df = pd.read_csv("cell_samples.csv") 
cell_df.head()  
#For the time being, we will work on two dimensional space, we can write conditions in bracket
ax = cell_df[cell_df['Class']==4][0:50].plot(kind = 'scatter',x='clump', y='UnifSize',color ='DarkBlue',label='malignant'); 
cell_df[cell_df['Class']==2][0:50].plot(kind = 'scatter',x='clump',y='UnifSize',color = 'Yellow',label='benign',ax=ax); 
plt.show() 
#For looking at data types, let's look at columns, write cell_df.dtypes 
#Now Barenuc has some dat awhich are not numeric,let's change them
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
cell_df.dtypes
feature_df = cell_df[['Clump','UnifSize','UnifShape','MargAdh','SingEpiSize','BareNuc','BlandChrom', 'NormNucl', 'Mit']] 
X = np.asarray(feature_df) 
X[0:5] 
#We want the model to predict the value of Class (that is, benign (=2) or malignant (=4)). As this field can have one of only two possible values, we need to change its measurement level to reflect this.

cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])
y [0:5] 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2,random_state=4) 
print('Train Set:',X_train.shape,y_train.shape) 
print('Test Set:',X_test.shape,y_test.shape) 
#The SVM algorithm offers a choice of kernel functions for performing its processing. Basically, mapping data into a higher dimensional space is called kernelling. The mathematical function used for the transformation is known as the kernel function, and can be of different types, such as:

#1.Linear
#2.Polynomial
#3.Radial basis function (RBF)
#4.Sigmoid 
#In this lab, we are using rbf as kernel function
from sklearn import svm 
clf = svm.SVC(kernel = 'rbf') 
clf.fit(X_train, y_train) 
yhat = clf.predict(X_test) 
yhat[0:5] 
from sklearn.metrics import classification_report,confusion_matrix 
print(confusion_matrix(y_test,yhat)) 
print(classification_report(y_test,yhat)) 


