#Code for logistic regression
import pandas as pd 
import pylab as pl 
import numpy as np 
import scipy.optimize as opt 
from sklearn import preprocessing 
import matplotlib.pyplot as plt
churn_df = pd.read_csv("ChurnData.csv") 
churn_df.head() 
churn_df = churn_df[['tenure','age','address','income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']] 
#counting total rows and columns
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']]) 
y = np.asarray(churn_df[['churn']])
total_rows = len(churn_df.axes[0]) 
total_cols = len(churn_df.axes[1]) 
# We will now process data
from sklearn import preprocessing 
X = preprocessing.StandardScaler().fit(X).transform(X) 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 4) 
print('Train Set:', X_train.shape, y_train.shape) 
print('Test Set',X_test.shape, y_test.shape)
#building a logistic regression model
#The version of Logistic Regression in Scikit-learn, support regularization. Regularization is a technique used to solve the overfitting problem in machine learning models. C parameter indicates inverse of regularization strength which must be a positive float. Smaller values specify stronger regularization. Now lets fit our model with train set:
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix 
LR = LogisticRegression(C=0.01,solver = 'liblinear').fit(X_train,y_train)
yhat = LR.predict(X_test) 
#predict_proba returns estimates for all classes, ordered by the label of classes. So, the first column is the probability of class 1, P(Y=1|X), and second column is probability of class 0, P(Y=0|X)
yhat_prob = LR.predict_proba(X_test) 
# We will use jaccard index for getting accuracy :) which is intersection divided by union of two sets
from sklearn.metrics import jaccard_similarity_score 
jaccard_similarity_score(y_test,yhat)
# We will use another thing which is a confusion matrix. 


