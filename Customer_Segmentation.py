import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 
import pandas as pd 
cust_df = pd.read_csv("Cust_Segmentation.csv") 
cust_df.head() 
df = cust_df.drop('Address',axis=1)
#Normalization is a statistical method that helps mathematical-based algorithms to interpret features with different magnitudes and distributions equally. We use StandardScaler() to normalize our dataset.
#So we need to normalize this data
from sklearn.preprocessing import StandardScaler
X = df.values[:,1:] 
X = np.nan_to_num(X) 
Clus_dataSet = StandardScaler().fit_transform(X) 
clusterNum = 3 
k_means = KMeans(init='k-means++',n_cluster=clusterNum,n_init=12)
k_means.fit(X) 
labels = k_means.labels_ 
df["Clus_km"] = labels 
#We are checking the cetroid values by averaging the features
df.groupby('Clus_km').mean() 
area = np.pi*(X[:,1])**2 
plt.scatter(X[:,0],X[:,3],s=area,c=labels.astype(np.float),alpha =0.5)
plt.xlabel('Age',fontsize=18)
plt.ylabel('Income',fontsize=16)
plt.show()  
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1,figsize=(8,6)) 
plt.cf() 
ax = Axes3D(fig,rect=[0,0,.95,1],elev=48,azim=134) 
plt.cla() 
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')
ax.scatter(X[:,1],X[:,0],X[:,3],c=labels.astype(np.float))