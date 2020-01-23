#We will be looking at a clustering technique, which is Agglomerative Hierarchical Clustering. Remember that agglomerative is the bottom up approach.

#In this lab, we will be looking at Agglomerative clustering, which is more popular than Divisive clustering.

#We will also be using Complete Linkage as the Linkage Criteria. 
import numpy as np 
import pandas as pd 
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets.samples_generator import make_blobs 
#We will be generatingf a lot of data  importing make_blobs class
X1, y1= make_blobs(n_samples=50,centers =[[4,4],[-2,-1],[1,1],[10,4]],cluster_std = 0.9) 
plt.scatter(X1[:,0],X1[:,1],marker='o') 
#Agglomerative Clustering will requires two inputs
#The number of clusters to form and the type of linkage 
agglom = AgglomerativeClustering(n_clusters =4,linkage ='average') 
agglom.fit(X1,y1) 
#Let's plot the clustering-
plt.figure(figsize=(6,4)) #6 inches by 4 inches 
#axis=0 gives the minimum of each column
#axis = 1 gives the minimum of each row 
x_min, x_max = np.min(X1, axis=0),np.max(X1, axis=0)
X1 = (X1 - x_min)/(x_max - x_min) 
for i in range(X1.shape[0]):
    #replace the data point with their respective cluster value 
    #(ex. 0) and is color coded with a colormap(plt.cm,.spectral)
    #Text has multiple arguments, the first two are positions,
    #next string is just the text you want to place

    plt.text(X1[i,0],X1[i,1],str(y1[i]),color = plt.cm.nipy_spectral(agglom.labels_[i]/10),
    fontdict= {'weight':'bold','size':9}) 
plt.xticks([])
plt.yticks([]) 
#Dislay the plot of original data
plt.scatter(X1[:,0],X1[:,1],marker = '.') 
plt.show()
dist_matrix = distance_matrix(X1,X1) 
Z = hierarchy.linkage(dist_matrix,'complete')
dendro = hierarchy.dendrogram(Z) 
filename = 'cars_clus.csv'

#Read csv
pdf = pd.read_csv(filename)
print ("Shape of dataset: ", pdf.shape)
pdf.head(5) 
#let's clear the dataset by dropping the rows that have null value 
print ("Shape of dataset before cleaning: ", pdf.size)
pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)
print ("Shape of dataset after cleaning: ", pdf.size)
pdf.head(5) 
featureset = pdf[['engine_s','horsepow','wheelbas','width','length','curb_wgt','fuel_cap','mpg']] 
#Now we can normalize the feature set. MinMaxScaler transforms features by scaling each feature to a given range. It is by default (0, 1). That is, this estimator scales and translates each feature individually such that it is between zero and one 
from sklearn.preprocessing import MinMaxScaler 
x = featureset.values #Returns a numpy array
min_max_scaler = MinMaxScaler() 
feature_mtx = min_max_scaler.fit_transform(x) 
#In this part, we will do this using scikit learn
dist_matrix = distance_matrix(feature_mtx, feature_mtx) 
#Now, we can use the 'AgglomerativeClustering' function from scikit-learn library to cluster the dataset. The AgglomerativeClustering performs a hierarchical clustering using a bottom up approach. The linkage criteria determines the metric used for the merge strategy:
#Ward minimizes the sum of squared differences within all clusters. It is a variance-minimizing approach and in this sense is similar to the k-means objective function but tackled with an agglomerative hierarchical approach.
#Maximum or complete linkage minimizes the maximum distance between observations of pairs of clusters.
#Average linkage minimizes the average of the distances between all observations of pairs of clusters. 
agglom = AgglomerativeClustering(n_clusters=6, linkage = 'complete') 
agglom.fit(feature_mtx) 
agglom.labels_  
pdf['cluster_'] = agglom.labels_ 
import matplotlib.cm as cm 
n_clusters = max(agglom.lables_)+1
colors = cm.rainbow(np.linspace(0,1,n_clusters)) 
cluster_labels = list(range(0,n_clusters)) 
plt.figure(figsize=(16,14))
for color, label in zip(colors, cluster_labels):
    subset = pdf[pdf.cluster_ == label] 
    for i in subset.index:
        plt.text(subset.horsepow[i],subset.mpg[i],str(subset['model'][i]),rotation =25) 

    plt.scatter(subset.horsepow,subset.mpg) 
plt.legend()
plt.title('Clusters')
plt.xlabel('Horsepow')
plt.ylabel('mpg')