#Density, in this context, is defined as the number of points within a specified radius.
import numpy as np 
from sklearn.cluster import DBSCAN 
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 
#This function will generate data points
#and requires these inputs 
#Coordinates of the centroids that will generate 
#random data, centroidLocation, numSamples,clusterDeviation(Standard Deviation between the 
# clusters)
def createDataPoints(centroidLocation, numSamples, clusterDeviation): 
    X, y= make_blobs(n_samples=numSamples, centers= centroidLocation,
    clusterstd = clusterDeviation) 
    X = StandardScaler().fit_transform(X) 
    return X, y
X, y = createDataPoints([[4,3], [2,-1], [-1,4]] , 1500, 0.5) 
#Modeling 
#DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise. This technique is one of the most common clustering algorithms which works based on density of object. The whole idea is that if a particular point belongs to a cluster, it should be near to lots of other points in that cluster.
#Works on two basic parameters.
#Epsilon and minimumSmaples, epsilon is the radius and minimumSamples 
#determine the minimum number of data points we want in a neighborhood to 
#define a cluster
epsilon = 0.3 
minimumSamples = 7 
db = DBSCAN(eps = epsilon, min_samples= minimumSamples).fit(X) 
labels = db.labels_ 
# Firts, create an array of booleans using the labels from db.
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
core_samples_mask 
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) 
# Remove repetition in labels by turning it into a set.
unique_labels = set(labels) 
colors = plt.cm.Spectral(np.linspace(0,1,len(unique_labels))) 
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = 'k'
    class_member_mask = (labels==k) 
    #Plot the datapoints that are clustered 
    xy = X[class_member_mask & core_samples_mask] 
    plt.scatter(xy[:,0],xy[:,1],s = 50, c = [col], marker = u'o', alpha =0.5) 
    #Plot the outliers 
    xy = X[class_member_mask &~ core_samples_mask]
    plt.scatter(xy[:,0],xy[:,1],s=50, c=[col], marker=u'o', alpha = 0.5)    