#Some real-world applications of k-means:

#Customer segmentation
#Understand what the visitors of a website are trying to accomplish
#Pattern recognition
#Machine learning
#Data compression
#In this notebook we practice k-means clustering with 2 examples:

#k-means on a random generated dataset
#Using k-means for customer segmentation
import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 
np.random.seed(0)
#We will create own dataset this lab 
#We will us make_blobs class, this class can take as amny inputs 
#Input - (n_samples(te total number of points equally divided among clusters),centers(The number of
#centers to generate, or fixed center locations),cluster_std(The standard deviation of clusters)) 
#Output - (X(array of shape, n_samples, n features)), y(array of shape) 
X, y = make_blobs(n_samples=5000, centers=[[4,4],[-2,-1],[2,-3],[1,1]],cluster_std=0.9) 
plt.scatter(X[:,0],X[:,1],marker='.') 
#We will set up kmeans.
#The Kmeans class has many parameters, we will use three,init(Initialization method of the controids, value
# will be 'k-means++') ,selects initial cluster centers for k-mean clustering in a smart way to speed up 
#convergence,n_clusters, the number of clusters or centroids to form, n_init, number of times kmenas algo will
#run to speed 
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12) 
k_means.fit(X) 
k_means_labels = k_means.labels_ #labels for each point in the model, in other words,  the no of point closest 
#to each of them. 
k_means_cluster_centers = k_means.cluster_centers_ 
fig = plt.figure(figsize=(6,4)) #Initialize the plot with dimensions .
colors = plt.cm.Spectral(np.linspace(0,1,len(set(k_means_labels)))) #creating a color set,returning that
ax = fig.add_subplot(1,1,1) #creating plot and this func returns axises
for k,col in zip(range(4),colors): 
    #So for each color, we will color all the points with that level 
    #k ranges from 0 to 3,
    #make the list of all memmbers with that k
    my_members = (k_means_labels==k) 
    #Define the centroid or cluster center
    cluster_center = k_means_cluster_centers[k] 
    #Now we need to add color.
    #coloring data points
    ax.plot(X[my_members,0],X[my_members,1],'w',markerfacecolor=col,marker='.')
    #coloring cluster_center 
    ax.plot(cluster_center[0],cluster_center[1],'o',markerfacecolor=col, marker='.')  
#Title of the plot 
ax.set_title('KMeans') 
#removing x ticks
ax.set_xticks(())
#tremoving y ticks
ax.set_yticks(())
plt.show()



