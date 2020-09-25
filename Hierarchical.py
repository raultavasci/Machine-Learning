import numpy as np 
import pandas as pd
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets.samples_generator import make_blobs 

df = pd.read_csv(r"\Users\Ssysuser\Desktop\New folder\Labs\ML\HierarchicalClustering\cars_clus.csv")

#Generate random data 
X1, y1 = make_blobs(n_samples=50, centers=[[4,4], [-2, -1], [1, 1], [10,4]], cluster_std=0.9)
plot scatter
plt.scatter(X1[:,0],X1[:,1], marker = "o")
agglom = AgglomerativeClustering(n_clusters = 4, linkage = 'average')
agglom.fit(X1,y1)

#Plot 
# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(6,4))

# These two lines of code are used to scale the data points down,
# Or else the data points will be scattered very far apart.

# Create a minimum and maximum range of X1.
x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0)

# Get the average distance for X1.
X1 = (X1 - x_min) / (x_max - x_min)
# This loop displays all of the datapoints.
for i in range(X1.shape[0]):
    # Replace the data points with their respective cluster value 
    # (ex. 0) and is color coded with a colormap (plt.cm.spectral)
	plt.text(X1[i, 0], X1[i, 1], str(y1[i]), #(X1[i, 0] n_samples, X1[i, 1] n_features, str(y1[i]) The integer labels for cluster membership of each sample.
			 color=plt.cm.nipy_spectral(agglom.labels_[i]/10), #/10 (or any number) change color of lables, idk why.. 
			 fontdict={'weight': 'bold', 'size': 9})      #This only plot the numbers (labels) of data
    
# Remove the x ticks, y ticks, x and y axis
plt.xticks([])
plt.yticks([])
#plt.axis('off')



# Display the plot of the original data before clustering
plt.scatter(X1[:, 0], X1[:, 1], marker='.') #With these, we combine labes and datapoints.
plt.show()

dist_matrix = distance_matrix(X1,X1) #Measures dist between all data, set x as rows and also columns, diagonal = 0. 
print(dist_matrix)
Z = hierarchy.linkage(dist_matrix, "complete")
dendro = hierarchy.dendrogram(Z)
print(Z)

#plot example from google
hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
dn1 = hierarchy.dendrogram(Z, ax=axes[0], above_threshold_color='y',
                           orientation='top')
dn2 = hierarchy.dendrogram(Z, ax=axes[1], above_threshold_color='#bcbddc', orientation='right')
hierarchy.set_link_color_palette(None)  # reset to default after use
#plt.show()