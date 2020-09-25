import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs 
np.random.seed(0)
x, y = make_blobs(n_samples=5000, centers=[[4,4],[-2,-1],[2,-3],[1,1]], cluster_std=0.9)
plt.scatter(x[:,0],x[:,1], marker=".")

#Initialize KMeans with these parameters, where the output parameter is called k_means.
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
#Fit x wiht the matrix created above.
k_means.fit(x)
#Get labels 
k_means_labels = k_means.labels_
print("Labels: ","\n",k_means_labels)
#We will also get the coordinates of the cluster centers using KMeans' .cluster_centers_ and save it as k_means_cluster_centers
k_means_cluster_centers = k_means.cluster_centers_
print("Cluster centers: ","\n",k_means_cluster_centers)
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data poitns that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col.
    ax.plot(x[my_members, 0], x[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()

#load a df
import pandas as pd
df = pd.read_csv(r"\Users\Ssysuser\Desktop\New folder\Labs\ML\K-means\Cust_Segmentation.csv").drop("Address", axis=1)
#print(df.head(5))
from sklearn.preprocessing import StandardScaler
#.values return a numpy representation of the DF
X = df.values[:,1:]
X = np.nan_to_num(X)
Clust_dataSet = StandardScaler().fit_transform(X)
print("\nClust dataset:\n",Clust_dataSet)

#Applying k-means
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print("\nLabels:\n", labels)
df["Clus_km"] = labels
print("\nClust km:\n",df["Clus_km"])
xx = df.groupby("Clus_km").mean()
print("###############################################################\n",xx)
#X[:,1] = Todo (:) de la columna 1 (1)
area = np.pi*(X[:,1])**2 
print(area)
print(np.count_nonzero(area==3.14159265))
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
plt.ylabel('Age', fontsize=18)
plt.xlabel('Income', fontsize=16)
plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))
plt.show()