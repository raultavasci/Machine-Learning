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
print("\nBefore cleaning: ",df.size)
df[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = df[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce') # If ‘coerce’, then invalid parsing will be set as NaN
df = df.dropna() #Default = 0 (rows)
df = df.reset_index(drop=True)
print ("Shape of dataset after cleaning: ", df.size)
print(df.head(5))

#Feature selection and normalization 
featureset = df[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]
from sklearn.preprocessing import MinMaxScaler
x = featureset.values #returns a numpy array
#print("X=",x)
#min_max_scaler transforms features by scaling each feature to a given range. It is by default (0, 1).
#That is, this estimator scales and translates each feature individually such that it is between zero and one.
min_max_scaler = MinMaxScaler() 
feature_mtx = min_max_scaler.fit_transform(x)
print(feature_mtx[0:5])
#Clustering using scipy
import scipy
leng = feature_mtx.shape[0]
D = scipy.zeros([leng,leng])
for i in range(leng):
	for j in range(leng):
		D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])

import pylab
import scipy.cluster.hierarchy
Z = hierarchy.linkage(D,"complete")
print(Z)


#Criterion decides how the max_d, k are used in fn 
from scipy.cluster.hierarchy import fcluster
max_d = 3
clusters = fcluster(Z,max_d, criterion ="distance")
print("\n\n", clusters)

from scipy.cluster.hierarchy import fcluster
k = 5
clusters = fcluster(Z, k, criterion='maxclust')
print("\n\n\n", clusters)

fig = pylab.figure(figsize=(18,50))
def llf(id):
	return '[%s %s %s]' % (df['manufact'][id], df['model'][id], int(float(df['type'][id])) )
    
dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =4, orientation = 'right')
plt.show()

#Clustering with scikit learn 
agglom = AgglomerativeClustering(n_clusters  = 6, linkage = "complete")
agglom.fit(feature_mtx)
print(agglom.labels_)
df["clusters_"] = agglom.labels_
print(df.head())

import matplotlib.cm as cm
n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(16,14))

for color, label in zip(colors, cluster_labels):
    subset = df[df.clusters_ == label]
    for i in subset.index:
            plt.text(subset.horsepow[i], subset.mpg[i],str(subset['model'][i]), rotation=25) 
    plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10, c=color, label='cluster'+str(label),alpha=0.5)
#    plt.scatter(subset.horsepow, subset.mpg)
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
plt.show()



df.groupby(['clusters_','type'])['clusters_'].count()
agg_cars = df.groupby(['clusters_','type'])['horsepow','engine_s','mpg','price'].mean()
print(agg_cars)
plt.figure(figsize=(16,10))
for color, label in zip(colors, cluster_labels):
    subset = agg_cars.loc[(label,),]
    for i in subset.index:
        plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k')
    plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, label='cluster'+str(label))
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
plt.show()