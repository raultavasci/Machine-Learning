import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA



pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)
df = pd.read_excel(r"C:\Users\cecyt\OneDrive\Desktop\ProcesoEntrevista\indicadores_geo.xlsx",index_col=0)
#print(len(df.columns), "\n\n",df.columns)

df.drop(["clave_municipio","clave_entidad","long","lat","direccion"], axis = "columns",inplace = True)
print(df.isnull().values.any(),"\n",df.describe())
print("\nTipos de variables:\n",df.dtypes)
print(len(df.columns))

print(df.isnull().sum())

pobe = df.sort_values(by=["pobreza_e"],ascending=False).head(10)
plt.bar(pobe["mun_name"],pobe["pobreza_e"])
plt.title("Top 10 municipios con más pobreza extrema.")
plt.ylabel("Porcentaje")
plt.xticks(rotation = 75)
plt.show()
oax=df[df["entidad_federativa"]=="Oaxaca"]
print(df["entidad_federativa"].value_counts(),"\n\n\n",len(oax))


newdf = df.filter(["entidad_federativa","pobreza_e"],axis=1).reset_index()
print(newdf.groupby(["entidad_federativa"]).mean().sort_values(by=["pobreza_e"],ascending=False).head(10))
newdf1 = newdf.groupby(["entidad_federativa"]).mean().sort_values(by=["pobreza_e"],ascending=False).head(10).reset_index()
plt.bar(newdf1["entidad_federativa"],newdf1["pobreza_e"])
plt.title("Top 10 estados con más pobreza extrema")
plt.ylabel("Porcentaje")
plt.xticks(rotation = 75)
plt.show()

rez = df.sort_values(by=["ic_rezedu"],ascending=False).head(10)
plt.bar(rez["mun_name"],rez["pobreza_e"])
plt.ylabel("Porcentaje")
plt.title("Top 10 municipios con más rezago en educación.")
plt.xticks(rotation = 75)
plt.show()

rezdf = df.filter(["entidad_federativa","ic_rezedu"],axis=1)
print(rezdf.groupby(["entidad_federativa"]).mean().sort_values(by=["ic_rezedu"],ascending=False).head(10))

rezago=rezdf.groupby(["entidad_federativa"]).mean().sort_values(by=["ic_rezedu"],ascending=False).head(10).reset_index()
print(rezago["entidad_federativa"],rezago["ic_rezedu"])
plt.bar(rezago["entidad_federativa"],rezago["ic_rezedu"])
plt.title("Top 10 estados con más rezago en educación.")
plt.ylabel("Porcentaje")
plt.xticks(rotation = 75)
plt.show()


a=(df["pobreza_e_pob"].sum()/df["poblacion"].sum())*100
b=(df["pobreza_m_pob"].sum()/df["poblacion"].sum())*100
c=(df["pobreza_pob"].sum()/df["poblacion"].sum())*100


print("\nPoblación en situación de pobreza extrema: ",a,"\nPoblación en situación de pobreza moderada: ",b,"\nPoblación en situación de pobreza: ",c,"\nPorcentaje total de pobreza en México: ",a+b+c)

nopob = df.filter(["entidad_federativa","npnv"],axis=1)
print(nopob.groupby(["entidad_federativa"]).mean().sort_values(by=["npnv"],ascending=False).head(10))
nopobreza=nopob.groupby(["entidad_federativa"]).mean().sort_values(by=["npnv"],ascending=False).head(10).reset_index()
plt.bar(nopobreza["entidad_federativa"],nopobreza["npnv"])
plt.title("Top 10 estados sin pobreza, ni vulnerabilidad.")
plt.ylabel("Porcentaje")
plt.xticks(rotation = 75)
plt.show()

noreza = df.filter(["entidad_federativa","ic_rezedu","npnv"],axis=1)
print(noreza.groupby(["entidad_federativa"]).mean().sort_values(by=["npnv"],ascending=False).head(10))
norezago=noreza.groupby(["entidad_federativa"]).mean().sort_values(by=["npnv"],ascending=False).head(10).reset_index()
plt.bar(norezago["entidad_federativa"],norezago["ic_rezedu"])
plt.title("Top 10 estados indice de rezago en educación")
plt.ylabel("Porcentaje")
plt.xticks(rotation = 75)
plt.show()

pobdf = df.filter(["entidad_federativa","poblacion"],axis=1)
print(pobdf.groupby(["entidad_federativa"]).sum().sort_values(by=["poblacion"],ascending=False).head(10))

heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True,annot_kws={'size': 6},xticklabels=True, yticklabels=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':16}, pad=12);
plt.figure(figsize=(16, 6))
plt.show()


heatmap3 = sns.heatmap(df.corr()[['carencias3']].sort_values(by='carencias3', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG',xticklabels=True, yticklabels=True)
heatmap3.set_title('Correlación de variables con carencias3.', fontdict={'fontsize':10}, pad=16)
plt.figure(figsize=(16, 6))
plt.show()

heatmap4 = sns.heatmap(df.corr()[['npnv']].sort_values(by='npnv', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG',xticklabels=True, yticklabels=True)
heatmap4.set_title('Correlación de variables con No pobreza, ni vulnerabilidad.', fontdict={'fontsize':10}, pad=16)
plt.figure(figsize=(16, 6))
plt.show()


print(df["carencias3"].describe())
boxdf = df.filter(["npnv","pobreza","vul_car","vul_ing"], axis=1)
boxdf.plot.box()
plt.title("Gráfica de Boxplot")
plt.show()

boxdf2 = df.filter(["carencias3","pobreza","pobreza_e","ic_sbv","plbm"],axis=1)
boxdf2.plot.box()
plt.show()
df.drop(["entidad_federativa","mun_name"], axis = "columns",inplace = True)
# #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxx",df.shape,"\n\nCOLUMNS: \n",df.columns)
X = df.values
X = np.nan_to_num(X)
Clust_dataSet = StandardScaler().fit_transform(X)

print("\nClust dataset:\n",Clust_dataSet)

wcss=[]
for i in range (1,10):
	kmeans=KMeans(n_clusters=i,init="k-means++",random_state=0)
	kmeans.fit(X)
	wcss.append(kmeans.inertia_)

plt.plot(range(1,10),wcss)
plt.xlabel("Número de Clusters")
plt.ylabel("Within cluster sum of square")
plt.title("Método del codo")
plt.show()



import matplotlib.cm as cm
range_n_clusters = [2,3,4,5,6]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()

pca = PCA(2)
data = pca.fit_transform(X)

plt.figure(figsize=(10,10))
var = np.round(pca.explained_variance_ratio_*100, decimals = 1)
lbls = [str(x) for x in range(1,len(var)+1)]
plt.bar(x=range(1,len(var)+1), height = var, tick_label = lbls)
plt.title("Principal component analysis")
plt.xlabel("Dimensiones")
plt.ylabel("Contribución porcentual")
plt.show()





model = KMeans(n_clusters = 3, init = "k-means++")
label = model.fit_predict(data)
centers = np.array(model.cluster_centers_)
plt.figure(figsize=(10,10))
uniq = np.unique(label)
print(uniq)
for i in uniq:
   plt.scatter(data[label == i , 0] , data[label == i , 1] , label = i)
plt.scatter(centers[:,0], centers[:,1], marker="x", color='k')
plt.title("Scatterplot separado por clusters")
plt.legend()
#plt.show()

df["Clus_km"] = label
#Avg de centroids por variables
centvalues = df.groupby("Clus_km").mean()
print(centvalues[["pobreza_e","carencias3","ic_sbv","plbm"]])

print(df["Clus_km"].value_counts())
