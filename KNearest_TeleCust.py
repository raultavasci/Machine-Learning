import itertools	
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import NullFormatter
from sklearn import preprocessing

df = pd.read_csv(r"\Users\Ssysuser\Desktop\New folder\Labs\ML\teleCust1000t.csv")
print(df["income"].head(10))
print(df["custcat"].value_counts())
df.hist(column='income', bins=50)
#plt.show()
print(df.columns)
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].to_numpy() #.astype(float)
y = df["custcat"].to_numpy()
print("X Before standarizing:" , X[0:5])
#Standarize data to givr zero mean and unit variance 
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print("After",X[0:5], y[0:5])

#Train test split 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size = 0.2, random_state = 4)
print ('Train set:', X_train.shape,  Y_train.shape)
print ('Test set:', X_test.shape,  Y_test.shape)

#K-NEAREST NEIGHBORS
from sklearn.neighbors import KNeighborsClassifier
k = 4
#train and predict
neigh=KNeighborsClassifier(n_neighbors = k).fit(X_train,Y_train)
print("\n\n****************************************************************************************************************************\n",neigh)
#Predict.
yhat=neigh.predict(X_test)
print("dlfkjdjjjjjjjjjjjjj",yhat[0:5],Y_test[0:5])

#ACCURACY
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(Y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(Y_test, yhat))
Ks=10 
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
confusionMx = [];
print("************************",np.std(yhat),np.std(Y_test), np.std(yhat==Y_test))

for n in range(1,Ks):	
#Train model and predict
	neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, Y_train)
	yhat = neigh.predict(X_test)
	mean_acc[n-1] = metrics.accuracy_score(Y_test, yhat)
	std_acc[n-1]=np.std(yhat==Y_test)/np.sqrt(yhat.shape[0])

print(mean_acc,std_acc)

plt.plot(mean_acc,range(1,Ks))
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.y  ticks(np.arange(min(range(1,Ks)), max(range(1,Ks))+1, 1.0))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
