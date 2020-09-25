import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_similarity_score
df = pd.read_csv(r"\Users\Ssysuser\Desktop\New folder\Labs\ML\LogisticReg\ChurnData.csv")
print(df.head(4))
df = df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
#algorithm doenst work if the target variable is != int
df["churn"] = df["churn"].astype(int)
print(df.shape,df.columns)	
#Define x, y as a np array 
X = np.asarray(df[["tenure",'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(df["churn"])
print("\n\n",X[0:5],"\n",y[0:5])
#Standarize data 
X = preprocessing.StandardScaler().fit(X).transform(X)
print("\n\n",X[0:5])
#Train and split data
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,y, test_size = 0.2, random_state=4)
print("\n\n******************************\n","Train set: ",Xtrain.shape," ",Ytrain.shape,"\nTest set: ",Xtest.shape," ",Ytest.shape)
#LR
LR = LogisticRegression(C =0.01, solver = "liblinear").fit(Xtrain,Ytrain)
print("\n LR = ",LR)
yhat = LR.predict(Xtest)
print(yhat)
#predict_proba output => A= P(Y=1|X), B= P(Y=0|X)
yhat_prob = LR.predict_proba(Xtest)
print("\n",yhat_prob)
#Uses Jaccard index to test accuracy 1 = perfect, ->0 = bad
Acc = jaccard_similarity_score(Ytest,yhat)
print("\n",Acc)
#Confusion matrix to look for accuracy
from sklearn.metrics import classification_report
import itertools
def plot_confussion_matrix(cm, classes,normalize=False,title="confusion_matrix", cmap=plt.cm.Blues):
	if normalize:
		cm = cm.astype("float")/cm.sum(axis=1)[:,np.newaxis]
		print("normaized Confusion matrix")
	else:
		print("Confusion matrix, without normaization: ")
	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()
	print(confusion_matrix(Ytest, yhat, labels=[1,0]))

# Compute confusion matrix
cnf_matrix = confusion_matrix(Ytest, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
print(plot_confussion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix'))
print (classification_report(Ytest, yhat))
from sklearn.metrics import log_loss
print(log_loss(Ytest, yhat_prob))