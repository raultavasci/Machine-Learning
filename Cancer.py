import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import pylab as pl
import scipy.optimize as opt
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import itertools
df = pd.read_csv(r"\Users\Ssysuser\Desktop\New folder\Labs\ML\Svm\cell_samples.csv")
print (df[["Class","UnifSize"]].head(20))

print("Value counts: \n", df["UnifSize"].value_counts())

#Creates plot
ax = df[df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
df[df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);

print(df.dtypes)
#Check the logic of thenext 2 lines****************************************************************************
df=df[pd.to_numeric(df["BareNuc"], errors = "coerce").notnull()]
df["BareNuc"] = df["BareNuc"].astype(int)
print(df["BareNuc"].dtypes)
feature_df = df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
print(X[0:5])
df["Class"] = df["Class"].astype(int)
y = np.asarray(df["Class"])
print(y[0:5])
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
print("\nXtrain: ",x_train.shape,"\nxtest.",x_test.shape,"\nytest: ",y_test.shape,"\nytrain: ",y_train.shape)
clf = svm.SVC(kernel="rbf", gamma="auto")
clf.fit(x_train,y_train)
yhat = clf.predict(x_test)
print(yhat[0:5])
#Evaluation
def plot_confusion_matrix(cm, classes, normalize = False, title="Confusion Matrix", cmap = plt.cm.Blues):
	if normalize:
		cm= cm.astype("float")/cm.sum(axis=1)[:,np.newaxis]
		print("normalized CF")
	else:
		print("CF without normalization")
	print(cm)
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
	    plt.text(j, i, format(cm[i, j], fmt),
	                 horizontalalignment="center",
	                 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')
from sklearn.metrics import f1_score
f1 = f1_score(y_test,yhat, average ="weighted")
from sklearn.metrics import jaccard_similarity_score
jacc= jaccard_similarity_score(y_test,yhat)
print("f1 Score: ",f1,"Jacccard: ",jacc)