import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn import metrics
from sklearn import tree
import graphviz
from sklearn.metrics import accuracy_score, precision_score, mean_squared_error, jaccard_similarity_score

df = pd.read_csv(r"\Users\cecyt\OneDrive\Desktop\ML\New folder\Datasets\student-mat.csv")
dfp = pd.read_csv(r"\Users\cecyt\OneDrive\Desktop\ML\New folder\Datasets\student-por.csv")
DF = df.append(dfp)
percentageM = DF[DF.sex == "M"].shape[0]/DF.shape[0]
print("Male percentage: ",percentageM, "\nFemale percentage: ", 1-percentageM)
columns = list(df)

#Histograma de features
for i in columns:
	sns.countplot(x = i, data = DF);
	plt.show()

#### Grades distributions by period####
grades =  ("G1","G2","G3")
for i in grades:
	sns.countplot(x = i, data = DF, color = "blue")
	plt.show()
columns = list(DF)
for i in columns:
	DF[i].astype("object")
print(DF.head())
DF = pd.get_dummies(DF)
print(DF.head())
y = DF["G3"].values
x = DF.drop(["G1", "G2", "G3"], axis=1).values

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size= .25,random_state =3)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

Grade_tree = DecisionTreeClassifier(criterion = "gini", max_depth=4, max_features=20,min_samples_leaf= 3,min_samples_split= 20,min_weight_fraction_leaf= 0)
Grade_tree.fit(x_train,y_train)
predgrades = Grade_tree.predict(x_test)
print("\n\nPredicted grades: ", predgrades[5:35],"\n\nActual grades:    ",y_test[5:35])
plt.figure(figsize= (10,10))
Grade_tree = tree.plot_tree(Grade_tree.fit(x_train,y_train),fontsize = 4)
plt.show()
Acc = metrics.accuracy_score(y_test,predgrades)
prec = precision_score(y_test,predgrades, average = "micro")
mse = mean_squared_error(y_test,predgrades)
jacc= jaccard_similarity_score(y_test,predgrades)
print("\nAccuracy score: ", Acc,"\nPrecision:  ",prec,"\nMean squared error: ",mse,"\nJaccard index: ",jacc)

