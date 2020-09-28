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

df = pd.read_csv(r"\Users\cecyt\OneDrive\Desktop\ML\New folder\Datasets\student-mat.csv")
dfp = pd.read_csv(r"\Users\cecyt\OneDrive\Desktop\ML\New folder\Datasets\student-por.csv")
DF = df.append(dfp)
percentageM = DF[DF.sex == "M"].shape[0]/DF.shape[0]
print("Male percentage: ",percentageM, "\nFemale percentage: ", 1-percentageM)
columns = list(df)

#Histograma de features
#for i in columns:
#	sns.countplot(x = i, data = DF);
#	plt.show()

#### Grades distributions by period####
# grades =  ("G1","G2","G3")
# for i in grades:
# 	sns.countplot(x = i, data = DF, color = "blue")
# 	plt.show()
columns = list(DF)
for i in columns:
	DF[i].astype("object")

DF = pd.get_dummies(DF)


def get_cvscores(data, model):
    y = DF["G3"].values
    X = DF.drop(["G1", "G2", "G3"], axis=1).values
    #Kfolds parte el dataset en Nsplits, acomoda en todas las combinaciones posibles, evalua y selecciona la mejor opción
    cv = KFold(n_splits=10, shuffle=True, random_state=0)
    print("Split: ", cv.split(X))
    scores = []
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))
    return scores, model

model = DecisionTreeClassifier(random_state = 0)
print(get_cvscores(DF,model))

#Segunda parte es probar nuestro modelo con diferentes parametros, para eso se una el GridSearchCV, corre el modelo con diferentes parametros y
#mide el accuracy que hubo al comparar su resultado con nuestro y_test
parameters = {'max_depth':[1,2,3,4,5,10,15,20], 'min_samples_leaf':[1,2,3,4,5,10,15,20], "min_weight_fraction_leaf": [0,0.3,0.5], "min_samples_split": [2, 20], "max_features": [1, 20]} 

model = DecisionTreeClassifier() 
clf = GridSearchCV(model, parameters, cv=5)

scores, model = get_cvscores(DF, model) 
print("\nScores: ",scores,"\nmodel:", model) 
#separamos nuestro DF en Target y features
y = DF["G3"].values
X = DF.drop(["G1", "G2", "G3"], axis=1).values 

clf = clf.fit(X,y)
best_params = clf.best_params_
print("Best params: ",best_params)

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
print("\n\nmeans: ",means)
model = DecisionTreeClassifier(max_depth = 4, max_features= 20, min_samples_leaf= 10, min_samples_split= 2, min_weight_fraction_leaf= 0 )
model.fit(X,y)
model.predict(X)
print(DF["G3"],y)
from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(model, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("students")
