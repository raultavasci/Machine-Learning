import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

df = pd.read_csv(r"\Users\Ssysuser\Desktop\New folder\Labs\ML\Decisiontrees\drug200.csv", delimiter = ",")
print(df.head(4))

X = df[["Age","Sex","BP","Cholesterol","Na_to_K"]]
print(X[0:5])
#.value converts the pd array to numpy array 
X = df[["Age","Sex","BP","Cholesterol","Na_to_K"]].values
print(X[0:5])

#This part get dummies for categorical variables *Need to check dif between this and get_dummies()
le_sex = preprocessing.LabelEncoder()
le_sex.fit(["F","M"])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

print(X[0:5])

y = df["Drug"]

#Setting up the decision tree
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=3)
print("************************************TRAINTESTSPLIT****************************************\n",X_train.shape,y_train.shape,X_test.shape,y_test.shape)
#Modeling the tree 
drugTree = DecisionTreeClassifier(criterion = "entropy", max_depth = 4)
print(drugTree)
#Then we fill the dara with the training feature matris X_train and training response vector y_train
drugTree.fit(X_train,y_train)
#Predicitons and store it into var
predTree = drugTree.predict(X_test)
print(predTree[0:5],"\n",y_test[0:5])
#Accuracy
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))
#Visualize the tree
print(X[0:5])
tree.plot_tree(drugTree.fit(X_train,y_train))
plt.show()
dot_data = StringIO()
filename = "drugtree.png"
featureNames = df.columns[0:5]
targetNames = df["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_train), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')