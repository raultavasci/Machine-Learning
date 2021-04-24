import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv(r"\Users\cecyt\OneDrive\Desktop\books.csv")
#print(df.head(5))

# genero = df.loc[df["Genero"]== "Fiction"]
# print(genero)
newdf = df.loc[df["Reviews"]>3000]
autor = newdf[["Autor","Rating","Reviews"]]
reviews = autor.groupby(['Autor'],sort=False).mean()
top10reviews = reviews.sort_values(by=["Reviews"],ascending=False).head(10)
top10reviews = top10reviews.drop(["Rating"],axis = 1) 
print("\nAutores con más reviews en promedio.\n",top10reviews)

print("Generos dentro de los 50 bestsellers\n", newdf["Genero"].value_counts())
fiction = newdf.loc[newdf["Genero"]=="Fiction"].count()[0]
nonfiction = newdf.loc[newdf["Genero"]=="Non Fiction"].count()[0]
labels = ["Fiction","Non Fiction"]
#print("\nNADA\n",fiction)
z = newdf["Autor"].value_counts().rename_axis('Autor').reset_index(name="Number")
#print(z)
dosmas = z["Autor"].to_list()
print(dosmas)
newdf.loc[newdf["Titulo"] == "Old School (Diary of a Wimpy Kid #10)",["Titulo","FechaLanzamiento"]] = ["Old School (Diary of a Wimpy Kid #10)","2015-11-03"]
# print(dosmas)
# print(newdf.sort_values(by=["Autor"],ascending=False))
prevbooks = []
A = []
###df2 = pd.DataFrame(columns=columns)

for x in dosmas:

	aut = newdf.loc[newdf["Autor"] == x].sort_values(by=["FechaLanzamiento"])
	
	for autor in range(len(aut)):
		
		A = autor
		prevbooks.append(A)

print(prevbooks)
new_df = pd.DataFrame()
for auth,num in newdf["Autor"].value_counts().items():
	Q = newdf.loc[newdf["Autor"]==auth].sort_values(by=["FechaLanzamiento"])
	new_df = pd.concat([new_df,Q],axis=0)
new_df["NumLib"] = prevbooks

 
print(new_df.head(20))
#print(aut)
#print(newdf["Autor"].value_counts())
# plt.pie([fiction, nonfiction],labels = labels,autopct="%.2f %%")
# plt.show()
#print(newdf)
# x = newdf.FechaLanzamiento.to_list()

# print("\n",x)
date_format = "%Y-%m-%d"

X=[]
for index, row in new_df.iterrows():
	a = datetime.strptime(row["FechaLanzamiento"], date_format)
	b = datetime.strptime('2021-02-16',date_format)
	delta = b - a
	X.append(delta.days)
new_df["DiasTransc"] = X
print(new_df.head(20))


from sklearn.preprocessing import LabelEncoder

new_df =new_df.copy(deep = True)
LE = LabelEncoder()
new_df["Genero"] = LE.fit_transform(new_df["Genero"])
new_df["Autor"] = LE.fit_transform(new_df["Autor"])
new_df["Titulo"] = LE.fit_transform(new_df["Titulo"])
print("\n\n\n",new_df)

y = new_df["Rating"]
X = new_df.drop(["Rating","FechaLanzamiento"], axis=1)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=3,shuffle=True)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape,X_train.head())

LR=LinearRegression()
LR.fit(X_train,y_train)
Prediction=LR.predict(X_test)
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test,Prediction)))
print(LR.score(X_test,y_test))