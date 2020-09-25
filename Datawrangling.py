import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as plt
path=r"C:\Users\Ssysuser\Desktop\New folder\Labs\Data Analysis Python\imports-85.data"
df = pd.read_csv(path)
#print(df.head(3))
#Add headers
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df.columns = headers
#Drop nan 
df.dropna(subset=["price"], axis=0)
#Replace "x" to Nan
df.replace("?",np.nan, inplace = True)
#Evaluation for missing data 
missing_data = df.isnull()
for column in missing_data.columns.values.tolist():
	print(column)
	print(missing_data[column].value_counts())
	print(" ")
Replace missing values
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
df["normalized-losses"].replace(np.nan,avg_norm_loss,inplace=True)
boreloss = df["bore"].astype("float").mean(axis=0)
df["bore"].replace(np.nan, boreloss,inplace=True)
strokeloss = df["stroke"].astype("float").mean(axis=0)
df["stroke"].replace(np.nan,strokeloss,inplace = True)

#Same, but using for
listt1 =["horsepower","stroke","peak-rpm","normalized-losses","bore"]
for listt in listt1:
	X = df[listt].astype("float").mean(axis=0)
	print("This is the avg of ",listt + ":", X)
	df[listt].replace(np.nan,X,inplace =True)
	print(df[listt].head(10))
#replacing num of doors by checking most common value using val_counts
print(df["num-of-doors"].value_counts())
df["num-of-doors"].replace(np.nan,"four",inplace=True)
print(df["num-of-doors"].value_counts())
print(df["num-of-doors"].value_counts().idxmax ()) # shows most used value
#Drop missing price rows
df.dropna(subset=["price"], axis = 0, inplace=True)
df.reset_index(drop=True, inplace=True)
#print(df.head(10))

#change data type to proper format 
print(df.dtypes)
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df["city-L/100km"]=235/df["city-mpg"]
df.drop(["city-mpg"],axis=1)
df.rename(columns={"city-mpg":"city-L/100km"}, inplace = True)
#normalize data xold/xnew
df["length"]=df["length"]/df["length"].max()
df["width"]=df["width"]/df["width"].max()
df["height"]=df["height"]/df["height"].max()

df["horsepower"]=df["horsepower"].astype(int, copy=True)
#Histogram of HP
#plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
#plt.pyplot.show()

#bins(Create categories for horsepower)
bins= np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
print(bins)
group_names = ["low","medium","high"]
df["horsepower-binned"]=pd.cut(df["horsepower"], bins, labels=group_names, include_lowest=True)
print(df[["horsepower","horsepower-binned"]].head(20))
print(df["horsepower-binned"].value_counts())


plt.pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
plt.pyplot.show()

# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
plt.pyplot.show()

#Create dummy variables, adding to df and droping old values 
dummy = pd.get_dummies(df["fuel-type"])
df = pd.concat([df, dummy], axis = 1)
dummy2=pd.get_dummies(df["aspiration"])
dummy2.rename(columns={"std":"aspiration-std","turbo":"aspiration-turbo"}, inplace=True)
df=pd.concat([df,dummy2],axis=1)
df.drop("aspiration",axis=1,inplace=True)

#Save csv file
df.to_csv("Clean_df.csv")

#Saving to specific path and with conditions 

def Rho (df): 
   if  (X1>65 and X2>65 and X3>65 and X4>65):
       df.to_csv(path='D:\My_Path\High.csv')
   elif (X1<55 and X2<55 and X3<55 and X4<55):
       df.to_csv(path='D:\My_Path\Low.csv')        
   else:
      print("Ignore")   