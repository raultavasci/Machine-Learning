import matplotlib.pyplot as plt 
import matplotlib as mpl
import pandas as pd 
import numpy as np 
import seaborn as sns
df_can = pd.read_excel(r"\Users\Ssysuser\Desktop\New folder\Labs\Visualization\Migration.xlsx")
print(df_can.columns.values)
print("\n\n\n\n", df_can.index.values)

print(type(df_can.columns.tolist()))
print(type(df_can.index.tolist()))

#check # of rows and columns 
print(df_can.shape)

#drop colums and rename them 
df_can.drop(["AREA","REG","DEV","Type","Coverage"],axis=1,inplace=True)
df_can.rename(columns={"OdName":"Country","AreaName":"Continent","RegName":"Region"}, inplace=True)
#Suma cada columna ---> y da el total
df_can["Total"]=df_can.sum(axis=1) 
print("Null data", df_can.isnull().sum())
print(df_can.describe())

#Set index, to reset: .reset_index() or df_can.index.name = None
df_can.set_index("Country", inplace=True)

#Immigrants from japan 
#1. The full row data (all columns)
print("Total immigrants from Japan: ", df_can.loc["Japan"])
# alternate methods
print(df_can.iloc[87])
print(df_can[df_can.index == 'Japan'].T.squeeze())

#2. For year 2013
print(df_can.loc["Japan",2013])

#3. For years 1980 to 1985 and adding total of immigrants during those years 
print(df_can.loc["Japan", [1980,1981,1982,1983,1984,1985]], "\nTotal : ", df_can.loc["Japan", [1980,1981,1982,1983,1984,1985]].sum(axis=0))
#Convert columns to str
df_can.columns = list(map(str, df_can.columns))
#create list of years 1980-2013 (range must be +1)
years = list(map(str, range(1980,2014)))
#create condition boolean series
conditon = df_can["Continent"] == "Asia"
print(conditon)

print(df_can[conditon])
print(df_can[(df_can["Continent"] == "Asia") & (df_can["Region"] =="Southern Asia")])
print('data dimensions:', df_can.shape)
print(df_can.columns)
print(df_can.head(2))
print(plt.style.available)
mpl.style.use(["ggplot"])
haiti = df_can.loc["Haiti",years]
haiti.index = haiti.index.map(int) # let's change the index values of Haiti to type integer for plotting
haiti.plot(kind='line')

plt.title('Immigration from Haiti')
plt.ylabel('Number of immigrants')
plt.xlabel('Years')
plt.text(2000, 6000, '2010 Earthquake')
plt.show()


#df have countries as axis, plot is not usefull, we transpose this V with transpose()
df_CI = df_can.loc[["India","China"], years]
print(df_CI.head())

df_CI.plot(kind="line")
plt.title("Immigration China and india")
plt.xlabel("Years")
plt.ylabel("# or immigrants")
plt.show()


#Correct format to plot 
df_CI = df_CI.transpose()
print(df_CI)
df_CI.plot(kind="line")
plt.title("Immigration China and india")
plt.xlabel("Years")
plt.ylabel("# or immigrants")
plt.show()

df_can.sort_values(by = "Total",ascending = False, inplace = True)
dftop5 = df_can.head(5)
print(dftop5)
dftop5 = dftop5[years].transpose()
print(dftop5.head())
dftop5.index = dftop5.index.map(int)
dftop5.plot(kind="line", figsize = (14,8))
plt.title("Trend immigrants top 5",size = 30)
plt.xlabel("years")
plt.ylabel("No. of immigrants")
plt.show()
dftop5.plot(kind ="area", alpha= 0.25, stacked = False,figsize=(20,10)) # alpha goes from 0 to 1, sets transparency (Default = 0.5)
plt.title("Immigration Trend of Top 5 Countries", size = 30)
plt.ylabel("Number of immigrants", size = 20)
plt.xlabel("Years", size = 20)
plt.show()

dfless = df_can.tail(5)
dfless = dfless[years].transpose()
dfless.index = dfless.index.map(int)
dfless.plot(kind= "area", alpha = 0.45, stacked = False, figsize = (20,10))
plt.title("Immigration Trend lowest 5 Countries", size = 30)
plt.ylabel("Number of immigrants", size = 20)
plt.xlabel("Years", size = 20)
plt.show()

#check 2013 data
print(df_can["2013"].head())
#np.histogram return 2 values
count, bin_edges = np.histogram(df_can["2013"])
print(count) #Frequency 
print(bin_edges)#Range (default = 10) intervalos 

df_can['2013'].plot(kind='hist', figsize=(8, 5))

plt.title('Histogram of Immigration from 195 Countries in 2013') # add a title to the histogram
plt.ylabel('Number of Countries') # add y-label
plt.xlabel('Number of Immigrants') # add x-label

plt.show()

# generate histogram
#Un-stacked hist
dfT = df_can.loc[["Denmark","Norway","Sweden"], years].transpose()
count, bin_edges = np.histogram(dfT, 15)
xmin = bin_edges[0] - 10   #  first bin value is 31.0, adding buffer of 10 for aesthetic purposes 
xmax = bin_edges[-1] + 10  #  last bin value is 308.0, adding buffer of 10 for aesthetic purposes
dfT.plot(kind="hist",alpha =.55, figsize=(10,6),bins = 15, xticks = bin_edges, color = ["coral","darkslateblue","mediumseagreen"],xlim=(xmin,xmax))
plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')
plt.savefig("plot.png",bbox_inches="tight")
plt.show()