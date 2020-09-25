import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl

df = pd.read_csv(r"\Users\Ssysuser\Desktop\New folder\Labs\ML\Regression\FuelConsumptionCo2.csv")
print(df.head(4),df.describe(include="all"))
cdf=df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]
viz = cdf[["CYLINDERS","ENGINESIZE","CO2EMISSIONS","FUELCONSUMPTION_COMB"]]
viz.hist()
list1 = ["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB"]
for x in list1:
	plt.scatter(cdf[x], cdf.CO2EMISSIONS, color = 'blue')
	plt.xlabel(x)
	plt.ylabel("emission")
	plt.draw()
	plt.show()

#Creating train ande test dataset
msk = np.random.rand(len(df)) <.08
train = cdf[msk]
test = cdf[~msk]
print(train,test)


#modeling sing sklearn
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[["ENGINESIZE"]])
train_y = np.asanyarray(train[["CO2EMISSIONS"]])
regr.fit (train_x, train_y)

#Coefficients 
print("coef;", regr.coef_)
print("Intercept: ", regr.intercept_)

#Plot scatter and fit line 
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# #Using r2 
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error:%.2f"% np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y))

#Arange (Start, end, space in between)
x = np.arange(-5.0,5.0,0.1)
#print(x,"The lenght of x is:",len(x))
y = 2*(x)+3
y_noise = 2*np.random.normal(size=x.size) 
ydata = y + y_noise

plt.plot(x,ydata, "bo")
plt.plot(x,y,"r")
plt.ylabel("Dependent")
plt.xlabel("indep")
plt.show()


