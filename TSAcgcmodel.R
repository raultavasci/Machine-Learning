library(tseries)
library(TSA)
setwd("C:/Users/cecyt/OneDrive/Desktop/estadística aplicada")
cgc<-read.csv("CGC.csv",sep = ",")
cgc
cgcts1<-ts(cgc$Adj.Close)
plot(cgcts1,main="Acciones de la empresa CGC", type="l", ylab="Precio de la acción en dlls.", xlab = "Tiempo en días.")

start(cgcts1)
end(cgcts1)
cgcts<-cgcts1[180:240]
plot(cgcts,main="Acciones de la empresa CGC", type="l", ylab="Precio de la acción en dlls.", xlab = "Tiempo en días.")
cgcts
#tendencia
tlin<-lm(cgcts~time(cgcts))
summary(tlin)
abline(tlin,col="orange1")
#cuadrática
t<-as.numeric(time(cgcts))
tcuad<-lm(cgcts~poly(t,degree=2))
summary(tcuad)
lines(t,fitted.values(tcuad),col="magenta",lwd=2)
#logarítmica 
xlog<-log(time(cgcts))
tlog<-lm(cgcts~xlog)
summary(tlog)
legend(x="topright", inset=0.01,title="Modelos",legend=c("Lineal","Cuadrático","Logaritmico"),lty=c(1,1,1),col=c("orange1","magenta","blue"),cex=0.7) 


tlog$coefficients
tenlog<-tlog$coefficients[1]+tlog$coefficients[2]*xlog
lines(tenlog,col="blue") 
adf.test(cgcts,alternative = "stationary")
#p-val = 0.2893 > 0.05 No es estacionaria

#corrección respecto a media y varianza 
c1<-diff(cgcts)
plot(c1,type="l", main="Primera Corrección" ,ylab="Precio con dif", xlab="Tiempo")
#revisamos la corrección 
#media 
summary(lm(c1~time(c1))) 
adf.test(c1,alternative = "stationary")
acf(c1,main="Autocorrelograma de la corrección")
pacf(c1,main="Autocorrelograma Parcial de la corrección")
#ENCONTRANDO EL MEJOR MODELO 
#AR
ar(c1)
AIC(arima(c1,order = c(4,0,0),method = "ML")) #271.7553

#MA
raul<-Inf
for(i in 1:15)
{
  v.aic<-AIC(arima(c1,order = c(0,0,i),method = "ML"))
  if(v.aic<raul)
  {
    raul<-v.aic
    MA.mejor<-i
  }
}
MA.mejor #el orden del mejor MA es 4
raul#AIC del mejor MA es 270.9876

#ARMA
arma.aic<-Inf
for(j in 1:15)
{
  for(i in 1:15)
  {
    aux<-AIC(arima(c1,order = c(j,0,i),method = "ML"))
    if(aux<arma.aic)
    {
      arma.aic<-aux
      ARMA.mejor<-c(j,i)
    }
  }
}
ARMA.mejor #orden del mejor arma (2,2)
arma.aic #AIC del mejor ARMA 270.13

#Predicciones para 10 meses 
AR.p<-predict(arima(c1,order=c(4,0,0),method ="ML"),n.ahead=20)$pred
MA.p<-predict(arima(c1,order=c(0,0,4),method ="ML"),n.ahead=20)$pred
ARMA.p<-predict(arima(c1,order=c(2,0,2),method ="ML"),n.ahead=20)$pred

plot(c1,type="l",xlim=c(0,80),main="Predicciones de los modelos AR, MA y ARMA")
lines(AR.p,col="blue",lwd=2)
lines(MA.p,col="green",lwd=2)
lines(ARMA.p,col="magenta",lwd=2)
legend(x="topleft",legend=c("AR(0)", "MA(1)","ARMA(3,1)"),lty=c(1,1,1),col =c("blue","green","magenta"),lwd=2,cex=.90)



#Ruido Blanco en residuales 
res.22<-residuals(arima(c1,order = c(3,0,1),method="ML"))
#media 
mean(res.22) #queremos que el valor este entre -1 y 1 y se cumple con un valor de -0.01261728
#varianza 
plot(res.22,type="p", main="Varianza de Residuales",col="blue", ylab = "Residuales",pch = 3) #no muestra patrón 
#incorrelacion
acf(res.22, main = "Autocorrelación de residuos")
pacf(res.22,main = "Autocorrelación parcial de residuos")
#incorrelación forma analitica 
Box.test(res.22) # por el p-value de 0.7349<0.05  no rechazo H0,  por lo tanto los residuales son independientes

#Debido a que cumple los 3 supuestos referentes de media, varianza y son incorrelacionados decimos que los valores residuales siguen un proceso de ruido blanco. 

#Prueba de normalidad 
shapiro.test(res.22) # p-value = 2.161e-05 <.05 se cumple,por lo tanto decimos que los residuales no provienen distribución normal 

#predicciones para c1 
p22<-predict(arima(c1,order = c(2,0,2),method="ML"),n.ahead=10)$pred
plot(c1,type="l",xlim=c(0,80),main="Corrección 1 y predicciones",xlab="Tiempo en días", ylab=c("¨Precio acción"))
lines(p22,col="magenta",lwd=2) 
legend(x="topleft",legend=c("Serie de tiempo", "Predicción 10 días"),lty=c(1,1),col =c("black","magenta"),lwd=2,cex=.75)

#predicciones datos originales 
p212<-predict(arima(cgcts,order = c(2,1,2),method="ML"),n.ahead=10)$pred
cgcts
plot(cgcts,type="l",main="Original y predicciones",xlim=c(0,80),xlab="Tiempo en días", ylab=c("¨Precio acción"))
lines(p212,col="magenta",lwd=2) 
lines(cgcts1[180:250],xlim=c(60,80),col="blue",lwd="2")
lines(cgcts,col="black",lwd=2)
legend(x="topright", inset=0.01,legend=c("Serie original","Datos reales","Predicciones ARIMA(2,1,2)"),lty=c(1,1,1),col=c("black","blue","magenta"),cex=0.7) 
cgcts1[241:250]
p212

