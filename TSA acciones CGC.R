#Como generar la serie
  #AR.1<-arima.sim(model=list(order=c(1,0,0),ar=.2),n=100)


setwd("C:/Users/admin/OneDrive - uanl.edu.mx/Documents/SEPTIMO SEMESTRE/ESTADISTICA APLICADA")
#leer el documento 
DAR.1<-read.table("Tasas.csv",sep=",",header=T)
dim(DAR.1)
AR.1<-ts(DAR.1$Tasa,start=c(1),frequency=1)
AR.1

#graficar
plot(AR.1,main="Serie de Tiempo de Porcentaje de Beneficio HCITY.MX",ylab=("Porcentaje de Beneficio HCITY.MX"),ylim=c(-3,3),xlab="Dias")
abline(h=-0.0,col="red")

#clasificar

#Tendencia
Tar.1<-lm(AR.1~time(AR.1))
summary(Tar.1)

#Varianza
library(tseries)
adf.test(AR.1)

#La serie de tiempo es estacionaria

#correlogramas
acf(AR.1,main=("Autocorrelograma"))
pacf(AR.1,main=("Autocorrelograma Parcial"))
  #Podria ser ARMA

#Buscando el mejor modelo
m.aic<-Inf
for (i in 0:15) {
  
  for (j in 0:15) {
    a.aic<-AIC(arima(AR.1,order = c(i,0,j),method ="ML"))
    if(a.aic<m.aic){
      orden.mejor.arma<-c(i,0,j)
      m.aic<-a.aic
    }
    
  }
}
orden.mejor.arma
m.aic

#Revisando Ruido Blanco
res<-residuals(arima(AR.1,order = c(1,0,0),method ="ML"))

#Media cero
mean(res)
var(res)
mean(res)/var(res)

#Varianza cte
plot(res,type="p",main="Grafico de dispersión (Residuales)")

#Incorelación
acf(res,main=("Autocorrelograma Residuos"))
pacf(res,main=("Autocorrelograma Parcial Residuos"))
Box.test(res)  
  #No rechazo Ho por lo tanto, son independientes

#Normalidad
shapiro.test(res) 
  #No rechazo ho es decir hay normalidad de los datos


#Predicciones
ele1<-predict(arima(AR.1,order = c(1,0,0),method ="ML"),n.ahead = 7)$pre
plot(AR.1,xlim=c(0,130),ylab="Porcentaje de Beneficio",xlab="Días",main="Predicciones Hoteles City a 7 días")
legend(x="topright",legend = c("ST Porcentaje de Beneficio","Predicciones"),col = c("Black","blue"),cex=0.55,lty=c(1,1))
lines(ele1,col="blue")
ele1

#Acercando grafica
plot(AR.1,xlim=c(0,130),ylim=c(-0.2,0.2),ylab="Porcentaje de Beneficio",xlab="Días",main="Predicciones Hoteles City a 7 días")
legend(x="topright",legend = c("ST Porcentaje de Beneficio","Predicciones"),col = c("Black","Blue"),cex=0.5,lty=c(1,1))
lines(ele1,col="blue")
