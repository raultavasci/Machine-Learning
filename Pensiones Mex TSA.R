library(tseries)
library(TSA)

#Crear serie de tiempo
setwd("C:/Users/cecyt/OneDrive/Desktop/estadística aplicada/primer parcial")
imss<-read.csv("Datos 1 parcial Pensiones IMSS.csv",sep = ",",header = TRUE)
imss

class(imss) #data frame 
ip<-ts(imss$SF65005,start = c(2006,12), end = c(2009,12),frequency = 12 ) #convirtiendo a serie de tiempo 
ip
class(ip)
start(ip) #Inicia en Diciembre de 2006
end(ip) # Termina en Diciembre de 2009
plot(ip, main="Fondo de Retiro IMSS", ylab="Monto", xlab = "Tiempo (meses)")


#### Clasificando la serie ####

#Tendencia
t.lin <- lm(ip~time(ip))
summary( t.lin ) #Ajuste de 87.87%
abline(t.lin, col="midnightblue", lwd=2)
#p-value: < 0.00000000000000022 es menor al p-valor elegido de 0.05
#Presenta una tendencia lineal
#Regresión significativa 


#tendencia cuadrática
t<-as.numeric(time(ip))
cuad<-lm(ip~poly(t,degree=2))
summary(cuad) #Ajuste de 89.41%
lines(t,fitted.values(cuad),col="orange1",lwd=2)
#p-value: < 0.00000000000000022 es menor al p-valor elegido de 0.05


#tendencia exponencial
y.<-log(ip)
lineal.asoc<- lm(y.~time(ip))
summary(lineal.asoc) #Ajuste de 89.09%
#p-value: < 0.00000000000000022 es menor al p-valor elegido de 0.05
coefficients(lineal.asoc)[1] #Beta cero
#calculando y gorro del modelo exponencial
expon<-exp(coefficients(lineal.asoc)[1])*exp(coefficients(lineal.asoc)[2]*t)
lines(t,expon,col="red1",lwd=2)

legend(x="topleft",legend=c("Serie de tiempo original", "Tendencia Lineal","Tendencia Cuadrática","Tendencia Exponencial"),lty=c(1,1,1,1),col =c("black","midnightblue","orange1","red1"),lwd=2)

#El mejor modelo es el cuadrático, por ser el modelo con mayor ajuste.  

#Varianza constante
adf.test(ip,alternative = "stationary")

#No rechazo H0, ya que  0.9842<.05 .La varianza es NO estacionaria 

#CLASIFICACIÓN DE LA SERIE: NO ESTACIONARIA 

#corrección respecto a media y varianza 
ci1<-diff(log(ip))
plot(ci1, main="Primera Corrección" ,ylab="Diff. Log. del Fondo de retiro IMSS", xlab="Tiempo")
#revisamos la corrección 
#media 
summary(lm(ci1~time(ci1))) # p-value 0.724 <.05 no se cumple, por lo tanto la regresión no muestra tendencia. 
#No hay evidencia de asociación 


#varianza
adf.test(ci1,alternative = "stationary") # p-value 0.4296<.05 no se cumple por lo tanto NO rechazo H0, la serie es NO estacionaria en varianza

#clasificamos la serie corregida como NO ESTACIONARIA 

# SEGUNDA CORRECCIÓN 
correc2<-diff((ci1)) #corrigiendo la no estacionariedad de la primera correccion 
plot(correc2, main="Segunda Corrección",ylab="Diff. del Fondo de retiro IMSS" ,xlab="Tiempo")

#revisar que sea efectiva la corrección
#media 
summary(lm(correc2~time(correc2)))# p-value 0.9853 <.05 no se cumple, por lo tanto la regresión no muestra tendencia.


#varianza
adf.test(correc2,alternative = "stationary") #p-value menor a 0.01, lo cual cumple que sea menor a 0.05, rechazo H0, la serie es estacionaria en varianza. 
#serie estacionaria en varianza 

#Clasificación de la serie corregida dos veces: ESTACIONARIA 

#ENCONTRANDO EL MEJOR MODELO 
#AR
ar(correc2) #AR(3)
AIC(arima(correc2,order = c(3,0,0),method = "ML")) #-149.4832

#MA
karla<-Inf
for(i in 1:15)
{
  v.aic<-AIC(arima(correc2,order = c(0,0,i),method = "ML"))
  if(v.aic<karla)
  {
    karla<-v.aic
    MA.mejor<-i
  }
}
MA.mejor #el orden del mejor MA es 5
karla#AIC del mejor MA es -150.8216

#ARMA
arma.aic<-Inf
for(j in 1:15)
{
  for(i in 1:15)
  {
    aux<-AIC(arima(correc2,order = c(j,0,i),method = "ML"))
    if(aux<arma.aic)
    {
      arma.aic<-aux
      ARMA.mejor<-c(j,i)
    }
  }
}
ARMA.mejor #orden del mejor arma (1,6)
arma.aic #AIC del mejor ARMA -156.4578

#El mejor modelo es el ARMA 

#Predicciones para 10 meses 
AR.p<-predict(arima(correc2,order=c(3,0,0),method ="ML"),n.ahead=10)$pred
MA.p<-predict(arima(correc2,order=c(0,0,5),method ="ML"),n.ahead=10)$pred
ARMA.p<-predict(arima(correc2,order=c(1,0,6),method ="ML"),n.ahead=10)$pred

#graficamente 
plot(correc2,main="Predicciones comparando modelos", xlim= c(2007,2011))
lines(AR.p,col="royalblue",lwd=2)
lines(MA.p,col="green",lwd=2)
lines(ARMA.p,col="sienna1",lwd=2)
legend(x="topleft",legend=c("AR(3)", "MA(5)","ARMA(1,6)"),lty=c(1,1,1),col =c("royalblue","green","sienna1"),lwd=2,cex=.90)

#Gráficamente podemos observar que el mejor modelo es el ARMA (1,6)


#Ruido Blanco en residuales 
res.16<-residuals(arima(correc2,order = c(1,0,6),method="ML"))
#media 
mean(res.16) #queremos que el valor este entre -1 y 1 y se cumple con un valor de -0.000951744
#varianza 
plot(res.16,type="p", main="Varianza de Residuales",col="sienna1", ylab = "Residuales",pch = 19) #no muestra patrón 
#incorrelacion
acf(res.16, main = "Autocorrelación de residuos")
pacf(res.16,main = "Autocorrelación parcial de residuos")
#incorrelación forma analitica 
Box.test(res.16) # por el p-value de 0.8194<0.05  no rechazo H0,  por lo tanto los residuales son independientes

#Debido a que cumple los 3 supuestos referentes de media, varianza y son incorrelacionados decimos que los valores residuales siguen un proceso de ruido blanco. 

#Prueba de normalidad 
shapiro.test(res.16) # p-value = 0.4643 <.05 no se cumple,por lo tanto decimos que los residuales provienen  distribución normal 

#predicciones para correc 2 
p.16<-predict(arima(correc2,order = c(1,0,6),method="ML"),n.ahead=10)$pred
plot(correc2, xlim=c(2006,2012),main="Corrección 2 y predicciones",xlab="Tiempo en Años", ylab=c("Fondo de Retiro IMSS"))
lines(p.16,col="sienna1",lwd=2) 
legend(x="topleft",legend=c("Serie de tiempo", "Predicción para 10 meses"),lty=c(1,1),col =c("black","sienna1"),lwd=2,cex=.75)


#predicciones para la correc 1 
p.116<-predict(arima(ci1,order = c(1,1,6),method = "ML"),n.ahead = 10)$pred
plot(ci1,xlim=c(2006,2012),main="Corrección 1 y predicciones",xlab="Tiempo en Años", ylab=c("Fondo de Retiro IMSS"))
lines(p.116,col="sienna1",lwd=2)
legend(x="topleft",legend=c("Serie de tiempo", "Predicción para 10 meses"),lty=c(1,1),col =c("black","sienna1"),lwd=2,cex=.75)



#predicciones para Serie Original 
p.126<-predict(arima(ip,order = c(1,2,6),method="ML"),n.ahead=10)$pred
plot(ip,xlim=c(2006,2012), ylim=c(722693150,1266098694), main="Original y predicciones",xlab="Tiempo en Años", ylab=c("Fondo de Retiro IMSS"))
lines(p.126,col="sienna1",lwd=2)
legend(x="topleft",legend=c("Serie de tiempo", "Predicción para 10 meses"),lty=c(1,1),col =c("black","sienna1"),lwd=2,cex=.75)

#El mejor modelo ARIMA (1,2,6) 