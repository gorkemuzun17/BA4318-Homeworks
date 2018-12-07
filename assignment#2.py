import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm


df=pd.read_csv("Danish Kron.txt",sep='\t', lineterminator='\n')

#this function finds the zero values in the data frame and takes neighboring nonzero numbers.
#Afterwards, takes averages of this numbers and set averages instead of zeroes
def replacing_zeroes(df):
    zeroes_index=[]
    for i, j in enumerate(df.VALUE):
        if j ==0.0:
            zeroes_index.append(i)
    points=[]
    for index in zeroes_index:
        bttm=df.loc[index-1,"VALUE"]
        upp=df.loc[index+1,"VALUE"]
        sum=bttm+upp
        avg=sum/2
        newpoint=(index,avg)
        points.append(newpoint)
    current=0
    for item in points:
        df.loc[item[0],"VALUE"]=item[1]
    return df


df=replacing_zeroes(df)

size = len(df)
train = df[0:size-200]
test = df[size-200:]
head = df[0:5]
tail = df [size-5:]
print("Head")
print(head)

print("Tail")
print(tail)

df.DATE = pd.to_datetime(df.DATE,format="%Y-%m-%d")
df.index = df.DATE 
train.DATE = pd.to_datetime(train.DATE,format="%Y-%m-%d") 
train.index = train.DATE 
test.DATE = pd.to_datetime(train.DATE,format="%Y-%m-%d") 
test.index = test.DATE 

#Naive approach
print("Naive")
dd= np.asarray(train.VALUE)
y_hat = test.copy()
y_hat['naive'] = dd[len(dd)-1]
rms = sqrt(mean_squared_error(test.VALUE, y_hat.naive))
print("RMSE: ",rms)

#Simple average approach
print("Simple Average")
y_hat_avg = test.copy()
y_hat_avg['avg_forecast'] = train['VALUE'].mean()
rms = sqrt(mean_squared_error(test.VALUE, y_hat_avg.avg_forecast))
print("RMSE: ",rms)

#Moving average approach
print("Moving Average")
windowsize = 60
y_hat_avg = test.copy()
y_hat_avg['moving_avg_forecast'] = train['VALUE'].rolling(windowsize).mean().iloc[-1]
rms = sqrt(mean_squared_error(test.VALUE, y_hat_avg.moving_avg_forecast))
print("RMSE: ",rms)

# Simple Exponential Smoothing
print("Simple Exponential Smoothing")
y_hat_avg = test.copy()
alpha = 0.0
fit2 = SimpleExpSmoothing(np.asarray(train['VALUE'])).fit(smoothing_level=alpha,optimized=False)
y_hat_avg['SES'] = fit2.forecast(len(test))
rms = sqrt(mean_squared_error(test.VALUE, y_hat_avg.SES))
print("RMSE: ",rms)

# Holt
print("Holt")
sm.tsa.seasonal_decompose(train.VALUE).plot()
result = sm.tsa.stattools.adfuller(train.VALUE)
# plt.show()
y_hat_avg = test.copy()
alpha = 0.03
fit1 = Holt(np.asarray(train['VALUE'])).fit(smoothing_level = alpha,smoothing_slope = 0.1)
y_hat_avg['Holt_linear'] = fit1.forecast(len(test))
rms = sqrt(mean_squared_error(test.VALUE, y_hat_avg.Holt_linear))
print("RMSE: ",rms)

# Holt-Winters
print("Holt-Winters")
y_hat_avg = test.copy()
seasons = 10
fit1 = ExponentialSmoothing(np.asarray(train['VALUE']) ,seasonal_periods=seasons ,trend='add', seasonal='add',).fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
rms = sqrt(mean_squared_error(test.VALUE, y_hat_avg.Holt_Winter))
print("RMSE: ",rms)

# Seasonal ARIMA
# This is a naive use of the technique. See - http://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/
#print("Seasonal ARIMA")
#y_hat_avg = test.copy()
#fit1 = sm.tsa.statespace.SARIMAX(train.VALUE, order=(1, 0, 0),seasonal_order=(0,1,1,1)).fit()
#y_hat_avg['SARIMA'] = fit1.predict(start="2008-12-01", end="2018-11-30", dynamic=True)
#rms = sqrt(mean_squared_error(test.VALUE, y_hat_avg.SARIMA))
#print("RMSE: ",rms)
