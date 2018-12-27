# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/11/28 11:10

@desc:

'''

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tools import data2df
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

df = data2df.csv2df("BTC2017-09-01-now-4H.csv")
df = df.astype(float)
df['Timestamp'] = df['Timestamp'].astype(int)
df['Date'] = df['Timestamp'].apply(
    lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
df['Date'] = pd.to_datetime(df['Date'])

training_set = df.iloc[:, 4:5].values

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

x_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    x_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

regressor = Sequential()

regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
regressor.add(Dropout(0, 2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0, 2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0, 2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0, 2))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(x_train, y_train, epochs=100, batch_size=32)

dataset_total = df['Close']
inputs = dataset_total[len(dataset_total) - 60 - 40:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
x_test = []
for i in range(60, len(inputs)):
    x_test.append(inputs[i - 60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_price = regressor.predict(x_test)
predicted_price = sc.inverse_transform(predicted_price)

plt.plot(predicted_price, color='green', label='Predicted BitMEX BTC Close Price')
plt.title('Predicted BitMEX BTC Close Price')
plt.xlabel('Time')
plt.ylabel('BitMEX Close Price')
plt.legend()
plt.show()
