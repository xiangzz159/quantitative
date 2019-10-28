# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/11/22 9:08

@desc:

'''

from tools.stockstats import StockDataFrame
from tools import data2df
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

hist_ema, hist_signal_ma, hist_signal_ma_, ii, jj = 1, 4, 4, 3.0, -5.0
# hist_ema, hist_signal_ma, hist_signal_ma_, ii, jj = 1,3,7,4.0,-2.5
df = data2df.csv2df('BTC2017-09-01-now-4H.csv')
df = df.astype(float)
df['Timestamp'] = df['Timestamp'].astype(int)
stock = StockDataFrame.retype(df)
df['date'] = df['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
df['date'] = pd.to_datetime(df['date'])

hist_ema_name = 'hist_%d_ema' % hist_ema
hist_signal_ma_name = 'hist_signal_%d_ma' % hist_signal_ma
hist_signal_ma1_name = 'hist_signal_%d_ma_' % hist_signal_ma_
stock['macd']
stock[hist_ema_name]

df[hist_signal_ma_name] = np.round(df['hist_signal'].rolling(min_periods=1, window=hist_signal_ma).mean(), 2)
df[hist_signal_ma1_name] = np.round(df[hist_signal_ma_name].rolling(min_periods=1, window=hist_signal_ma_).mean(), 2)
df[hist_ema_name] = np.round(df[hist_ema_name], 2)

# DIF->macd  DEA->macds  MACD->macdh
df['signal'] = np.where(
    ((abs(df[hist_ema_name]) >= ii) & (df[hist_signal_ma_name] < jj)) & ((
            (df[hist_signal_ma_name] >= -1) & (df[hist_signal_ma_name].shift(1) < -1))), 'long', 'wait')
df['signal'] = np.where((df['hist'].shift(1) < 0) & (df['hist'] > 0), 'long', df['signal'])

df['signal'] = np.where(
    ((abs(df[hist_ema_name]) >= ii) & (df[hist_signal_ma_name] >= jj)) & ((
            (df[hist_signal_ma_name] <= 1) & (df[hist_signal_ma_name].shift(1) > 1))), 'short', df['signal'])
df['signal'] = np.where((df['hist'].shift(1) > 0) & (df['hist'] < 0), 'short', df['signal'])

df_signal = df.loc[df['signal'] != 'wait']
# 筛选出重复信号
unless_signal = df_signal.loc[
    ((df_signal['signal'] == 'short') & (df_signal['signal'].shift(1) == 'short')) | (
            (df_signal['signal'] == 'long') & (df_signal['signal'].shift(1) == 'long'))]
# 过滤重复信号
for index, row in unless_signal.iterrows():
    df.signal[index] = 'wait'
# 去掉前面100条指标不准数据
df = df[100:]
df_signal = df.loc[df['signal'] != 'wait']

df_long = df.loc[df['signal'] == 'long']
long_nparray = np.empty(shape=[0, 2])
for index, row in df_long.iterrows():
    long_nparray = np.append(long_nparray, [[index, row['close']]], axis=0)

df_short = df.loc[df['signal'] == 'short']
short_nparray = np.empty(shape=[0, 2])
for index, row in df_short.iterrows():
    short_nparray = np.append(short_nparray, [[index, row['close']]], axis=0)

plt.figure(figsize=(20, 10))
# plt.plot(range(df.shape[0]), df['close'])

plt.scatter(long_nparray[:, 0], long_nparray[:, 1], marker='^', c='g')
plt.scatter(short_nparray[:, 0], short_nparray[:, 1], marker='v', c='r')

plt.xticks(range(0, df.shape[0], 50), df['date'].loc[::50], rotation=45)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()
# plt.savefig('../data/buy_sell.png')
# plt.close()
