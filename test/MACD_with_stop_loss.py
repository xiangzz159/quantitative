# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/11/22 13:35

@desc:

'''

from tools.stockstats import StockDataFrame
from tools import data2df
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

money = 10000
hist_ema, hist_signal_ma, hist_signal_ma_, ii, jj = 2.0,5.0,3.0,2.0,1.0
# hist_ema, hist_signal_ma, hist_signal_ma_, ii, jj = 1,3,7,4.0,5
df = data2df.csv2df('BTC2017-09-01-now-2H.csv')
df = df.astype(float)
df['Timestamp'] = df['Timestamp'].astype(int)
stock = StockDataFrame.retype(df)
df['date'] = df['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
df['date'] = pd.to_datetime(df['date'])

hist_ema_name = 'hist_%d_ema' % int(hist_ema)
hist_signal_ma_name = 'hist_signal_%d_ma' % int(hist_signal_ma)
hist_signal_ma1_name = 'hist_signal_%d_ma_' % int(hist_signal_ma_)
stock['macd']
stock[hist_ema_name]

df[hist_signal_ma_name] = np.round(df['hist_signal'].rolling(min_periods=1, window=int(hist_signal_ma)).mean(), 2)
df[hist_signal_ma1_name] = np.round(df[hist_signal_ma_name].rolling(min_periods=1, window=int(hist_signal_ma_)).mean(), 2)
df[hist_ema_name] = np.round(df[hist_ema_name], 2)

# DIF->macd  DEA->macds  MACD->macdh
df['signal'] = np.where(
    ((abs(df[hist_ema_name]) >= ii) & (df[hist_signal_ma1_name] < jj)) & ((
            (df[hist_signal_ma_name] >= -1) & (df[hist_signal_ma_name].shift(1) < -1))), 'long', 'wait')
df['signal'] = np.where((df['hist'].shift(1) < 0) & (df['hist'] > 0), 'long', df['signal'])

df['signal'] = np.where(
    ((abs(df[hist_ema_name]) >= ii) & (df[hist_signal_ma1_name] >= jj)) & ((
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

side = 'wait'
signal_close = 0
stop_price = 0
l = []
for i in range(len(df)):
    row = df.iloc[i]
    high_price = max(df[i - 6:i].high) if i > 6 else row['high']
    low_price = max(df[i - 6:i].low) if i > 6 else row['low']
    now_money = money
    side_num = 1 if side == 'long' else -1
    if row['signal'] == 'wait':
        if signal_close > 0:
            now_money = (row['close'] / signal_close) ** side_num * money

    elif row['signal'] == 'long':
        if side == 'wait':
            side = row['signal']
            signal_close = row['close']
            stop_price = low_price
        elif side == 'short':
            money = (row['close'] / signal_close) ** side_num * money
            now_money = money
            side = row['signal']
            signal_close = row['close']
            stop_price = low_price
        else:
            now_money = (row['close'] / signal_close) ** side_num * money
    else:
        if side == 'wait':
            side = row['signal']
            signal_close = row['close']
            stop_price = high_price
        elif side == 'long':
            money = (row['close'] / signal_close) ** side_num * money
            now_money = money
            side = row['signal']
            signal_close = row['close']
            stop_price = high_price
        else:
            now_money = (row['close'] / signal_close) ** side_num * money

    l.append([row['date'], round(row['close'], 2), row['signal'], round(now_money, 2)])

df_ = pd.DataFrame(l, columns=['date', 'close', 'signal', 'money'])

df_.to_csv('../data/result.csv', index=None)
df_.loc[df_['signal'] != 'wait'].to_csv('../data/result1.csv', index=None)

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(20, 10))
plt.plot(range(df_.shape[0]), df_['close'], label="价格")
plt.plot(range(df_.shape[0]), df_['money'], label="收益")
plt.xticks(range(0, df_.shape[0], 100), df_['date'].loc[::100], rotation=45)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.legend()
plt.title('不加止损收益')
plt.show()
# plt.savefig('../data/macd_no_stop_loss.png')
# plt.close()
