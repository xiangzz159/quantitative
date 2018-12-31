# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/12/29 18:01

@desc: 波动率因子

'''

import pandas as pd
import numpy as np
from tools.stockstats import StockDataFrame
import time
import matplotlib.ticker as ticker
from tools import data2df
import matplotlib.pyplot as plt
from tools import wave_guess

# filename = 'BTC2017-09-01-now-4H'
filename = 'BTC2015-02-19-now-4H'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)
df['Timestamp'] = df['Timestamp'].astype(int)
stock = StockDataFrame.retype(df)

late_cycles = 18
mean_cycles = 3

df['date'] = df['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
df['date'] = pd.to_datetime(df['date'])

df['volatility_rate'] = (df['close'] - df['close'].shift(late_cycles)) / df['close'].shift(late_cycles)
df['volatility_mean'] = df['volatility_rate'].rolling(window=late_cycles).mean()
df['volatility_mean'] = df['volatility_mean'].rolling(window=mean_cycles).mean()
df['volatility_std'] = df['volatility_rate'].rolling(window=late_cycles).std()
df['volatility_abs'] = abs(df['volatility_rate'])

df = df[50:]

df['signal'] = np.where((df['volatility_rate'].shift(1) < 0) & (df['volatility_rate'] > 0), 'long', 'wait')
df['signal'] = np.where((df['volatility_rate'] > 0) & (df['volatility_mean'] > 0) & (
            df['volatility_rate'].shift(1) < df['volatility_mean'].shift(1)) & (
                                    df['volatility_rate'] > df['volatility_mean']), 'long', df['signal'])
df['signal'] = np.where((df['volatility_rate'].shift(1) > 0) & (df['volatility_rate'] < 0), 'short', df['signal'])
df['signal'] = np.where((df['volatility_rate'] < 0) & (df['volatility_mean'] < 0) & (
            df['volatility_rate'].shift(1) > df['volatility_mean'].shift(1)) & (
                                    df['volatility_rate'] < df['volatility_mean']), 'short', df['signal'])
df['signal'] = np.where((df['signal'] == 'wait') & (df['volatility_rate'].shift(1) > df['volatility_mean'].shift(1)) & (
            df['volatility_rate'] < df['volatility_mean']), 'close_long', df['signal'])
df['signal'] = np.where((df['signal'] == 'wait') & (df['volatility_rate'].shift(1) < df['volatility_mean'].shift(1)) & (
            df['volatility_rate'] > df['volatility_mean']), 'close_short', df['signal'])

df[['close']] = round(df[['close']], 1)
df[['volatility_rate', 'volatility_mean', 'volatility_std']] = round(
    df[['volatility_rate', 'volatility_mean', 'volatility_std']], 5)
df_ = df[['date', 'close', 'volatility_rate', 'volatility_mean', 'volatility_std', 'signal']].loc[
    df['signal'] != 'wait']
df_['signal'] = np.where(df_['signal'].shift(1) == df_['signal'], 'wait', df_['signal'])
if (df_[:1]['signal'] == 'close_long').bool() or (df_[:1]['signal'] == 'close_short').bool():
    df_ = df_[1:]
if (df_[-1:]['signal'] == 'long').bool() or (df_[-1:]['signal'] == 'short').bool():
    df_ = df_[:len(df_) - 1]

# df_.loc[df_['signal'] != 'wait'].to_csv('../data/volatility_rate.csv', index=False)
l = []
money = 10000
for i in range(1, len(df_), 2):
    row_ = df_.iloc[i - 1]
    row = df_.iloc[i]
    if row['signal'] == 'close_long' and row_['signal'] == 'long':
        money = money * row['close'] / row_['close']
        l.append([row['date'], row['close'], money, 'long', 1])
    elif row['signal'] == 'close_short' and row_['signal'] == 'short':
        money = money * row_['close'] / row['close']
        l.append([row['date'], row['close'], money, 'short', -1])
    elif row['signal'] == 'short' and row_['signal'] == 'long':
        money = money * row['close'] / row_['close']
        l.append([row['date'], row['close'], money, 'long', 1])
        i -= 1
    elif row['signal'] == 'long' and row['signal'] == 'short':
        money = money * row_['close'] / row['close']
        l.append([row['date'], row['close'], money, 'short', -1])
        i -= 1

profits = pd.DataFrame(l, columns=['date', 'close', 'money', 'type', 's'])

ax = profits[['close', 'money']].plot(figsize=(20, 10), grid=True, xticks=profits.index, rot=90, subplots=True,
                                      style='b')

# 设置x轴刻度数量
ax[0].xaxis.set_major_locator(ticker.MaxNLocator(40))
# 以date为x轴刻度
ax[0].set_xticklabels(profits.date)
# 美观x轴刻度
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title(filename)
# plt.savefig('../data/' + filename + '.png')
plt.show()
