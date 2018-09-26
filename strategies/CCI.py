# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/9/25 8:37

@desc:

'''

import pandas as pd
import numpy as np
from tools.stockstats import StockDataFrame
import time
import matplotlib.ticker as ticker
from tools import data2df
import matplotlib.pyplot as plt

# 本金
principal = 10000.0
title = 'BTC2017-09-01-now-1D'
df = data2df.csv2df(title + '.csv')
df = df.astype(float)
df['Timestamp'] = df['Timestamp'].astype(int)
stock = StockDataFrame.retype(df)

df['date'] = df['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
df['date'] = pd.to_datetime(df['date'])

stock['cci']
# 去掉前几个cci指标不准数据
df = df[5:]

df['regime'] = np.where((df['cci'] >= 100) & (df['cci'].shift(1) < 100), 1, 0)
df['regime'] = np.where((df['cci'] <= 100) & (df['cci'].shift(1) > 100), -1, df['regime'])

df['regime'] = np.where((df['cci'] <= -100) & (df['cci'].shift(1) >= -100), -1, df['regime'])
df['regime'] = np.where((df['cci'] >= -100) & (df['cci'].shift(1) <= -100), 1, df['regime'])

df_regime = df.loc[(df['regime'] == 1) | (df['regime'] == -1)]

# 数据清理
l = []
for index, row in df_regime.iterrows():
    if row['regime'] == 1 and row['cci'] >= 100:
        l.append([row['date'], row['close'], row['cci'], row['regime'], 'long', 'long'])
    elif row['regime'] == -1 and (row['cci'] <= 100 and row['cci'] > -100):
        l.append([row['date'], row['close'], row['cci'], row['regime'], 'close_long', 'long'])
    elif row['regime'] == -1 and row['cci'] <= -100:
        l.append([row['date'], row['close'], row['cci'], row['regime'], 'short', 'short'])
    elif row['regime'] == 1 and (row['cci'] >= -100 and row['cci'] < 100):
        l.append([row['date'], row['close'], row['cci'], row['regime'], 'close_short', 'short'])

df_signals = pd.DataFrame(l, columns=['date', 'price', 'cci', 'regime', 'signal', 'type'])


# 做多
long_df_signals = df_signals.loc[df_signals['type'] == 'long']
# 过滤第一行为平多，最后一行为做多数据
if (long_df_signals[:1]['signal'] == 'close_long').bool():
    long_df_signals = long_df_signals[1:]
if (long_df_signals[-1:]['signal'] == 'long').bool():
    long_df_signals = long_df_signals[:len(long_df_signals) - 1]

# 过滤两个做多/平多并列行
long_df_useless = long_df_signals.loc[
    ((long_df_signals['signal'] == 'long') & (long_df_signals['signal'].shift(-1) == 'long')) | (
            (long_df_signals['signal'] == 'close_long') & (long_df_signals['signal'].shift(1) == 'close_long'))]
for index, row in long_df_useless.iterrows():
    long_df_signals = long_df_signals.drop([index])

# 做空
short_df_signals = df_signals.loc[df_signals['type'] == 'short']
# 过滤第一行为平空，最后一行为做空数据
if (short_df_signals[:1]['signal'] == 'close_short').bool():
    short_df_signals = short_df_signals[1:]
if (short_df_signals[-1:]['signal'] == 'short').bool():
    short_df_signals = short_df_signals[:len(short_df_signals) - 1]

# 过滤两个做空/平空并列行
short_df_useless = short_df_signals.loc[
    ((short_df_signals['signal'] == 'short') & (short_df_signals['signal'].shift(-1) == 'short')) | (
            (short_df_signals['signal'] == 'close_short') & (short_df_signals['signal'].shift(1) == 'close_short'))]
for index, row in short_df_useless.iterrows():
    short_df_signals = short_df_signals.drop([index])


df_signals = pd.concat([long_df_signals, short_df_signals])
df_signals = df_signals.sort_index()

l = []
for i in range(1, len(df_signals), 2):
    row_ = df_signals.iloc[i - 1]
    row = df_signals.iloc[i]
    if row['type'] == 'long' and row_['type'] == 'long':
        principal = principal * row['price'] / row_['price']
        l.append([row['date'], row['price'], principal, 'long', 1])
    elif row['type'] == 'short' and row_['type'] == 'short':
        principal = principal * row_['price'] / row['price']
        l.append([row['date'], row['price'], principal, 'short', -1])
profits = pd.DataFrame(l, columns=['date', 'price', 'principal', 'type', 's'])
print(profits)

ax = profits[['price', 'principal']].plot(figsize=(20, 10), grid=True, xticks=profits.index, rot=90, subplots=True, style='b')

# 设置x轴刻度数量
ax[0].xaxis.set_major_locator(ticker.MaxNLocator(40))
# 以date为x轴刻度
ax[0].set_xticklabels(profits.date)
# 美观x轴刻度
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title(title)
plt.savefig('../data/' + title + '.png')
# plt.show()
