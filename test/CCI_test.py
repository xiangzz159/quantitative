# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/9/25 8:37

@desc: CCI 策略回测

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
filename = 'BTC2018-01-01-now-4H'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)
df['Timestamp'] = df['Timestamp'].astype(int)
stock = StockDataFrame.retype(df)

df['date'] = df['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
df['date'] = pd.to_datetime(df['date'])

stock['cci']
stock['stoch_rsi']
# 去掉前几个cci指标不准数据
df = df[5:]

df['regime'] = np.where((df['cci'] >= 95) & (df['cci'].shift(1) < 95), 'long', 'wait')
df['regime'] = np.where(
    ((df['cci'] <= 95) & (df['cci'].shift(1) > 95)),
    'close_long', df['regime'])
df['regime'] = np.where((df['cci'] <= -95) & (df['cci'].shift(1) >= -95), 'short', df['regime'])
df['regime'] = np.where((df['cci'] >= -95) & (df['cci'].shift(1) <= -95),
                        'close_short', df['regime'])


df_regime = df.loc[df['regime'] != 'wait']
# 逆序
df_regime = df_regime[::-1]
unless_regime = df_regime.loc[
    ((df_regime['regime'] == 'close_short') & (df_regime['regime'].shift(1) == 'close_short')) | (
            (df_regime['regime'] == 'close_long') & (df_regime['regime'].shift(1) == 'close_long')) | (
                (df_regime['regime'] == 'close_short') & (df_regime['regime'].shift(1) == 'close_long')) | (
                (df_regime['regime'] == 'close_long') & (df_regime['regime'].shift(1) == 'close_short'))]

for index, row in unless_regime.iterrows():
    df_regime = df_regime.drop([index])
df_regime = df_regime[::-1]
if (df_regime[:1]['regime'] == 'close_long').bool() or (df_regime[:1]['regime'] == 'close_short').bool():
    df_regime = df_regime[1:]
if (df_regime[-1:]['regime'] == 'long').bool() or (df_regime[-1:]['regime'] == 'short').bool():
    df_regime = df_regime[:len(df_regime) - 1]

# print(df_regime[['date', 'close', 'regime']].tail(50))
# 计算本息
l = []
for i in range(1, len(df_regime), 2):
    row_ = df_regime.iloc[i - 1]
    row = df_regime.iloc[i]
    if row['regime'] == 'close_long' and row_['regime'] == 'long':
        principal = principal * row['close'] / row_['close']
        l.append([row['date'], row['close'], principal, 'long', 1])
    elif row['regime'] == 'close_short' and row_['regime'] == 'short':
        principal = principal * row_['close'] / row['close']
        l.append([row['date'], row['close'], principal, 'short', -1])
profits = pd.DataFrame(l, columns=['date', 'close', 'principal', 'type', 's'])
# print(profits)

ax = profits[['close', 'principal']].plot(figsize=(20, 10), grid=True, xticks=profits.index, rot=90, subplots=True,
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
