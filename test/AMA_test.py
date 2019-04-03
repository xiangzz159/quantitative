# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2019/4/2 15:57

@desc: 

'''

import time
from tools import public_tools, data2df
from tools.stockstats import StockDataFrame
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

filename = 'BitMEX-180101-190227-1H'
# filename = 'BTC2017-09-01-now-4H'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)
df['Timestamp'] = df['Timestamp'] / 1000
df['date'] = pd.to_datetime(df['Timestamp'], unit='s')
df.index = df.date

stock = StockDataFrame.retype(df)

n = 21
# 1. 价格方向
df['direction'] = df['close'] - df['close'].shift(n)
# 2. 波动性
df['price_diff'] = abs(df['close'] - df['close'].shift(1))
df['volatility'] = df['price_diff'].rolling(center=False, window=n).sum()
# 3. 效率系数 ER
df['ER'] = df['direction'] / df['volatility']
# 4. 变换上述系数为趋势速度
fastest = 2 / (2 + 1)
slowest = 2 / (30 + 1)
df['smooth'] = df['ER'] * (fastest - slowest) + slowest
df['smooth'] = df['smooth'] ** 2
df = df[30:]
ama = 0
arr = [0]
for i in range(1, len(df)):
    row = df.iloc[i]
    ama = ama + row['smooth'] * (row['close'] - ama)
    arr.append(ama)

df['ama'] = np.array(arr)

# df['ama_diff'] = df['ama_'] - df['ama_'].shift(1)
# df['filter'] = 0.1 * df['ama_diff'].rolling(center=False, window=n).std()
# df['ama_min'] = df['ama_'].rolling(center=False, window=n).min()
# df['ama_max'] = df['ama_'].rolling(center=False, window=n).max()
# df['signal'] = 'wait'
# df['signal'] = np.where((df['signal'] == 'wait') & (df['ama_'] - df['ama_min'] > df['filter']), 'long', df['signal'])
# df['signal'] = np.where((df['signal'] == 'wait') & (df['ama_max'] - df['ama_'] > df['filter']), 'short', df['signal'])


# df[['close', 'direction', 'price_diff', 'volatility', 'ER']].to_csv('../data/data.csv', index=False)

plt.plot(df['close'])
plt.plot(df['ama'])


# ax = df[['close', 'ama']].plot(figsize=(20, 10), grid=True, xticks=df.index, rot=90, subplots=True, style='b')
# interval = int(len(df) / (40 - 1))
# ax[0].xaxis.set_major_locator(ticker.MaxNLocator(40))
# ax[0].set_xticklabels(df.index[::interval])
# plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
# ax[0].set_title('Title')
plt.show()
