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

# filename = 'BitMEX-180101-190227-1H'
filename = 'BTC2017-09-01-now-4H'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)
df['Timestamp'] = df['Timestamp'] / 1000
df['date'] = pd.to_datetime(df['Timestamp'], unit='s')
df.index = df.date

stock = StockDataFrame.retype(df)

n = 20
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
# 删除包含NaN值的任何行 axis：0行，1列;inplace修改原始df
df.dropna(axis=0, inplace=True)
ama = 0
arr = [0]
for i in range(1, len(df)):
    row = df.iloc[i]
    ama = ama + row['smooth'] * (row['close'] - ama)
    arr.append(ama)

df['ama'] = np.array(arr)

# 策略1
df['ama_diff'] = df['ama'] - df['ama'].shift(1)
percentage = 0.1
N = 20
df['ama_std'] = df['ama_diff'].rolling(window=N, center=False).std()
df['filter1'] = df['ama_std'] * percentage
df['signal1'] = 'wait'
df['signal1'] = np.where((df['signal1'] == 'wait') & (
        (df['ama'] - df['ama'].shift(1) > df['filter1']) | (df['ama'] - df['ama'].shift(2) > df['filter1'])), 'long',
                         df['signal1'])
df['signal1'] = np.where((df['signal1'] == 'wait') & ((df['ama'] - df['ama'].shift(1) < -1 * df['filter1']) | (
        df['ama'] - df['ama'].shift(2) < -1 * df['filter1'])), 'short', df['signal1'])

# 策略2
df['ma'] = df['close'].rolling(window=n, center=False).mean()
df['ma_std'] = df['ma'].rolling(window=N, center=False).std()
df['filter2'] = percentage * df['ma_std']
df['signal2'] = 'wait'
df['signal2'] = np.where((df['signal2'] == 'wait') & (df['close'] - df['ma'] > df['filter2']), 'long', df['signal2'])
df['signal2'] = np.where((df['signal2'] == 'wait') & (df['close'] - df['ma'] < -1 * df['filter2']), 'short',
                         df['signal2'])

# 策略3
df['signal3'] = 'wait'
df['signal3'] = np.where((df['signal3'] == 'wait') & (df['close'] - df['ama'] > df['filter1']), 'long', df['signal3'])
df['signal3'] = np.where((df['signal3'] == 'wait') & (df['close'] - df['ama'] < -1 * df['filter1']), 'short', df['signal3'])

df = df[50:]

df['signal'] = df['signal1']
# df['signal'] = np.where((df['signal1'] == df['signal2']) & (df['signal1'] == df['signal3']) & (df['signal3'] == df['signal2']), df['signal1'], 'wait')

df['signal'] = np.where((df['signal'] == 'wait') & (df['signal'].shift(1) != 'wait'), 'close', df['signal'])
df['signal'] = np.where(df['signal'] == df['signal'].shift(1), 'wait', df['signal'])

df = df.loc[df['signal'] != 'wait']

side = 'wait'
asset = 1
last_price = 0
market_rate = 7.5 / 10000

l = []
for i in range(len(df)):
    l.append(asset)
    row = df.iloc[i]
    if row['signal'] in ['long', 'short'] and side == 'wait':
        side = row['signal']
        last_price = row['close']
        asset *= (1 - market_rate)
    elif row['signal'] in ['long', 'short'] and side != 'wait':
        close = row['close']
        asset = asset * close / last_price if side == 'long' else asset * last_price / close
        asset *= (1 - market_rate)
        last_price = close
        side = row['signal']
    elif row['signal'] == 'close' and side != 'wait':
        close = row['close']
        asset = asset * close / last_price if side == 'long' else asset * last_price / close
        asset *= (1 - market_rate)
        last_price = 0
        side = 'wait'
    else:
        print(i, last_price, side, row['signal'])


plt.plot(np.array(l))
plt.show()

# df[['signal', 'signal1']].to_csv('../data/signal.csv')


# ax = df[['close', 'ama']].plot(figsize=(20, 10), grid=True, xticks=df.index, rot=90, subplots=True, style='b')
# interval = int(len(df) / (40 - 1))
# ax[0].xaxis.set_major_locator(ticker.MaxNLocator(40))
# ax[0].set_xticklabels(df.index[::interval])
# plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
# ax[0].set_title('Title')
# plt.show()
