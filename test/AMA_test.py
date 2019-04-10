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


def analysis(df):
    n = 20
    percentage = 0.1
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
    df['ama_diff'] = df['ama'] - df['ama'].shift(1)
    df['ama_std'] = df['ama_diff'].rolling(window=n, center=False).std()
    df['filter'] = df['ama_std'] * percentage
    df['signal'] = 'wait'
    df['signal'] = np.where((df['signal'] == 'wait') & (
            (df['ama'] - df['ama'].shift(1) > df['filter']) | (df['ama'] - df['ama'].shift(2) > df['filter'])),
                            'long',
                            df['signal'])
    df['signal'] = np.where((df['signal'] == 'wait') & ((df['ama'] - df['ama'].shift(1) < -1 * df['filter']) | (
            df['ama'] - df['ama'].shift(2) < -1 * df['filter'])), 'short', df['signal'])
    return df.iloc[-1]


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

filename = 'BTC2017-09-01-now-1D'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)
stock = StockDataFrame.retype(df)

df['date'] = pd.to_datetime(df['timestamp'], unit='s')
df.index = df.date
df[['close']] = round(df[['close']], 1)

asset_changes = []
asset = 1
side = 'wait'
open_price = 0
market_rate = 7.5 / 10000
for i in range(200, len(df)):
    asset_changes.append(asset)
    test_df = df[i - 125: i]
    row = analysis(test_df)

    if row['signal'] in ['long', 'short'] and side == 'wait':
        side = row['signal']
        open_price = row['close']
        asset *= (1 - market_rate)
    elif row['signal'] in ['long', 'short'] and side != row['signal']:
        asset = asset * (row['close'] / open_price) if side == 'long' else asset * (open_price / row['close'])
        asset *= (1 - market_rate)
        open_price = row['close']
        side = row['signal']
    elif row['signal'] == 'wait' and side in ['long', 'short']:
        asset = asset * (row['close'] / open_price) if side == 'long' else asset * (open_price / row['close'])
        asset *= (1 - market_rate)
        open_price = 0
        side = 'wait'

num = len(asset_changes)
df = df[-1 * num:]
df['asset'] = np.array(asset_changes)

ax = df[['close', 'asset']].plot(figsize=(20, 10), grid=True, xticks=df.index, rot=90, subplots=True, style='b')
interval = int(len(df) / (40 - 1))
ax[0].xaxis.set_major_locator(ticker.MaxNLocator(40))
ax[0].set_xticklabels(df.index[::interval])
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
ax[0].set_title('AMA')
plt.show()
