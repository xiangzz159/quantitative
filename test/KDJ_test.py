# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2019/6/5 22:29

@desc:

'''
import time
from tools import public_tools, data2df
from tools.stockstats import StockDataFrame
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import random
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

filename = 'BTC2017-09-01-now-4H'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)
stock = StockDataFrame.retype(df)
df['date'] = pd.to_datetime(df['timestamp'], unit='s')
df.index = df.date
df['signal'] = 'wait'
stock['kdjk']

high_limit = 80
low_limit = 20

# k上穿d时做多
df['signal'] = np.where(
    (df['signal'] == 'wait') & (df['kdjk'].shift(1) < df['kdjd'].shift(1)) & (df['kdjk'] > df['kdjd']), 'long',
    df['signal'])

# k下穿d时做空
df['signal'] = np.where(
    (df['signal'] == 'wait') & (df['kdjk'].shift(1) > df['kdjd'].shift(1)) & (df['kdjk'] < df['kdjd']), 'short',
    df['signal'])

df['signal'] = np.where(
    (df['signal'] == 'wait') & (df['kdjk'] < low_limit) & (df['kdjd'] < low_limit) & (df['kdjj'] < low_limit),
    'close_short', df['signal'])
df['signal'] = np.where(
    (df['signal'] == 'wait') & (df['kdjk'] > high_limit) & (df['kdjd'] > high_limit) & (df['kdjj'] > high_limit),
    'close_long', df['signal'])


df = df[100:]
# 市价手续费
market_rate = 0.00075
# 限价手续费
limit_rate = -0.00025
# limit_rate = 0.00075
# 账户BTC数量
btc_amount = 1
side = 'wait'
trade_price = 0
btc_amounts = []
for idx, row in df.iterrows():
    btc_amounts.append(btc_amount)
    close = row['close']
    print(idx, close, row['signal'], btc_amount)
    if side == 'wait' and row['signal'] in ['short', 'long']:
        side = row['signal']
        trade_price = row['close']
        btc_amount *= (1 - market_rate)
    elif (row['signal'] == 'long' and side == 'short') or (row['signal'] == 'short' and side == 'long'):
        earnings = close / trade_price if side == 'long' else trade_price / close
        btc_amount *= earnings * (1 - market_rate * 2)
        side = row['signal']
        trade_price = row['close']
        btc_amount *= (1 - market_rate)
    elif (row['signal'] == 'close_long' and side == 'long') or (row['signal'] == 'close_short' and side == 'short'):
        earnings = close / trade_price if side == 'long' else trade_price / close
        btc_amount *= earnings * (1 - market_rate)
        side = 'wait'
        trade_price = 0

df['assets'] = np.array(btc_amounts)
ax = df[['close', 'assets']].plot(figsize=(20, 10), grid=True, xticks=df.index, rot=90, subplots=True, style='b')
interval = int(len(df) / (40 - 1))
# 设置x轴刻度数量
ax[0].xaxis.set_major_locator(ticker.MaxNLocator(40))
# 以date为x轴刻度
ax[0].set_xticklabels(df.index[::interval])
# 美观x轴刻度
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
ax[0].set_title(filename + ' KDJ策略回测')
plt.show()
