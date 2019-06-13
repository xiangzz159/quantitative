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


filename = 'BitMEX-170901-190606-4H'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)
stock = StockDataFrame.retype(df)
df['date'] = pd.to_datetime(df['timestamp'], unit='s')

df.index = df.date
stock = StockDataFrame.retype(df)
stock['cci']
stock['stoch_rsi']
# 去掉前几个cci指标不准数据
df = df[10:]

df['signal'] = np.where((df['cci'] >= 95) & (df['cci'].shift(1) < 95), 'long', 'wait')
df['signal'] = np.where(
    ((df['cci'] <= 95) & (df['cci'].shift(1) > 95)),
    'close_long', df['signal'])
df['signal'] = np.where((df['cci'] <= -95) & (df['cci'].shift(1) >= -95), 'short', df['signal'])
df['signal'] = np.where((df['cci'] >= -95) & (df['cci'].shift(1) <= -95),
                        'close_short', df['signal'])

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
    elif row['signal'] == 'trend' and trade_price > 0:
        earnings = close / trade_price if side == 'long' else trade_price / close
        btc_amount *= earnings * (1 - market_rate)
        side = 'wait'
        trade_price = 0
    elif (row['signal'] == 'long' and side == 'short') or (row['signal'] == 'short' and side == 'long'):
        earnings = close / trade_price if side == 'long' else trade_price / close
        btc_amount *= earnings * (1 - market_rate)
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
ax[0].set_title(filename + ' CCI策略回测')
plt.show()
