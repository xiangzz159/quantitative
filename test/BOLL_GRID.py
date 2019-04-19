# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2019/4/11 15:50

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

filename = 'poloniex_1H'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)
stock = StockDataFrame.retype(df)
df['date'] = pd.to_datetime(df['timestamp'], unit='s')
df.index = df.date
stock['boll']
df['signal'] = 'wait'
# 参数设置
trend_A1 = 0.3
trend_A2 = 0.02
trend_B = 0.04
trend_C = 0.05
# trend_D = 0.07
stop_trade_times = 5
ts = 3600
std_percentage = 0.6
w = 200
volatility = 0.001

# 通道宽度
df['boll_width'] = abs(df['boll_ub'] - df['boll_lb'])
# 单次涨跌幅
df['change'] = abs((df['close'] - df['close'].shift(1)) / df['close'].shift(1))
# 两次涨幅
df['change_2'] = abs((df['close'] - df['close'].shift(2)) / df['close'].shift(2))
# 趋势判断A：超过通道一定幅度且单次涨幅超过一定幅度
df['signal'] = np.where(((df['close'] > df['boll_ub'] + df['boll_width'] * trend_A1) | (
        df['close'] < df['boll_lb'] - df['boll_width'] * trend_A1)) & (df['change'] > trend_A2), 'trend',
                        df['signal'])
# 趋势判断B：单次涨幅累计超过一定幅度
df['signal'] = np.where(df['change'] > trend_B, 'trend', df['signal'])
# 趋势判断C：两次涨幅累计超过一定幅度
df['signal'] = np.where(df['change_2'] > trend_C, 'trend', df['signal'])

for i in range(stop_trade_times):
    df['signal'] = np.where((df['signal'] == 'wait') & (df['signal'].shift(1 + i) == 'trend'), 'can_not_trade',
                            df['signal'])
df['signal'] = np.where(df['signal'] == 'can_not_trade', 'trend', df['signal'])
df['close'] = df['open']
df['close_mean'] = df['close'].rolling(
    center=False, window=w).mean()
df['close_std'] = df['close'].rolling(
    center=False, window=w).std()

df['close_std'] = df['close_std'] * std_percentage
df['rate'] = df['close_std'] / df['boll_width']
# df['close_std'] = np.where(df['rate'] > 0.1, df['boll_width'] * 0.1, df['close_std'])

# 策略1
df['signal'] = np.where(
    (df['signal'] == 'wait') & (df['change'] > volatility) & (
            df['close'].shift(1) < df['boll'].shift(1) + df['close_std'].shift(1)) & (
            df['close'] > df['boll'] + df['close_std']), 'long_', df['signal'])
df['signal'] = np.where(
    (df['signal'] == 'wait') & (df['close'].shift(1) > df['boll'].shift(1) + df['close_std'].shift(1) / 2) & (
            df['close'] < df['boll'] + df['close_std'] / 2), 'close_long', df['signal'])
# 策略2
df['signal'] = np.where(
    (df['signal'] == 'wait') & (df['change'] > volatility) & (
            df['close'].shift(1) > df['boll'].shift(1) - df['close_std'].shift(1)) & (
            df['close'] < df['boll'] - df['close_std']), 'short_', df['signal'])
df['signal'] = np.where(
    (df['signal'] == 'wait') & (df['close'].shift(1) < df['boll'].shift(1) - df['close_std'].shift(1) / 2) & (
            df['close'] > df['boll'] - df['close_std'] / 2), 'close_short', df['signal'])

# 策略3
df['signal'] = np.where(
    (df['signal'] == 'wait') & (df['change'] > volatility) & (df['close'] > df['boll_ub'] - df['close_std'] / 4) & (
            df['close'] < df['boll_ub'] + df['close_std'] / 2), 'short', df['signal'])
# 策略4
df['signal'] = np.where(
    (df['signal'] == 'wait') & (df['change'] > volatility) & (df['close'] < df['boll_lb'] + df['close_std'] / 4) & (
            df['close'] > df['boll_lb'] - df['close_std'] / 2), 'long', df['signal'])

df['stop_price'] = 0
df['stop_price'] = np.where(df['signal'] == 'short', df['boll_ub'] + df['close_std'] / 2, df['stop_price'])
df['stop_price'] = np.where(df['signal'] == 'long', df['boll_lb'] - df['close_std'] / 2, df['stop_price'])
df['signal'] = np.where(df['signal'] == 'long_', 'long', df['signal'])
df['signal'] = np.where(df['signal'] == 'short_', 'short', df['signal'])
df = df[w:]

signal_df = df.loc[(df['signal'] != 'wait')]
signal_df['signal'] = np.where((signal_df['signal'] != 'trend') & (signal_df['signal'] == signal_df['signal'].shift(1)),
                               'wait', signal_df['signal'])
signal_df = signal_df.loc[(signal_df['signal'] != 'wait')]
df['signal'] = np.where(df['signal'] != 'trend', 'wait', df['signal'])
long = []
short = []
close_long = []
close_short = []
for idx, row in signal_df.iterrows():
    t = idx.timestamp()
    if row['signal'] == 'long':
        long.append(pd.Timestamp(datetime.utcfromtimestamp(t)))
    elif row['signal'] == 'short':
        short.append(pd.Timestamp(datetime.utcfromtimestamp(t)))
    elif row['signal'] == 'close_long':
        close_long.append(pd.Timestamp(datetime.utcfromtimestamp(t)))
    elif row['signal'] == 'close_short':
        close_short.append(pd.Timestamp(datetime.utcfromtimestamp(t)))

if len(long) > 0:
    df.loc[long, 'signal'] = 'long'
if len(short) > 0:
    df.loc[short, 'signal'] = 'short'
if len(close_short) > 0:
    df.loc[close_short, 'signal'] = 'close_short'
if len(close_long) > 0:
    df.loc[close_long, 'signal'] = 'close_long'

# 市价手续费
market_rate = 0.00075
# 限价手续费
limit_rate = -0.00025
# limit_rate = 0.00075
# 账户BTC数量
btc_amount = 1
side = 'wait'
trade_price = 0
stop_price = 0
btc_amounts = []
# df = df[9820:]
for idx, row in df.iterrows():
    btc_amounts.append(btc_amount)
    close = row['close']
    print(idx, close, row['signal'], btc_amount)
    # if stop_price > 0:
    #     if (side == 'long' and row['low'] < stop_price) or (side == 'short' and row['high'] > stop_price):
    #         earnings = trade_price / stop_price if side == 'short' else stop_price / trade_price
    #         btc_amount *= earnings * (1 - market_rate)
    #         side = 'wait'
    #         trade_price = 0
    #         stop_price = 0
    #         continue
    if side == 'wait' and row['signal'] in ['short', 'long']:
        side = row['signal']
        trade_price = row['close']
        btc_amount *= (1 - market_rate)
        stop_price = row['stop_price']
    elif row['signal'] == 'trend' and trade_price > 0:
        earnings = close / trade_price if side == 'long' else trade_price / close
        btc_amount *= earnings * (1 - market_rate)
        side = 'wait'
        trade_price = 0
        stop_price = 0
    elif (row['signal'] == 'long' and side == 'short') or (row['signal'] == 'short' and side == 'long'):
        earnings = close / trade_price if side == 'long' else trade_price / close
        btc_amount *= earnings * (1 - market_rate)
        side = row['signal']
        trade_price = row['close']
        btc_amount *= (1 - market_rate)
        stop_price = row['stop_price']
    elif (row['signal'] == 'close_long' and side == 'long') or (row['signal'] == 'close_short' and side == 'short'):
        earnings = close / trade_price if side == 'long' else trade_price / close
        btc_amount *= earnings * (1 - market_rate)
        side = 'wait'
        trade_price = 0
        stop_price = 0

df['assets'] = np.array(btc_amounts)
ax = df[['close', 'assets']].plot(figsize=(20, 10), grid=True, xticks=df.index, rot=90, subplots=True, style='b')
interval = int(len(df) / (40 - 1))
# 设置x轴刻度数量
ax[0].xaxis.set_major_locator(ticker.MaxNLocator(40))
# 以date为x轴刻度
ax[0].set_xticklabels(df.index[::interval])
# 美观x轴刻度
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
ax[0].set_title('BOLL_网格策略回测')
plt.show()
