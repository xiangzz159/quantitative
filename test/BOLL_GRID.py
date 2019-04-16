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

filename = 'BitMEX-180101-190227-1H'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)
stock = StockDataFrame.retype(df)
df['date'] = pd.to_datetime(df['timestamp'], unit='s')
df.index = df.date
stock['boll']
df['signal'] = 'wait'


def analysis(df):
    # 参数设置
    trend_A1 = 0.3
    trend_A2 = 0.02
    trend_B = 0.04
    trend_C = 0.05
    stop_trade_times = 10
    ts = 3600
    std_percentage = 0.9

    # 通道宽度
    df['boll_width'] = abs(df['boll_ub'] - df['boll_lb'])
    # 单次涨跌幅
    df['change'] = abs((df['close'] - df['close'].shift(1)) / df['close'].shift(1))
    # 趋势判断A：超过通道一定幅度且单次涨幅超过一定幅度
    df['signal'] = np.where(((df['close'] > df['boll_ub'] + df['boll_width'] * trend_A1) | (
            df['close'] < df['boll_lb'] - df['boll_width'] * trend_A1)) & (df['change'] > trend_A2), 'trend',
                            df['signal'])
    # 趋势判断B：单次涨幅累计超过一定幅度
    df['signal'] = np.where(df['change'] > trend_B, 'trend', df['signal'])
    # 趋势判断C：两次涨幅累计超过一定幅度
    df['signal'] = np.where(df['change'] + df['change'].shift(1) > trend_C, 'trend', df['signal'])
    trend_df = df.loc[df['signal'] != 'wait']
    tts = []
    last_row_timestamp = df.iloc[-1]['timestamp']
    for idx, row in trend_df.iterrows():
        last_timestamp = tts[-1].timestamp() if len(tts) > 0 else 0
        t = idx.timestamp()
        for i in range(stop_trade_times):
            if t + (i + 1) * ts > last_timestamp and t + (i + 1) * ts <= last_row_timestamp:
                tts.append(pd.Timestamp(datetime.utcfromtimestamp(t + (i + 1) * ts)))
    if len(tts) > 0:
        df.loc[tts, 'signal'] = 'can_not_trade'
    df['signal'] = np.where(df['signal'] == 'can_not_trade', 'trend', df['signal'])
    w = 50
    df['close'] = df['open']
    df['close_mean'] = df['close'].rolling(
        center=False, window=w).mean()
    df['close_std'] = df['close'].rolling(
        center=False, window=w).std()
    df = df[w:]
    size = 6
    df['grid'] = 0
    for i in range(size, 0, -1):
        df['grid'] = np.where((df['grid'] == 0) & (df['close'] > df['close_mean'] + df['close_std'] * i), i,
                              df['grid'])
        df['grid'] = np.where((df['grid'] == 0) & (df['close'] < df['close_mean'] - df['close_std'] * i), -i,
                              df['grid'])
    df['close_std'] = df['close_std'] * std_percentage

    # 策略1
    df['signal'] = np.where(
        (df['signal'] == 'wait') & (df['close'].shift(1) < df['boll'].shift(1) + df['close_std'].shift(1)) & (
                df['close'] > df['boll'] + df['close_std']), 'long', df['signal'])
    df['signal'] = np.where(
        (df['signal'] == 'wait') & (df['close'].shift(1) > df['boll'].shift(1) + df['close_std'].shift(1)) & (
                df['close'] < df['boll'] + df['close_std']), 'close_long', df['signal'])
    # 策略2
    df['signal'] = np.where(
        (df['signal'] == 'wait') & (df['close'].shift(1) > df['boll'].shift(1) - df['close_std'].shift(1)) & (
                df['close'] < df['boll'] - df['close_std']), 'short', df['signal'])
    df['signal'] = np.where(
        (df['signal'] == 'wait') & (df['close'].shift(1) < df['boll'].shift(1) - df['close_std'].shift(1)) & (
                df['close'] > df['boll'] - df['close_std']), 'close_short', df['signal'])
    # 策略3
    df['signal'] = np.where(
        (df['signal'] == 'wait') & (df['close'] > df['boll_ub'] - df['close_std']) & (df['grid'] > 0) & (
                df['grid'] < 4),
        'short', df['signal'])
    # 策略4
    df['signal'] = np.where(
        (df['signal'] == 'wait') & (df['close'] < df['boll_ub'] + df['close_std']) & (df['grid'] < 0) & (
                df['grid'] > -4),
        'long', df['signal'])
    return df.iloc[-1]


# 账户BTC数量
btc_amount = 1
# 网格权重
grid_power = [0.15, 0.3, 0.55]
# 市价手续费
market_rate = 0.00075
# 限价手续费
limit_rate = -0.00025
# limit_rate = 0.00075
# 初始化参数
side = 'wait'
# 最新交易价格
avg_price = 0
# 仓位
positions = []
# 最新的网格区间
last_grid = 0
cost = 0
btc_amounts = []
for idx in range(500, len(df)):
    btc_amounts.append(btc_amount)
    df_ = df[idx - 500:idx]
    row = analysis(df_)
    print(idx, row['signal'], row['grid'], row['close'])
    if row['signal'] in ['long_', 'short_'] and side == 'wait':
        avg_price = row['close']
        positions.append(avg_price * btc_amount)
        side = row['signal']
        btc_amount *= (1 - market_rate)
    elif (row['signal'] == 'long_' and side == 'short') or (row['signal'] == 'short_' and side == 'long') or (
            row['signal'] == 'long' and side == 'short_') or (row['signal'] == 'short' and side == 'long_'):
        # 平仓
        earnings = (row['close'] - avg_price) * sum(positions) / row['close'] / avg_price \
            if side == 'long' or side == 'long_' else \
            (avg_price - row['close']) * sum(positions) / row['close'] / avg_price
        btc_amount += earnings - (sum(positions) / row['close']) * market_rate

        positions = []
        side = row['signal']
        avg_price = row['close']
        # 开仓，带下划线的全仓，不带下划线的分批建仓
        if side in ['long_', 'short_']:
            positions.append(avg_price * btc_amount)
            btc_amount *= (1 - market_rate)
        else:
            for i in range(abs(row['grid'])):
                positions.append(avg_price * btc_amount * grid_power[i])
            btc_amount -= (sum(positions) / avg_price) * market_rate
            cost = sum(positions) * avg_price
            last_grid = row['grid']
    elif abs(row['grid']) > abs(last_grid) and abs(row['grid']) < len(grid_power) and row['signal'] in ['long',
                                                                                                        'short']:
        if (side == 'long' and row['close'] < avg_price) or (side == 'short' and row['close'] > avg_price):
            num = 0
            for i in range(len(positions), abs(row['grid'])):
                num += int(btc_amount * row['close'] * grid_power[i])
                positions.append(int(btc_amount * row['close'] * grid_power[i]))
            last_grid = row['grid']
            cost += num * row['close']
            avg_price = cost / sum(positions)
            btc_amount -= num / row['close'] * limit_rate
    elif len(positions) > 0 and (abs(row['grid']) > len(grid_power) or row['signal'] == 'trend' or (
            side in ['long', 'short'] and row['signal'] == 'wait')):
        earnings = (row['close'] - avg_price) * sum(positions) / row['close'] / avg_price \
            if side == 'long' else \
            (avg_price - row['close']) * sum(positions) / row['close'] / avg_price
        btc_amount = btc_amount + earnings - (sum(positions) / row['close']) * market_rate
        side = 'wait'
        # 最新交易价格
        avg_price = 0
        # 仓位
        positions = []
        # 最新的网格区间
        last_grid = row['grid']
        cost = 0

df['assets'] = np.array(btc_amounts)
ax = df[['close', 'assets']].plot(figsize=(20, 10), grid=True, xticks=df.index, rot=90, subplots=True, style='b')
interval = int(len(df) / (40 - 1))
# 设置x轴刻度数量
ax[0].xaxis.set_major_locator(ticker.MaxNLocator(40))
# 以date为x轴刻度
ax[0].set_xticklabels(df.index[::interval])
# 美观x轴刻度
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
ax[0].set_title('网格策略回测')
# plt.savefig('../data/' + filename + '_zero_rates_' + str(w) + '.png')
plt.show()
