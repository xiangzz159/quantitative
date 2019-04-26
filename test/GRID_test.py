# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2019/3/22 11:02

@desc: 网格策略优化

'''
import time
from tools import public_tools, data2df
from tools.stockstats import StockDataFrame
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import random

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

filename = 'BitMEX-180101-190227-5M'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)
stock = StockDataFrame.retype(df)
df['date'] = pd.to_datetime(df['timestamp'], unit='s')
df.index = df.date
w = 100

df['close_mean'] = df['close'].rolling(
    center=False, window=w).mean()

df['close_std'] = df['close'].rolling(
    center=False, window=w).std()


df = df[w:]


size = 10
df['grid'] = 0

for i in range(size, 0, -1):
    df['grid'] = np.where((df['grid'] == 0) & (df['close'] > df['close_mean'] + df['close_std'] * i), i,
                          df['grid'])
    df['grid'] = np.where((df['grid'] == 0) & (df['close'] < df['close_mean'] - df['close_std'] * i), -i,
                          df['grid'])

# 账户BTC数量
btc_amount = 1
# 网格权重
grid_power = [0.15, 0.3, 0.55]
rev_grid_power = [0.625, 0.375]
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
# 开仓失败率
failed = 0.225

btc_amounts = []
for idx in range(len(df)):
    btc_amounts.append(btc_amount)
    row = df.iloc[idx]
    close = row['close']
    grid = row['grid']
    if row['grid'] == 0:
        last_grid = row['grid']
        if sum(positions) != 0:
            earnings = (row['close'] - avg_price) * sum(positions) / row['close'] / avg_price \
                if side == 'long' else \
                (avg_price - row['close']) * sum(positions) / row['close'] / avg_price

            if earnings > 0 and abs(last_grid) < 4 and random.random() < failed:
                side = 'wait'
                positions = []
                avg_price = 0
                last_grid = 0
                cost = 0
                continue

            btc_amount = btc_amount + earnings - (sum(positions) / row['close']) * limit_rate
            side = 'wait'
            positions = []
            avg_price = 0
            last_grid = 0
            cost = 0
    elif row['grid'] in [1, 2, 3]:
        if last_grid < row['grid'] and len(positions) < row['grid']:  # 加仓
            if avg_price > 0 and avg_price > row['close']:
                continue
            side = 'short'
            num = 0
            for i in range(len(positions), abs(row['grid'])):
                num += int(btc_amount * row['close'] * grid_power[i])
                positions.append(int(btc_amount * row['close'] * grid_power[i]))
            last_grid = row['grid']
            cost += num * row['close']
            avg_price = cost / sum(positions)
            btc_amount -= (num / row['close']) * limit_rate

    elif row['grid'] in [-1, -2, -3]:
        if last_grid > row['grid'] and len(positions) < abs(row['grid']):  # 加仓
            if avg_price > 0 and avg_price < row['close']:
                continue
            side = 'long'
            num = 0
            for i in range(len(positions), abs(row['grid'])):
                num += int(btc_amount * row['close'] * grid_power[i])
                positions.append(int(btc_amount * row['close'] * grid_power[i]))

            last_grid = row['grid']
            cost += num * row['close']
            avg_price = cost / sum(positions)
            btc_amount -= num / row['close'] * limit_rate
    elif abs(row['grid']) >= 4:
        # 平仓
        if (side == 'long' and row['grid'] < 0) or (side == 'short' and row['grid'] > 0):
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

        if row['grid'] > 0:
            # 做多
            side = 'long'
            num = 0
            limit = min(2, abs(row['grid']) - 3)
            for i in range(len(positions), limit):
                num += int(btc_amount * row['close'] * rev_grid_power[i])
                positions.append(int(btc_amount * row['close'] * rev_grid_power[i]))

            last_grid = row['grid']
            cost += num * row['close']
            avg_price = cost / sum(positions)
            btc_amount -= num / row['close'] * market_rate
        else:
            # 做空
            side = 'short'
            num = 0
            limit = min(2, abs(row['grid']) - 3)
            for i in range(len(positions), limit):
                num += int(btc_amount * row['close'] * rev_grid_power[i])
                positions.append(int(btc_amount * row['close'] * rev_grid_power[i]))
            last_grid = row['grid']
            cost += num * row['close']
            avg_price = cost / sum(positions)
            btc_amount -= (num / row['close']) * market_rate

df['assets'] = np.array(btc_amounts)

ax = df[['close', 'assets']].plot(figsize=(20, 10), grid=True, xticks=df.index, rot=90, subplots=True,
                                  style='b')
interval = int(len(df) / (40 - 1))
# 设置x轴刻度数量
ax[0].xaxis.set_major_locator(ticker.MaxNLocator(40))
# 以date为x轴刻度
ax[0].set_xticklabels(df.index[::interval])
# 美观x轴刻度
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
ax[0].set_title('网格策略回测-未优化')
# plt.savefig('../data/' + filename + '_zero_rates_' + str(w) + '.png')
plt.show()
