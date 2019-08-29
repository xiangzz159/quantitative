# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2019/3/22 11:02

@desc: 网格策略优化

'''
from tools.BmBackTest import BmBackTest
from tools import public_tools, data2df
import numpy as np
import pandas as pd
import copy
import random


def analysis(kline):
    df = pd.DataFrame(kline, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = df['timestamp'] / 1000
    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    df.index = df.date
    w = 100
    df['close_mean'] = df['close'].rolling(center=False, window=w).mean()
    df['close_std'] = df['close'].rolling(center=False, window=w).std()
    df = df[w:]
    size = 8
    df['grid'] = 0
    for i in range(size, 0, -1):
        df['grid'] = np.where((df['grid'] == 0) & (df['close'] > df['close_mean'] + df['close_std'] * i), i,
                              df['grid'])
        df['grid'] = np.where((df['grid'] == 0) & (df['close'] < df['close_mean'] - df['close_std'] * i), -i,
                              df['grid'])
    return df.iloc[-1]


filename = 'BitMEX-170901-190606-15M'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)

datas = df.values

backtest = BmBackTest({
    'asset': 1
})

level = 1

last_grid = 0
positions = []

# 网格权重
grid_power = [0.15, 0.3, 0.55]
lever = 1
open_price = 0
stop_rate = 0.099

# for i in range(750, len(df)):
for i in range(750, 30000):
    test_df = datas[i - 750: i - 1]
    row = analysis(test_df)

    close = row['close']
    grid = row['grid']
    side_ = 'long' if grid < 0 else 'short'
    total_amount = 0
    if row['grid'] == 0:
        last_grid = row['grid']
        if sum(positions) != 0:
            backtest.close_positions(row['close'], 'limit')
            positions = []


    elif abs(grid) <= 3:
        if (last_grid < 0 and grid > 0) or (last_grid > 0 and grid < 0):
            backtest.close_positions(row['close'], 'limit')
            positions = []
            last_grid = 0

        if abs(last_grid) < abs(grid) and len(positions) < abs(grid):
            if (backtest.open_price > 0 and backtest.open_price > close and grid > 0) or (
                    backtest.open_price > 0 and backtest.open_price < close and grid < 0):
                continue

            old_positions = copy.deepcopy(positions)

            for j in range(len(positions), abs(grid)):
                total_amount += int(close * (backtest.asset + backtest.float_profit) * lever * grid_power[j])
                positions.append(int(close * (backtest.asset + backtest.float_profit) * lever * grid_power[j]))

            # 2-high 3-low
            if (row['close'] >= datas[i][3] and side_ == 'long') or (row['close'] <= datas[i][2] and side_ == 'short'):
                if open_price > 0:
                    backtest.create_order(side_, 'limit', open_price, total_amount)
                    open_price = 0
                else:
                    backtest.create_order(side_, 'limit', row['close'], total_amount)
                # backtest.stop_price = backtest.open_price / stop_rate if side_ == 'short' else backtest.open_price * stop_rate
            else:
                open_price = row['close']
                # print(row['close'], datas[i][2], datas[i][3], side_)
                positions = old_positions
            last_grid = grid
        elif abs(last_grid) > abs(grid):
            last_grid = grid
            open_price = 0



    else:
        last_grid = grid
        positions = []
        backtest.close_positions(row['close'], 'market')

    print(i, row['close'], row['grid'], backtest.open_price, positions, backtest.side, last_grid,
          backtest.asset + backtest.float_profit)

backtest.show("GRID")
