# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2019/12/30 15:10

@desc:

'''
from tools import data2df
from tools.BmBackTest import BmBackTest
import pandas as pd
import numpy as np


def heikin_ashi(df1):
    df1.reset_index(inplace=True)

    df1['HA close'] = (df1['open'] + df1['close'] + df1['high'] + df1['low']) / 4

    # initialize heikin ashi open
    df1['HA open'] = float(0)
    df1['HA open'][0] = df1['open'][0]

    for n in range(1, len(df1)):
        df1.at[n, 'HA open'] = (df1['HA open'][n - 1] + df1['HA close'][n - 1]) / 2

    temp = pd.concat([df1['HA open'], df1['HA close'], df1['low'], df1['high']], axis=1)
    df1['HA high'] = temp.apply(max, axis=1)
    df1['HA low'] = temp.apply(min, axis=1)

    return df1


def analysis(kline):
    df1 = pd.DataFrame(kline, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df1 = heikin_ashi(df1)
    stls = 2

    df1['signals'] = 0

    # i use cumulated sum to check how many positions i have longed
    # i would ignore the exit signal prior to no long positions in the portfolio
    # i also keep tracking how many long positions i have got
    # long signals cannot exceed the stop loss limit
    df1['cumsum'] = 0
    df1['signals'] = np.where((df1['HA open'] > df1['HA close']) & (df1['HA open'] == df1['HA high']) &
                              (abs(df1['HA open'] - df1['HA close']) > abs(
                                  df1['HA open'].shift(1) - df1['HA close'].shift(1))) &
                              (df1['HA open'].shift(1) > df1['HA close'].shift(1)), 1, 0)

    df1['signals'] = np.where((df1['HA open'] < df1['HA close']) & (df1['HA open'] == df1['HA low']) &
                              (df1['HA open'].shift(1) < df1['HA close'].shift(1)), -1, 0)

    for n in range(1, len(df1)):

        if (df1['HA open'][n] > df1['HA close'][n] and df1['HA open'][n] == df1['HA high'][n] and
                np.abs(df1['HA open'][n] - df1['HA close'][n]) < np.abs(
                    df1['HA open'][n - 1] - df1['HA close'][n - 1]) and
                df1['HA open'][n - 1] > df1['HA close'][n - 1]):

            df1.at[n, 'signals'] = 1
            df1['cumsum'] = df1['signals'].cumsum()

            # stop longing positions
            if df1['cumsum'][n] > stls:
                df1.at[n, 'signals'] = 0


        elif (df1['HA open'][n] < df1['HA close'][n] and df1['HA open'][n] == df1['HA low'][n] and
              np.abs(df1['HA open'][n] - df1['HA close'][n]) < np.abs(
                    df1['HA open'][n - 1] - df1['HA close'][n - 1]) and
              df1['HA open'][n - 1] < df1['HA close'][n - 1]):

            df1.at[n, 'signals'] = -1
            df1['cumsum'] = df1['signals'].cumsum()

            # stop shorting positions
            if df1['cumsum'][n] < -stls:
                df1.at[n, 'signals'] = 0

    df1['cumsums'] = df1['signals'].cumsum()

    # df1['signal'] = 'wait'
    # df1['signal'] = np.where((df1['cumsums'] > 0) & (df1['cumsums'] - df1['cumsums'].shift(1) == 1), 'long',
    #                          df1['signal'])
    # df1['signal'] = np.where((df1['cumsums'] == 0) & (df1['cumsums'].shift(1) > 0), 'close_long', df1['signal'])
    # df1['signal'] = np.where((df1['cumsums'] < 0) & (df1['cumsums'] - df1['cumsums'].shift(1) == -1), 'short',
    #                          df1['signal'])
    # df1['signal'] = np.where((df1['cumsums'] == 0) & (df1['cumsums'].shift(1) < 0), 'close_short', df1['signal'])

    # fileNames = '../data/HA2.csv'
    # df1[['close', 'signals']].to_csv(fileNames, index=None)
    return df1.iloc[-1]


# filename = 'BitMEX-ETH-180803-190817-4H'
filename = 'BitMEX-170901-190606-1H'
# filename='BitMEX-170901-190606-4H'
df1 = data2df.csv2df(filename + '.csv')
df1 = df1.astype(float)

datas = df1.values

backtest = BmBackTest({
    'asset': 1
})

level = 0.33333
cumsums = 0
order_amts = []
for i in range(510, len(df1)):
    test_data = datas[i - 500: i]
    row = analysis(test_data)

    if row['cumsums'] != 0:
        if abs(row['cumsums']) > abs(cumsums):
            side = 'long' if row['cumsums'] > 0 else 'short'
            if backtest.side == 'wait':
                # create order
                amount = 0
                for j in range(int(abs(row['cumsums']))):
                    amt = int(backtest.asset * row['close'] * level)
                    amount += amt
                    order_amts.append(amt)
                backtest.create_order(side, "market", row['close'], amount)
            elif backtest.side == 'long' and row['cumsums'] > cumsums and backtest.open_price > row['close'] and len(
                    order_amts) < abs(row['cumsums']):
                # add long positions
                amount = int((backtest.asset + backtest.float_profit) * row['close'] * level)
                backtest.create_order(side, "market", row['close'], amount)
                order_amts.append(amount)
            elif backtest.side == 'short' and row['cumsums'] < cumsums and backtest.open_price < row['close'] and len(
                    order_amts) < abs(row['cumsums']):
                # add short positions
                amount = int((backtest.asset + backtest.float_profit) * row['close'] * level)
                backtest.create_order(side, "market", row['close'], amount)
                order_amts.append(amount)

        elif row['cumsums'] == cumsums:
            backtest.add_data(row['close'], row['high'], row['low'])
        else:
            times = int(abs(cumsums - row['cumsums']))
            amount = 0
            for idx in range(times):
                amount += order_amts.pop()
            backtest.close_positions(row['close'], 'market', amount)
    else:
        if backtest.side != 'wait':
            backtest.close_positions(row['close'], 'market')
            order_amts = []
        else:
            backtest.add_data(row['close'], row['high'], row['low'])

    cumsums = row['cumsums']
    print(i, row['close'], row['cumsums'], backtest.open_price, order_amts, backtest.side, backtest.asset,
          backtest.float_profit,
          backtest.asset + backtest.float_profit)

backtest.show("HeikinAshi")
