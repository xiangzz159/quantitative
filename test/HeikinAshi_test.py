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
    stls = 3

    df1['signal'] = 0
    df1['signals'] = 0

    # i use cumulated sum to check how many positions i have longed
    # i would ignore the exit signal prior to no long positions in the portfolio
    # i also keep tracking how many long positions i have got
    # long signals cannot exceed the stop loss limit
    df1['cumsum'] = 0

    for n in range(1, len(df1)):

        if (df1['HA open'][n] > df1['HA close'][n] and df1['HA open'][n] == df1['HA high'][n] and
                np.abs(df1['HA open'][n] - df1['HA close'][n]) > np.abs(
                    df1['HA open'][n - 1] - df1['HA close'][n - 1]) and
                df1['HA open'][n - 1] > df1['HA close'][n - 1]):

            df1.at[n, 'signal'] = 1
            df1['cumsum'] = df1['signal'].cumsum()

            # stop longing positions
            if df1['cumsum'][n] > stls:
                df1.at[n, 'signal'] = 0


        elif (df1['HA open'][n] < df1['HA close'][n] and df1['HA open'][n] == df1['HA low'][n] and
              df1['HA open'][n - 1] < df1['HA close'][n - 1]):

            df1.at[n, 'signal'] = -1
            df1['cumsum'] = df1['signal'].cumsum()

            # if long positions i hold are more than one
            # its time to clear all my positions
            # if there are no long positions in my portfolio
            # ignore the exit signal
            if df1['cumsum'][n] > 0:
                df1.at[n, 'signals'] = -1 * (df1['cumsum'][n - 1])

            if df1['cumsum'][n] < 0:
                df1.at[n, 'signals'] = 0

    df1['cumsums'] = df1['signals'].cumsum()

    fileNames = '../data/HA.csv'
    df1[['signal', 'signals', 'cumsum', 'cumsums']].to_csv(fileNames, index=None)
    return df1.iloc[-1]



# filename = 'BitMEX-ETH-180803-190817-4H'
filename = 'BitMEX-170901-191107-4H'
# filename='BitMEX-170901-190606-4H'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)

datas = df.values

backtest = BmBackTest({
    'asset': 1
})

level = 1

analysis(datas)

# for i in range(370, len(df)):
#     test_data = datas[i - 370: i]
#     row = analysis(test_data)
#     print(i, row['signals'], row['close'])
#
#     if row['signal'] in ['long', 'short']:
#         amount = int(backtest.asset * row['close'] * level)
#         backtest.create_order(row['signal'], "market", row['close'], amount)
#         # backtest.stop_price = row['%s_stop_price' % row['signal']]
#
#     elif (row['signal'] == 'close_long' and backtest.side == 'long') or (
#             row['signal'] == 'close_short' and backtest.side == 'short'):
#         backtest.close_positions(row['close'], 'market')
#
#     else:
#         backtest.add_data(row['close'], row['high'], row['low'])
#
#     print(i, row['close'], row['signal'], backtest.open_price, backtest.side, backtest.asset, backtest.float_profit,
#           backtest.asset + backtest.float_profit)
#
# backtest.show("HeikinAshi")
