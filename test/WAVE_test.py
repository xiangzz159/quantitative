# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2019/8/3 18:32

@desc:

'''

from tools import data2df
import numpy as np
import pandas as pd
from tools.BmBackTest import BmBackTest
import heapq
from tools.stockstats import StockDataFrame


def wave_guess(arr, wn):
    # 计算最大的N个值，认为是波峰
    wave_crest = heapq.nlargest(wn, enumerate(arr), key=lambda x: x[1])

    # 计算最小的N个值，认为是波谷
    wave_base = heapq.nsmallest(wn, enumerate(arr), key=lambda x: x[1])

    wave_crest_x = []  # 波峰x
    wave_crest_y = []  # 波峰y
    for i, j in wave_crest:
        wave_crest_x.append(i)
        wave_crest_y.append(j)

    wave_base_x = []  # 波谷x
    wave_base_y = []  # 波谷y
    for i, j in wave_base:
        wave_base_x.append(i)
        wave_base_y.append(j)

    return wave_crest_x, wave_base_x


# 波峰波谷猜想
wn = 10
hs = -60


def analysis(kline):
    data_len = len(kline)
    df = pd.DataFrame(kline, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    stock = StockDataFrame.retype(df)

    df['wave'] = 0
    df = df[hs:]
    data_len_ = len(df)
    wave_crest_x, wave_base_x = wave_guess(df['close'].values, wn)
    wave_base_x_ = []
    wave_crest_x_ = []
    for x in wave_crest_x:
        idx = data_len - data_len_ + x
        wave_crest_x_.append(idx)
    for x in wave_base_x:
        idx = data_len - data_len_ + x
        wave_base_x_.append(idx)
    df.loc[wave_crest_x_, 'wave'] = 1
    df.loc[wave_base_x_, 'wave'] = -1

    df['signal'] = 'wait'
    # 策略：short
    # df['signal'] = np.where((df['wave'] == 0) & (df['wave'].shift(1) == 1) & (df['wave'].shift(2) == 1), 'short',
    #                         df['signal'])
    df['signal'] = np.where((df['wave'] == 0) & (df['wave'].shift(1) == 0) & (df['wave'].shift(2) == 1), 'short',
                            df['signal'])
    df['signal'] = np.where(
        (df['wave'] == 0) & (df['wave'].shift(1) == 0) & (df['wave'].shift(2) == 0) & (df['wave'].shift(3) == 1),
        'short', df['signal'])
    # 策略：long
    # df['signal'] = np.where((df['wave'] == 0) & (df['wave'].shift(1) == -1) & (df['wave'].shift(2) == -1), 'long',
    #                         df['signal'])
    df['signal'] = np.where((df['wave'] == 0) & (df['wave'].shift(1) == 0) & (df['wave'].shift(2) == -1), 'long',
                            df['signal'])
    df['signal'] = np.where(
        (df['wave'] == 0) & (df['wave'].shift(1) == 0) & (df['wave'].shift(2) == 0) & (df['wave'].shift(3) == -1),
        'long', df['signal'])

    # 平仓
    df['signal'] = np.where((df['signal'] == 'wait') & (df['wave'] == -1) & (df['wave'].shift(1) == 0), 'close_long',
                            df['signal'])
    df['signal'] = np.where((df['signal'] == 'wait') & (df['wave'] == 1) & (df['wave'].shift(1) == 0), 'close_short',
                            df['signal'])

    return df.iloc[-1]


filename = 'BitMEX-170901-190606-1H'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)
df = df.astype(float)

datas = df.values

backtest = BmBackTest({
    'asset': 1
})

level = 1

for i in range(750, len(df)):
    test_df = datas[i - 750: i]
    row = analysis(test_df)

    if row['signal'] in ['long', 'short'] and backtest.side == 'wait':
        amount = int(backtest.asset * row['close'] * level)
        backtest.create_order(row['signal'], "market", row['close'], amount)
        # backtest.stop_price = row['%s_stop_price' % row['signal']]
    elif row['signal'] in ['long', 'short'] and backtest.side != 'wait' and backtest.side != row['signal']:
        amount = int(backtest.asset * row['close'] * level)
        backtest.create_order(row['signal'], "market", row['close'], amount)

    elif row['signal'] == 'trend' or (row['signal'] == 'close_long' and backtest.side == 'long') or (
            row['signal'] == 'close_short' and backtest.side == 'short'):
        backtest.close_positions(row['close'], 'market')

    else:
        backtest.add_data(row['close'], row['high'], row['low'])

    print(i, row['close'], row['signal'], backtest.open_price, backtest.side, backtest.asset, backtest.float_profit,
          backtest.asset + backtest.float_profit, row['wave'])

backtest.show("Wave guess wn:%d, hs:%d" % (wn, hs))
