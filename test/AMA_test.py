# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2019/4/2 15:57

@desc:

'''

from tools import data2df
import numpy as np
from tools.BmBackTest import BmBackTest
import pandas as pd


def analysis(datas):
    df = pd.DataFrame(datas, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    n = 15
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


def analysis_(datas):
    df = pd.DataFrame(datas, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    n = 15
    percentage = 0.1
    atr_length = 5
    natr_stop = 4

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
    # ATR计算
    df['tr'] = df['high'] - df['low']
    df['tr'] = np.where(df['tr'] >= abs(df['high'] - df['close'].shift(1)), df['tr'],
                        abs(df['high'] - df['close'].shift(1)))
    df['tr'] = np.where(df['tr'] >= abs(df['low'] - df['close'].shift(1)), df['tr'],
                        abs(df['low'] - df['close'].shift(1)))
    df['atr'] = df['tr'].rolling(window=atr_length, center=False).mean()

    df['signal'] = 'wait'
    df['signal'] = np.where((df['signal'] == 'wait') & (
            (df['ama'] - df['ama'].shift(1) > df['filter']) | (df['ama'] - df['ama'].shift(2) > df['filter'])),
                            'long', df['signal'])
    df['signal'] = np.where((df['signal'] == 'wait') & ((df['ama'] - df['ama'].shift(1) < -1 * df['filter']) | (
            df['ama'] - df['ama'].shift(2) < -1 * df['filter'])), 'short', df['signal'])
    signal_df = df.loc[df['signal'] != 'wait']
    signal_df['signal'] = np.where(signal_df['signal'] == signal_df['signal'].shift(1), 'wait', signal_df['signal'])
    signal_df = signal_df.loc[signal_df['signal'] != 'wait']

    df['hp'] = 0
    df['lp'] = 0
    for i in range(1, len(signal_df)):
        row = signal_df.iloc[i]
        row_ = signal_df.iloc[i - 1]
        part_df = df.loc[(df['timestamp'] >= row_['timestamp']) & (df['timestamp'] < row['timestamp'])]
        max_price = max(part_df['high'])
        min_price = min(part_df['low'])
        df['hp'] = np.where((df['timestamp'] >= row_['timestamp']) & (df['timestamp'] < row['timestamp']), max_price,
                            df['hp'])
        df['lp'] = np.where((df['timestamp'] >= row_['timestamp']) & (df['timestamp'] < row['timestamp']), min_price,
                            df['lp'])

    row = signal_df.iloc[-1]
    part_df = df.loc[df['timestamp'] >= row['timestamp']]
    max_price = max(part_df['high'])
    min_price = min(part_df['low'])
    df['hp'] = np.where(df['timestamp'] >= row['timestamp'], max_price, df['hp'])
    df['lp'] = np.where(df['timestamp'] >= row['timestamp'], min_price, df['lp'])

    ls = []
    ss = []
    df['signal'] = 'wait'
    for idx, row in signal_df.iterrows():
        if row['signal'] == 'long':
            ls.append(idx)
        elif row['signal'] == 'short':
            ss.append(idx)
    if len(ls) > 0:
        df.loc[ls, 'signal'] = 'long'
    if len(ss) > 0:
        df.loc[ss, 'signal'] = 'short'

    df['signal'] = np.where((df['signal'] == 'wait') & (df['low'] <= df['hp'] - df['atr'] * natr_stop), 'close_long',
                            df['signal'])
    df['signal'] = np.where((df['signal'] == 'wait') & (df['high'] >= df['lp'] + df['atr'] * natr_stop), 'close_short',
                            df['signal'])
    df['signal'] = np.where(df['signal'] == df['signal'].shift(1), 'wait', df['signal'])
    return df.iloc[-1]


filename = 'BitMEX-170901-190606-4H'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)
df = df.astype(float)

datas = df.values

backtest = BmBackTest({
    'asset': 1
})

level = 1

for i in range(370, len(df)):
    test_data = datas[i - 370: i]
    row = analysis(test_data)

    if row['signal'] in ['long', 'short']:
        amount = int(backtest.asset * row['close'] * level)
        backtest.create_order(row['signal'], "market", row['close'], amount)
        # backtest.stop_price = row['%s_stop_price' % row['signal']]

    elif (row['signal'] == 'close_long' and backtest.side == 'long') or (
            row['signal'] == 'close_short' and backtest.side == 'short'):
        backtest.close_positions(row['close'], 'market')

    else:
        backtest.add_data(row['close'], row['high'], row['low'])

    print(i, row['close'], row['signal'], backtest.asset, backtest.float_profit, backtest.asset + backtest.float_profit)

# backtest.show("AMA")
backtest.real_time_show("AMA")
