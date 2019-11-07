# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2019/8/3 18:32

@desc:
1. 时间周期定为7天，分别取5个波峰和波谷值作为价格上限区间和下限区间
2. 当当前价格在上限区间时，做空；当当前价格在下限区间时，做多。
3. 当当前时间周期最高值大于上限区间最高值或当前时间周期最小值小于下限区间最小值时，平仓并在接下来的12个小时内不再开仓
'''

from tools import data2df
import numpy as np
import pandas as pd
from tools.BmBackTest import BmBackTest
import heapq
from tools.stockstats import StockDataFrame


def wave_guess(arr, wn):
    close_list = arr[:, [0, 4]]

    # 计算最大的N个值，认为是波峰
    # <class 'list'>: [(167, 4343.0), (166, 4335.0), (144, 4329.5), (146, 4329.3), (145, 4327.3)]
    wave_crest = heapq.nlargest(wn, close_list, key=lambda x: x[1])

    # 计算最小的N个值，认为是波谷
    # <class 'list'>: [(1, 3660.4), (0, 3677.0), (2, 3677.5), (3, 3677.8), (11, 3760.8)]
    wave_base = heapq.nsmallest(wn, close_list, key=lambda x: x[1])

    return wave_crest, wave_base


# 波峰波谷猜想
wn = 5
hs = -7 * 24


def analysis(kline):
    data_len = len(kline)
    df = pd.DataFrame(kline, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    stock = StockDataFrame.retype(df)

    df['wave'] = 0
    df = df[hs:]
    data_len_ = len(df)
    wave_crest_x, wave_base_x = wave_guess(df.values, wn)
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


filename = 'BitMEX-170901-191107-1H'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)

datas = df.values

backtest = BmBackTest({
    'asset': 1
})

level = 1

for i in range(750, len(df)):
    test_df = datas[i - 750: i]
    row = analysis(test_df)

    if row['signal'] in ['long', 'short']:
        amount = int(backtest.asset * row['close'] * level)
        backtest.create_order(row['signal'], "market", row['close'], amount)

    elif row['signal'] == 'trend' or (row['signal'] == 'close_long' and backtest.side == 'long') or (
            row['signal'] == 'close_short' and backtest.side == 'short'):
        backtest.close_positions(row['close'], 'market')

    else:
        backtest.add_data(row['close'])

    print(i, row['close'], row['signal'], backtest.asset, backtest.float_profit, backtest.asset + backtest.float_profit)

backtest.show("Wave guess wn:%d, hs:%d" % (wn, hs))
