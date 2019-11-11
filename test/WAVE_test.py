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
    # 取timestamp和close列
    high_list = arr[:, [0, 2]]
    low_list = arr[:, [0, 3]]
    # 计算最大的N个值，认为是波峰
    wave_crest = heapq.nlargest(wn, high_list, key=lambda x: x[1])
    # 计算最小的N个值，认为是波谷
    wave_base = heapq.nsmallest(wn, low_list, key=lambda x: x[1])
    return wave_crest, wave_base


def analysis(kline):
    df = pd.DataFrame(kline, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['signal'] = 'wait'

    # TODO 参数优化
    wn = 10  # 取wn个最大最小值
    hs = -300  # 截取的时间周期
    stop_trade_times = 8  # 停止交易时间

    df = df[hs:]
    wave_crest, wave_base = wave_guess(df.values, wn)
    df.index = df.timestamp
    hts = []
    lts = []
    for x in wave_crest:
        hts.append(x[0])
    for x in wave_base:
        lts.append(x[0])

    mid_price = (wave_base[0][1] + wave_crest[0][1] + wave_base[-1][1] + wave_crest[-1][1]) / 4
    # print(wave_base)
    # print(wave_crest)

    df.loc[[wave_crest[0][0]], 'signal'] = 'close_short'
    df.loc[[wave_base[0][0]], 'signal'] = 'close_long'
    for i in range(stop_trade_times):
        df['signal'] = np.where((df['signal'] == 'wait') & (df['signal'].shift(1 + i) == 'close_short'), 'close_short',
                                df['signal'])
        df['signal'] = np.where((df['signal'] == 'wait') & (df['signal'].shift(1 + i) == 'close_long'), 'close_long',
                                df['signal'])

    df['signal'] = np.where(
        (df['signal'] == 'wait') & (df['close'] > wave_crest[-1][1]) & (df['close'] < wave_crest[0][1]) & (
                df['high'] < wave_crest[0][1]), 'short', df['signal'])

    df['signal'] = np.where(
        (df['signal'] == 'wait') & (df['close'] < wave_base[-1][1]) & (df['close'] > wave_base[0][1]) & (
                df['low'] > wave_base[0][1]), 'long', df['signal'])

    # df['signal'] = np.where((df['signal'] == 'wait') & (df['close'].shift(1) < mid_price) & (df['close'] >= mid_price),
    #                         'close_short', df['signal'])
    # df['signal'] = np.where((df['signal'] == 'wait') & (df['close'].shift(1) > mid_price) & (df['close'] <= mid_price),
    #                         'close_long', df['signal'])

    return df.iloc[-1]


# filename = 'BitMEX-170901-190606-5M'
filename = 'BitMEX-170901-190606-15M'
# filename = 'BitMEX-170901-190606-30M'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)

datas = df.values

backtest = BmBackTest({
    'asset': 1
})

level = 1

# for i in range(750, len(df)):
for i in range(750, 10000):
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
          backtest.asset + backtest.float_profit)

backtest.show("Wave guess")
