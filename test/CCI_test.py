# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/9/25 8:37

@desc: CCI 策略回测

'''

from tools import data2df
from stockstats import StockDataFrame
import numpy as np
import pandas as pd
from tools.BmBackTest import BmBackTest


def analysis(datas):
    df = pd.DataFrame(datas, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df.astype(float)
    stock = StockDataFrame.retype(df)
    df['date'] = pd.to_datetime(df['timestamp'], unit='s')

    df.index = df.date
    stock = StockDataFrame.retype(df)
    stock['cci']
    stock['stoch_rsi']
    # 去掉前几个cci指标不准数据
    df = df[10:]

    df['signal'] = 'wait'
    df['signal'] = np.where((df['cci'] >= 100) & (df['cci'].shift(1) < 100), 'long', df['signal'])
    df['signal'] = np.where(
        ((df['cci'] <= 100) & (df['cci'].shift(1) > 100)),
        'close_long', df['signal'])
    df['signal'] = np.where((df['cci'] <= -100) & (df['cci'].shift(1) >= -100), 'short', df['signal'])
    df['signal'] = np.where((df['cci'] >= -100) & (df['cci'].shift(1) <= -100),
                            'close_short', df['signal'])
    return df.iloc[-1]



filename = 'BitMEX-170901-190606-4H'
df = data2df.csv2df(filename + '.csv')
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

backtest.real_time_show("CCI")
