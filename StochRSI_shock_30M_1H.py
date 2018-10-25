# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/10/24 11:26

@desc: StochRSI 短期震荡策略 15M-30M和30M-1H周期

'''

import pandas as pd
import numpy as np
from tools.stockstats import StockDataFrame
import time
import ccxt
from tools import public_tools

# 参数
dk, d, cci, rsi, compare_cci = 0, 15, -85, 20, 80
dk_, d_, cci_, rsi_ = 0, 85, 85, 80

ex = ccxt.bitmex({
    'timeout': 60000
})


def is_true(df, compare_df, compare_time):
    for index, row in df.loc[df['regime'] == 1].iterrows():
        t = row['timestamp'] - row['timestamp'] % compare_time - compare_time
        compare_row = compare_df.loc[compare_df['timestamp'] == t].iloc[0]
        if compare_row['cci'] <= compare_cci and compare_row['cci'] >= -1 * compare_cci:
            continue
        else:
            df['regime'][index] = 'wait'
    return df


def anyasis(k, compare_k, compare_time):
    df = pd.DataFrame(k, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    compare_df = pd.DataFrame(compare_k, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    t = k[0][0]
    if len(str(t)) == 13:
        df['timestamp'] = df['timestamp'] / 1000
    t = compare_k[0][0]
    if len(str(t)) == 13:
        compare_df['timestamp'] = compare_df['timestamp'] / 1000
    stock = StockDataFrame.retype(df)
    df['date'] = df['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
    df['date'] = pd.to_datetime(df['date'])
    stock['cci']
    stock['stoch_rsi']
    # 震荡行情判断
    compare_stock = StockDataFrame.retype(compare_df)
    compare_df['date'] = compare_df['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
    compare_df['date'] = pd.to_datetime(compare_df['date'])
    compare_stock['cci']
    compare_stock['stoch_rsi']

    # 做多时机
    df['regime'] = np.where(
        (df['stoch_k'] - df['stoch_k'].shift(1) + df['stoch_d'].shift(1) - df['stoch_d'] > dk) & (
                df['stoch_d'] < d) & (
                df['cci'] < cci) | (
                df['stoch_k'] - df['stoch_k'].shift(1) + df['stoch_d'].shift(1) - df['stoch_d'] > dk) & (
                df['stoch_rsi'] < rsi) & (
                df['cci'] < cci), 'long', 'wait')
    # 做空时机
    df['regime'] = np.where(
        (df['stoch_k'] - df['stoch_k'].shift(1) + df['stoch_d'].shift(1) - df['stoch_d'] > dk_) & (
                df['stoch_d'] > d_) & (
                df['cci'] > cci_) | (
                df['stoch_k'] - df['stoch_k'].shift(1) + df['stoch_d'].shift(1) - df['stoch_d'] > dk_) & (
                df['stoch_rsi'] > rsi_) & (
                df['cci'] > cci_), 'short', df['regime'])

    df = is_true(df, compare_df, compare_time)

    df['regime'] = np.where(df['regime'].shift(1) == 'long', 'close_long', df['regime'])
    df['regime'] = np.where(df['regime'].shift(1) == 'short', 'close_short', df['regime'])

    re = df.iloc[-1]

    # df[['date', 'open', 'high', 'low', 'close', 'regime', 'volume']].loc[(df['regime'] != 'wait')].to_csv('result-30M.csv',
    #                                                                                             index=None)
    if re['regime'] != 'wait':
        print(df[['date', 'close', 'regime']].tail(10))
    return re


def run():
    limit = 750
    since = int(time.time()) * 1000 - 300000 * (limit - 1)
    k_5m = ex.fetch_ohlcv('BTC/USD', '5m', since, limit)
    k_30m = public_tools.kline_fitting(k_5m, 6, 1800)
    k_1H = public_tools.kline_fitting(k_5m, 12, 3600)
    re = anyasis(k_30m, k_1H, 3600)


if __name__ == '__main__':
    while True:
        run()
        time.sleep(60 * 5)
