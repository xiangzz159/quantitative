#！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/11/19 17:23

@desc:

'''
from tools.stockstats import StockDataFrame
import time
import pandas as pd
import ccxt
from tools import public_tools
import numpy as np

def analysis(k):
    # 参数设置
    hist_ema, hist_signal_ma, hist_signal_ma_, ii, jj = 3, 3, 3, 1, 0

    # 转DataFrame格式
    df = pd.DataFrame(k, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    t = k[0][0]
    # 毫秒级
    if len(str(t)) == 13:
        df['timestamp'] = df['timestamp'] / 1000
    # 分析
    stock = StockDataFrame.retype(df)

    hist_ema_name = 'hist_%d_ema' % hist_ema
    hist_signal_ma_name = 'hist_signal_%d_ma' % hist_signal_ma
    hist_signal_ma1_name = 'hist_signal_%d_ma_' % hist_signal_ma_
    stock['macd']
    stock[hist_ema_name]
    df[hist_signal_ma_name] = np.round(df['hist_signal'].rolling(min_periods=1, window=hist_signal_ma).mean(), 2)
    df[hist_signal_ma1_name] = np.round(df[hist_signal_ma_name].rolling(min_periods=1, window=hist_signal_ma_).mean(),
                                        2)
    df[hist_ema_name] = np.round(df[hist_ema_name], 2)

    df['date'] = df['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
    df['date'] = pd.to_datetime(df['date'])
    # 去掉前面100条指标不准数据
    df = df[100:]

    df['signal'] = np.where(
        ((abs(df[hist_ema_name]) >= ii) & (df[hist_signal_ma_name] < jj)) & ((
                (df[hist_signal_ma_name] >= -1) & (df[hist_signal_ma_name].shift(1) < -1))), 'long', 'wait')
    df['signal'] = np.where((df['hist'].shift(1) < 0) & (df['hist'] > 0), 'long', df['signal'])

    df['signal'] = np.where(
        ((abs(df[hist_ema_name]) >= ii) & (df[hist_signal_ma_name] >= jj)) & ((
                (df[hist_signal_ma_name] <= 1) & (df[hist_signal_ma_name].shift(1) > 1))), 'short', df['signal'])
    df['signal'] = np.where((df['hist'].shift(1) > 0) & (df['hist'] < 0), 'short', df['signal'])

    df_signal = df.loc[df['signal'] != 'wait']
    # 筛选出重复信号
    unless_signal = df_signal.loc[
        ((df_signal['signal'] == 'short') & (df_signal['signal'].shift(1) == 'short')) | (
                (df_signal['signal'] == 'long') & (df_signal['signal'].shift(1) == 'long'))]
    # 过滤重复信号
    for index, row in unless_signal.iterrows():
        df.signal[index] = 'wait'

    last_serise = df.iloc[-1]

    high = max(df[-6:].high)
    low = min(df[-6:].low)
    return last_serise, high, low


ex = ccxt.bitmex()
limit = 750
symbol = 'BTC/USD'
while True:
    since = int(time.time()) * 1000 - 3600000 * (limit - 1)
    k_1H = ex.fetch_ohlcv(symbol, '1h', since, limit)
    k_4H = public_tools.kline_fitting_1H_4H(k_1H)
    series, high, low = analysis(k_4H)
    if series['signal'] == 'long':
        print(public_tools.get_time(), symbol, 'make long trade, now price is', series['close'])
    elif series['signal'] == 'short':
        print(public_tools.get_time(), symbol, 'make short trade, now price is', series['close'])

    time.sleep(100)