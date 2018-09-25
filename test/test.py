# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/9/21 18:03

@desc:

'''
import pandas as pd
import random

import pandas as pd
import numpy as np
from tools.stockstats import StockDataFrame
from job import poloniex_data_collection
import time
import matplotlib.ticker as ticker
from tools import data2df
import matplotlib.pyplot as plt


def Stoch(close, high, low, smoothk, smoothd, n):
    lowestlow = pd.Series.rolling(low, window=n, center=False).min()
    highesthigh = pd.Series.rolling(high, window=n, center=False).max()
    rsv = 100 * (close - lowestlow) / (highesthigh - lowestlow)
    K = pd.Series([50] * len(rsv))
    K = rsv / 3 + 2 * K.shift(1) / 3
    D = pd.Series([50] * len(rsv))
    D = K / 3 + 2 * D.shift(1) / 3
    return K, D


df = data2df.csv2df('BTC2018-09-15-now-30M.csv')
df = df.astype(float)
df['Timestamp'] = df['Timestamp'].astype(int)
stock = StockDataFrame.retype(df)
df['date'] = df['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
df['date'] = pd.to_datetime(df['date'])

stock['rsi_14']
np_float_data = np.array(df['close'])
myStochRSI = Stoch(df.close, df.high, df.low, 3, 3, 14)
df['rsi_k'], df['rsi_d'] = myStochRSI

# print(df.dtypes)
print(df[['date', 'rsi_14', 'rsi_k', 'rsi_d']].tail(30))
