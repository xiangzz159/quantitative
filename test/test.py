# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/9/21 18:03

@desc:

'''
# coding: utf-8
import pandas as pd
import numpy as np #computing multidimensionla arrays
import datetime
import time
from tools import data2df
from tools.stockstats import StockDataFrame

# StochasticRSI Function
def Stoch(close,high,low, smoothk, smoothd, n):
    lowestlow = pd.Series.rolling(low,window=n,center=False).min()
    highesthigh = pd.Series.rolling(high, window=n, center=False).max()
    K = pd.Series.rolling(100*((close-lowestlow)/(highesthigh-lowestlow)), window=smoothk).mean()
    D = pd.Series.rolling(K, window=smoothd).mean()
    return K, D


df = data2df.csv2df('BTC2018-09-19-now-30M.csv')
df = df.astype(float)
df['Timestamp'] = df['Timestamp'].astype(int)
stock = StockDataFrame.retype(df)
df['date'] = df['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
df['date'] = pd.to_datetime(df['date'])

# stock['cci']
stock['stoch_rsi']

# df['stoch_k'], df['stoch_d'] = Stoch(stock.rsi_14, stock.rsi_14, stock.rsi_14, 3, 3, 14)

print(df.dtypes)
print(df[['date', 'close', 'stoch_  rsi', 'stoch_k', 'stoch_d']].tail(50))
