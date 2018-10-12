#！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/10/8 20:32

@desc:

'''
import json
import numpy as np
import os
import pandas as pd
from urllib import request
import time
from tools import data2df
from tools.stockstats import StockDataFrame

filename = 'BTC2017-09-01-now-4H'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)
df['Timestamp'] = df['Timestamp'].astype(int)
stock = StockDataFrame.retype(df)

df['date'] = df['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
df['date'] = pd.to_datetime(df['date'])

stock['cci']
stock['stoch_rsi']
# 去掉前几个指标不准数据
df = df[5:]

print(df[['date', 'stoch_d', 'stoch_k', 'stoch_rsi']])
# df['kkdd'] = df['stoch_k'] - df['stoch_k'].shift(1) + df['stoch_d'].shift(1) - df['stoch_d']
# df['dk'] = df['stoch_d'] - df['stoch_k']
#
# print(df[['date', 'cci', 'dk', 'kkdd']].loc[(df['cci'] > 80) & (df['timestamp'] >= 1531584000) & (df['timestamp'] <= 1532505600)])