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

filename = 'BTC2017-09-01-now-30M'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)
df['Timestamp'] = df['Timestamp'].astype(int)

if df['Timestamp'][0] % 3600 != 0:
    df = df[1:]

l = len(df) - len(df) % 2
ll = []
for i in range(1, l, 2):
    row_ = df.iloc[i - 1]
    row = df.iloc[i]
    timestamp = row_['Timestamp']
    open = row_['Open']
    close = row_['Close']
    high = max(row_['High'], row['High'])
    low = min(row_['Low'], row['Low'])
    vol = row_['Volume'] + row['Volume']
    adj =(row_['Adj Close'] + row['Adj Close']) / 2
    ll.append([timestamp, high, low, open, close, vol, adj])

new_df = pd.DataFrame(ll, columns=['Timestamp','High','Low','Open','Close','Volume','Adj Close'])
fileName = '../data/BTC2017-09-01-now-1H.csv'
new_df.to_csv(fileName, index=None)