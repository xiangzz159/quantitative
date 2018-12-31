#！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/9/16 10:12

@desc:

'''

from tools import data2df, wave_guess
import pandas as pd
import time
import numpy as np
from tools.stockstats import StockDataFrame

df = data2df.csv2df('BTC2017-09-01-now-4H.csv')
df = df.astype(float)
df['Timestamp'] = df['Timestamp'].astype(int)
stock = StockDataFrame.retype(df)

df['date'] = df['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
df['date'] = pd.to_datetime(df['date'])

stock['cci']
stock['stoch_rsi']
# 去掉前几个cci指标不准数据
df = df[5:]

df['ma5'] = np.round(df['close'].rolling(window=5, center=False).mean(), 2)
df['v_ma5'] = np.round(df['volume'].rolling(window=5, center=False).mean(), 2)


# arr1 = pd.Series(df['close'].values)
# wave_guess.wave_guess(arr1)
#
# arr2 = pd.Series(df['ma5'].values)
# wave_guess.wave_guess(arr2)
#
# arr3 = pd.Series(df['v_ma5'].values)
# wave_guess.wave_guess(arr3)

arr4 = pd.Series(df['cci'].values)
wave_guess.wave_guess(arr4)