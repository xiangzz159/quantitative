#！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/9/16 10:12

@desc:

'''

from tools import data_transform
import pandas as pd
import time
from job import wave_guess
import numpy as np

df = data_transform.transform('BTC2016-now-1D.csv')
df['ts'] = df['Timestamp'].apply(lambda x: time.strftime("%Y--%m--%d", time.localtime(int(x))))
# 转换datetime格式
df['ts'] = pd.to_datetime(df['ts'])

df['ma5'] = np.round(df['Close'].rolling(window=5, center=False).mean(), 2)
df['v_ma5'] = np.round(df['Volume'].rolling(window=5, center=False).mean(), 2)


arr1 = pd.Series(df['Close'].values)
wave_guess.wave_guess(arr1)

arr2 = pd.Series(df['ma5'].values)
wave_guess.wave_guess(arr2)

arr3 = pd.Series(df['v_ma5'].values)
wave_guess.wave_guess(arr3)