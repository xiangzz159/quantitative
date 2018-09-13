# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/9/13 9:41

@desc: 均线交叉策略
* 当短期均线越过长期均线时，交易金融资产。
* 当短期均线再一次越过长期均线时，结束交易
当短期均线高于长期均线时，我们应进行多头交易，当短期均线再次越过（低于）长期均线时，结束此类交易。当短期均线低于长期均线时，我们应进行空头交易，当短期均线再次越过（高于）长期均线时，结束此类交易。
也就是说，如果短期均线高于长期均线，那么这是一个牛市行情（牛市规则），如果短期均线低于长期均线，则目前为熊市行情（熊市规则）

'''

import pandas as pd
import numpy as np
from tools import data_transform
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time

df = data_transform.transform('BTC2016-now-4H.csv')

df['MA20'] = np.round(df['Close'].rolling(window=20, center=False).mean(), 2)
df['MA50'] = np.round(df['Close'].rolling(window=50, center=False).mean(), 2)

df['MA20-MA50'] = df['MA20'] - df['MA50']

# print(df.tail())

df['Regime'] = np.where(df['MA20-MA50'] > 0, 1, 0)
df['Regime'] = np.where(df['MA20-MA50'] < 0, -1, df['Regime'])

df[['Regime']].plot(ylim=(-2, 2), figsize=(50, 25), grid=True).axhline(y=0, color="black", lw=2)
plt.show()

# print(df['Regime'].value_counts())
'''
 1    7057
-1    4722
 0      52
'''
