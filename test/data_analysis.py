# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/9/16 11:07

@desc:

'''
from tools import data_transform
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

df = data_transform.transform('BTC2016-now-1D.csv')
df['ts'] = df['Timestamp'].apply(lambda x: time.strftime("%Y--%m--%d", time.localtime(int(x))))
# 转换datetime格式
df['ts'] = pd.to_datetime(df['ts'])
df['Close'] = df['Close'].astype(float)

df['5d'] = np.round(df['Close'].rolling(window=5, center=False).mean(), 2)
df['10d'] = np.round(df['Close'].rolling(window=10, center=False).mean(), 2)
df['30d'] = np.round(df['Close'].rolling(window=30, center=False).mean(), 2)
df['60d'] = np.round(df['Close'].rolling(window=60, center=False).mean(), 2)

# ax = df[['5d', '10d', '30d', '60d']].plot(figsize=(40, 20), grid=True, xticks=df.index, rot=90)


# 计算收益价格
df['return'] = np.log(df['Close'] / df['Close'].shift(1))
# subplots=True 为分图显示，所以ax类型为numpy.ndarray
# ax = df[['Close', 'return']].plot(figsize=(20, 10), grid=True, xticks=df.index, rot=90, subplots=True, style='b')

# 收益率的移动历史标准差
move_day = int(len(df) / 30)
df['mov_vol'] = df['return'].rolling(window=move_day).std() * math.sqrt(move_day)
ax = df[['Close', 'mov_vol', 'return']].plot(figsize=(20, 10), grid=True, xticks=df.index, rot=90, subplots=True, style='b')


# 设置x轴刻度数量
ax[0].xaxis.set_major_locator(ticker.MaxNLocator(40))
# 以ts为x轴刻度
ax[0].set_xticklabels(df.ts)
# 美观x轴刻度
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()
