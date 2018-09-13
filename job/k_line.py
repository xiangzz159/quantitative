# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/9/12 21:00

@desc: K线图

'''

import datetime
import matplotlib
from matplotlib.dates import DateFormatter, WeekdayLocator, \
    DayLocator, MONDAY
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from mpl_finance import candlestick2_ochl
import time
import numpy as np
from tools import data_transform

df = data_transform.transform('BTC2018-08-15-now-2H.csv')

# Ma5, MA20, MA50, MA200 均线
df['MA5'] = np.round(df['Close'].rolling(window=5, center=False).mean(), 2)
df['MA20'] = np.round(df['Close'].rolling(window=20, center=False).mean(), 2)
df['MA50'] = np.round(df['Close'].rolling(window=50, center=False).mean(), 2)
df['MA200'] = np.round(df['Close'].rolling(window=200, center=False).mean(), 2)

# print(type(btc))
# print(btc.head())

df = df.sort_index(0)
y_min = float(min(df['Low'].values)) * 0.9
y_max = float(max(df['High'].values)) * 1.1
df_idx = df.values

fig, ax = plt.subplots(figsize=(50, 25))  # 设置图片大小

# https://matplotlib.org/api/finance_api.html#module-matplotlib.finance
candlestick2_ochl(ax=ax,
                  opens=df["Open"].values, closes=df["Close"].values,
                  highs=df["High"].values, lows=df["Low"].values,
                  width=0.75, colorup='g', colordown='r', alpha=0.75)

# 展示MA5, MA20
df[['MA5', 'MA20', 'MA50', 'MA200']].plot(figsize=(50, 25), grid=True)

# x轴坐标数量
ax.xaxis.set_major_locator(ticker.MaxNLocator(40))


# 设置自动格式化时间。
def mydate_formatter(x, pos):
    try:
        ts = int(df_idx[int(x)][0])
        timeArray = time.localtime(ts)
        otherStyleTime = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
        return otherStyleTime
    except IndexError:
        return ''


ax.xaxis.set_major_formatter(ticker.FuncFormatter(mydate_formatter))

ax.yaxis.set_major_locator(ticker.MaxNLocator(40))

plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
ax.grid(True)
plt.title("BTC 2018-08-15-now-2H KLine")

# 设置y轴坐标范围，解决y轴显示不全问题
plt.ylim(y_min, y_max)

plt.savefig("../data/BTC2018-08-15-now-2H.png")
