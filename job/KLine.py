# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/9/12 21:00

@desc: K线图

'''

import csv
import datetime
import matplotlib
from matplotlib.dates import DateFormatter, WeekdayLocator, \
    DayLocator, MONDAY
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from mpl_finance import candlestick2_ochl
import time

lines = list(csv.reader(open(r'../data/BTC2017-now-4H.csv')))
header, values = lines[0], lines[1:]
data_dict = {h: v for h, v in zip(header, zip(*values))}
df = pd.DataFrame(data_dict)

# print(type(btc))
# print(btc.head())

df = df.sort_index(0)
df_idx = df.values

fig, ax = plt.subplots(figsize=(50, 25))  # 设置图片大小

# https://matplotlib.org/api/finance_api.html#module-matplotlib.finance
candlestick2_ochl(ax=ax,
                  opens=df["Open"].values, closes=df["Close"].values,
                  highs=df["High"].values, lows=df["Low"].values,
                  width=0.75, colorup='r', colordown='g', alpha=0.75)

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

ax.yaxis.set_major_locator(ticker.MaxNLocator(20))


plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
ax.grid(True)
plt.title("BTC 2016-now KLine")

# 设置y轴坐标范围，解决y轴显示不全问题
plt.ylim(0, 20000)

plt.savefig("../data/BTC2017-now-4H.png")

