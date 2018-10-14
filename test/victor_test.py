#！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/10/11 10:14

@desc:

'''

import pandas as pd
import numpy as np
from tools import data2df
import time
from tools.stockstats import StockDataFrame
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

filename = 'BTC2018-01-01-now-4H'
k_data = data2df.csv2df(filename + '.csv')
k_data = k_data.astype(float)
k_data['Timestamp'] = k_data['Timestamp'].astype(int)
stock = StockDataFrame.retype(k_data)

k_data['date'] = k_data['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
k_data['date'] = pd.to_datetime(k_data['date'])

stock['middle_hl']

after_fenxing = pd.DataFrame()
temp_data = k_data.iloc[0]
zoushi = []    # 0:持平 -1:向下 1:向上

for i, row in k_data.iterrows():
    # 第一根包含第二根
    case1_1 = temp_data['high'] > row['high'] and temp_data['low'] < row['low']
    case1_2 = temp_data['high'] > row['high'] and temp_data['low'] == row['low']
    case1_3 = temp_data['high'] == row['high'] and temp_data['low'] < row['low']
    # 第二根包含第一根
    case2_1 = temp_data['high'] < row['high'] and temp_data['low'] > row['low']
    case2_2 = temp_data['high'] < row['high'] and temp_data['low'] == row['low']
    case2_3 = temp_data['high'] == row['high'] and temp_data['low'] > row['low']
    # 第一根等于第二根
    case3 = temp_data['high'] == row['high'] and temp_data['low'] == row['low']
    # 向下趋势
    case4 = temp_data['high'] > row['high'] and temp_data['low'] > row['low']
    # 向上趋势
    case5 = temp_data['high'] < row['high'] and temp_data['low'] < row['low']

    if case1_1 or case1_2 or case1_3:
        if zoushi[-1] == -1:
            temp_data['high'] = row['high']
        else:
            temp_data['low'] = row['low']
    elif case2_1 or case2_2 or case2_3:
        temp_temp = temp_data
        temp_data = row
        if zoushi[-1] == -1:
            temp_data['high'] = temp_temp['high']
        else:
            temp_data['low'] = temp_temp['low']
    elif case3:
        zoushi.append(0)
        pass
    elif case4:
        zoushi.append(-1)
        # 使用默认index: ignore_index=True:
        after_fenxing = pd.concat([after_fenxing, temp_data.to_frame().T], ignore_index=True)
        temp_data = row
    elif case5:
        zoushi.append(1)
        after_fenxing = pd.concat([after_fenxing, temp_data.to_frame().T], ignore_index=True)
        temp_data = row
# print(after_fenxing[['high', 'open', 'close', 'low']].head(50))

# 因为使用candlestick2函数，要求输入open、close、high、low。为了美观，处理k线的最大最小值、开盘收盘价，之后k线不显示影线。
for i, row in after_fenxing.iterrows():
    if row['open'] > row['close']:
        row['open'] = row['high']
        row['close'] = row['low']
    else:
        row['open'] = row['low']
        row['close'] = row['high']
# print(after_fenxing[['high', 'open', 'close', 'low']].head(50))

# 画出k线
# fig, ax = plt.subplots(figsize=(50, 20))
# fig.subplots_adjust(bottom = 0.2)
# mpf.candlestick2_ochl(ax, list(after_fenxing.open), list(after_fenxing.close), list(after_fenxing.high), list(after_fenxing.low), width=0.5, colorup='g', colordown='r', alpha=0.75)
# plt.grid(True)
# # 设置x轴刻度数量
# ax.xaxis.set_major_locator(ticker.MaxNLocator(40))
# ax.set_xticklabels(after_fenxing[::40].date)
# plt.plot(after_fenxing.middle_hl,'k', lw=1)
# # plt.plot(after_fenxing.middle_hl,'ko')
# plt.setp(plt.gca().get_xticklabels(), rotation=45)
# plt.savefig("../data/kline.png")
# # plt.show()

# 找出顶和底
temp_num = 0    # 上一个顶或底的位置
temp_high = 0   # 上一个顶的high值
temp_low = 0    # 上一个底的low值
temp_type = 0   # 上一个记录位置的类型 1-顶分型， 2-底分型
i = 1
fenxing_type = []   # 记录分型点的类型，1为顶分型，-1为底分型
fenxing_time = []   # 记录分型点的时间
fenxing_plot = []   # 记录点的数值，为顶分型去high值，为底分型去low值
fenxing_data = pd.DataFrame()   # 分型点的DataFrame值
interval = 4
while (i < len(after_fenxing) - 1):
    # 顶分型
    case1 = after_fenxing.high[i-1] < after_fenxing.high[i] and after_fenxing.high[i] > after_fenxing.high[i + 1]
    # 底分型
    case2 = after_fenxing.low[i-1] > after_fenxing.low[i] and after_fenxing.low[i] < after_fenxing.low[i + 1]
    if case1:
        # 如果上一个分型为顶分型，则进行比较，选取更高的分型
        if temp_type == 1:
            if after_fenxing.high[i] <= temp_high:
                i += 1
                continue
            else:
                temp_high = after_fenxing.high[i]
                temp_num = i
                temp_type = 1
                i += interval
        # 如果上一个分型为底分型，则记录上一个分型，用当前分型与后面的分型比较，选取通向更极端的分型
        elif temp_type == 2:
            if temp_low >= after_fenxing.high[i]:
                i += 1
            else:
                fenxing_type.append(-1)
                fenxing_time.append(after_fenxing.date[temp_num])
                fenxing_data = pd.concat([fenxing_data, after_fenxing[temp_num : temp_num + 1]],axis=0)
                fenxing_plot.append(after_fenxing.high[i])
                temp_high = after_fenxing.high[i]
                temp_num = i
                temp_type = 1
                i += interval
        else:
            temp_high = after_fenxing.high[i]
            temp_num = i
            temp_type = 1
            i += interval
    elif case2:
        # 如果上一个分型为底分型，则进行比较，选取低点更低的分型
        if temp_type == 2:
            if after_fenxing.low[i] >= temp_low:
                i += 1
                continue
            else:
                temp_low = after_fenxing.low[i]
                temp_num = i
                temp_type = 2
                i += interval
        # 如果上一个分型为顶分型，则记录上一个分型，用当前分型与后面的分型比较，选取通向更极端的分型
        elif temp_type == 1:
            # 如果上一个顶分型的底比当前底分型的底低，则跳过当前的底分型
            if temp_high <= after_fenxing.low[i]:
                i += 1
            else:
                fenxing_type.append(1)
                fenxing_time.append(after_fenxing.date[temp_num])
                fenxing_data = pd.concat([fenxing_data, after_fenxing[temp_num : temp_num + 1]], axis=0)
                fenxing_plot.append(after_fenxing.low[i])
                temp_low = after_fenxing.low[i]
                temp_num = i
                temp_type = 2
                i += interval
        else:
            temp_low = after_fenxing.low[i]
            temp_num = i
            temp_type = 2
            i += interval
    else:
        i += 1

print(len(fenxing_type), fenxing_type)
print('*' * 50)
print(len(fenxing_time), fenxing_time)
print('*' * 50)
print(len(fenxing_plot), fenxing_plot)
print('*' * 50)
print(len(fenxing_data), fenxing_data[['date', 'timestamp', 'open', 'close', 'high', 'low']])
print('*' * 50)

ding_fenxing = []
for i in range(len(fenxing_type)):
    if fenxing_type[i] == 1:
        ding_fenxing.append(fenxing_plot[i])
print(len(ding_fenxing), ding_fenxing)
print('*' * 50)


# 画出k线
fig, ax = plt.subplots(figsize=(20, 10))
# 设置x轴刻度数量
ax.xaxis.set_major_locator(ticker.MaxNLocator(40))
ax.set_xticklabels(after_fenxing[::35].date)
ax.autoscale_view()
plt.plot(fenxing_plot,'k', lw=1)
plt.plot(fenxing_plot,'o')
plt.grid(True)
plt.setp(plt.gca().get_xticklabels(), rotation=30)
plt.show()





