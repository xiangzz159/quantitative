#！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/10/12 15:55

@desc:

'''

import pandas as pd
from tools.stockstats import StockDataFrame
import time
import ccxt


def k_analysis(k):
    # 转DataFrame格式
    k_data = pd.DataFrame(k, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    t = k[0][0]
    # 毫秒级
    if len(str(t)) == 13:
        k_data['timestamp'] = k_data['timestamp'] / 1000
    stock = StockDataFrame.retype(k_data)
    stock['middle_hl']
    k_data['date'] = k_data['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
    k_data['date'] = pd.to_datetime(k_data['date'])

    after_fenxing = pd.DataFrame()
    temp_data = k_data.iloc[0]
    zoushi = []  # 0:持平 -1:向下 1:向上

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

    # 因为使用candlestick2函数，要求输入open、close、high、low。为了美观，处理k线的最大最小值、开盘收盘价，之后k线不显示影线。
    for i, row in after_fenxing.iterrows():
        if row['open'] > row['close']:
            row['open'] = row['high']
            row['close'] = row['low']
        else:
            row['open'] = row['low']
            row['close'] = row['high']

    # 找出顶和底
    temp_num = 0  # 上一个顶或底的位置
    temp_high = 0  # 上一个顶的high值
    temp_low = 0  # 上一个底的low值
    temp_type = 0  # 上一个记录位置的类型 1-顶分型， 2-底分型
    i = 1
    fenxing_type = []  # 记录分型点的类型，1为顶分型，-1为底分型
    fenxing_time = []  # 记录分型点的时间
    fenxing_plot = []  # 记录点的数值，为顶分型去high值，为底分型去low值
    fenxing_data = pd.DataFrame()  # 分型点的DataFrame值
    interval = 4
    while (i < len(after_fenxing) - 1):
        # 顶分型
        case1 = after_fenxing.high[i - 1] < after_fenxing.high[i] and after_fenxing.high[i] > after_fenxing.high[i + 1]
        # 底分型
        case2 = after_fenxing.low[i - 1] > after_fenxing.low[i] and after_fenxing.low[i] < after_fenxing.low[i + 1]
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
                    fenxing_data = pd.concat([fenxing_data, after_fenxing[temp_num: temp_num + 1]], axis=0)
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
                    fenxing_data = pd.concat([fenxing_data, after_fenxing[temp_num: temp_num + 1]], axis=0)
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
    return fenxing_type, fenxing_time, fenxing_plot, fenxing_data

def analysis(k):
    fenxing_type, fenxing_time, fenxing_plot, fenxing_data = k_analysis(k)
    if len(fenxing_type) > 7:
        if fenxing_type[0] == -1 and len(fenxing_type) > 7:
            location_1 = [i for i, a in enumerate(fenxing_type) if a == 1] # 找出1在列表中的所有位置
            location_2 = [i for i, a in enumerate(fenxing_type) if a == -1] # 找出-1在列表中的所有位置
            # 线段破坏
            case1 = fenxing_data.low[location_2[0]] > fenxing_data.low[location_2[1]]
            # 线段形成
            case2 = fenxing_data.high[location_1[1]] < fenxing_data.high[location_1[2]] < fenxing_data.high[
                location_1[3]]
            case3 = fenxing_data.low[location_2[1]] < fenxing_data.low[location_2[2]] < fenxing_data.low[
                location_2[3]]
            # 第i笔中底比第i+2笔顶高(扶助判断条件，根据实测加入之后条件太苛刻，很难有买入机会)
            case4 = fenxing_data.low[location_2[1]] > fenxing_data.high[location_1[3]]
            if case1 and case2 and case3:
                # 买入
                print('create long order')



if __name__ == '__main__':
    # 1. 获取k线数据
    ex = ccxt.bitmex()
    limit = 750
    since = int(time.time()) * 1000 - 3600000 * (limit - 1)
    kline = ex.fetch_ohlcv('BTC/USD', '1h', since, limit)

    # timestamp(ms), open, high, low, close, volumn

    fenxing_type, fenxing_time, fenxing_plot, fenxing_data = k_analysis(kline)
    print(len(fenxing_type), fenxing_type)
    print('*' * 50)
    print(len(fenxing_time), fenxing_time)
    print('*' * 50)
    print(len(fenxing_plot), fenxing_plot)
    print('*' * 50)
    print(len(fenxing_data), fenxing_data[['date', 'timestamp', 'open', 'close', 'high', 'low']])
    print('*' * 50)

    analysis(kline)




