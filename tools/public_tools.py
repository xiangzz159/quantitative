# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/9/28 8:46

@desc:

'''

import time
import numpy as np
from functools import wraps


# K线拟合：1h->4h
def kline_fitting_1H_4H(kline):
    # 毫秒时间戳转化为秒
    begin_time = kline[0][0] / 1000 if len(str(kline[0][0])) == 13 else kline[0][0]
    begin_day = int(begin_time - begin_time % 14400) + 14400
    index = 0
    for i in range(len(kline)):
        t = kline[i][0] / 1000 if len(str(kline[0][0])) == 13 else kline[i][0]
        if int(t) == begin_day:
            index = i

    kline = kline[index:]
    num = len(kline) % 4
    kline_ = kline[len(kline) - num:]
    kline = kline[:len(kline) - num]

    l = []
    # 存在第4小时k线没走完的情况，例如00：00 - 03：02，这时候03：00-04：00仍然有k线数据
    for i in range(0, len(kline), 4):
        timestamp = int(kline[i][0] / 1000) if len(str(kline[0][0])) == 13 else kline[i][0]
        open = kline[i][1]
        high = max(kline[i][2], kline[i + 1][2], kline[i + 2][2], kline[i + 3][2])
        low = min(kline[i][3], kline[i + 1][3], kline[i + 2][3], kline[i + 3][3])
        close = kline[i + 3][4]
        volumn = kline[i][5] + kline[i + 1][5] + kline[i + 2][5] + kline[i + 3][5]
        l.append([timestamp, open, high, low, close, volumn])
    if len(kline_) > 2:
        timestamp = int(kline_[0][0] / 1000) if len(str(kline_[0][0])) == 13 else kline_[0][0]
        open = kline_[0][1]
        close = kline_[len(kline_) - 1][4]
        volumn = 0
        high = 0
        low = 2 << 16
        for k in kline_:
            volumn += k[5]
            high = max(high, k[2])
            low = min(low, k[3])
        l.append([timestamp, open, high, low, close, volumn])
    return l


# K线拟合
def kline_fitting(kline, n, fitting_time):
    # 毫秒时间戳转化为秒
    begin_time = kline[0][0] / 1000 if len(str(kline[0][0])) == 13 else kline[0][0]
    begin_day = int(begin_time - begin_time % fitting_time) + fitting_time
    index = 0
    for i in range(len(kline)):
        t = kline[i][0] / 1000 if len(str(kline[0][0])) == 13 else kline[i][0]
        if int(t) == begin_day:
            index = i
            break

    kline = kline[index:]
    num = len(kline) % n
    kline = kline[:len(kline) - num]

    l = []
    for i in range(0, len(kline), n):
        timestamp = int(kline[i][0] / 1000) if len(str(kline[0][0])) == 13 else kline[i][0]
        open = kline[i][1]
        high = kline[i][2]
        for j in range(i + 1, i + n):
            high = max(high, kline[j][2])
        low = kline[i][3]
        for j in range(i + 1, i + n):
            low = min(low, kline[j][3])
        close = kline[i + n - 1][4]
        volumn = 0
        for j in range(i, i + n):
            volumn += kline[j][5]
        l.append([timestamp, open, high, low, close, volumn])
    return l


def get_time():
    timestamp = time.time() + 8 * 3600  # 处理时间戳的时区
    time_local = time.localtime(timestamp)
    return time.strftime("%Y-%m-%d %H:%M", time_local)


def floatrange(start, stop, num, decimal=5):
    return [round(start + float(i) * num, decimal) for i in range(int((stop - start) / num))]


def uniform_random(min, max):
    random = np.random.RandomState()
    return random.uniform(min, max)


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" %
              (function.__name__, str(t1 - t0))
              )
        return result

    return function_timer


if __name__ == '__main__':
    import csv
    import pandas as pd

    lines = list(csv.reader(open(r'../data/BitMEX-BTCUSD-20190611-20200125-1h.csv')))
    header, values = lines[0], lines[1:]
    data_dict = {h: v for h, v in zip(header, zip(*values))}
    df = pd.DataFrame(data_dict)
    df = df.astype(float)
    # size = len(df)
    # df = df[size - 100:]
    l = df.values.tolist()
    l_ = kline_fitting_1H_4H(l)

    df = pd.DataFrame(l_, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df.to_csv('../data/BitMEX-BTCUSD-20190611-20200125-4h.csv', index=False)
