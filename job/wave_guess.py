#！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/9/16 10:01

@desc: 波谷猜想

DataFrame数据：open, high, close, low, volume, price_change, p_change, ma5, ma10, ma20, v_ma5, v_ma10, v_ma20, turnover

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heapq
import datetime



def wave_guess(arr):
    wn = int(len(arr) / 4)  # 没有经验数据，先设置成1/4
    print(wn)
    # 计算最小的N个值，认为是波谷
    wave_crest = heapq.nlargest(wn, enumerate(arr), key=lambda x : x[1])
    wave_crest_mean = pd.DataFrame(wave_crest).mean()

    # 计算最大的N个值，认为是波峰
    wave_base = heapq.nsmallest(wn, enumerate(arr), key=lambda x : x[1])
    wave_base_mean = pd.DataFrame(wave_base).mean()

    print(len(wave_base_mean))
    print("######### result #########")
    # 波峰，波谷的平均值的差，是波动周期
    wave_period = abs(int(wave_crest_mean[0] - wave_base_mean[0]))
    print("wave_period_day:", wave_period)
    print("wave_crest_mean:", round(wave_crest_mean[0], 2))
    print("wave_base_mean:", round(wave_base_mean[0], 2))

    ############### 以下为画图显示用 ###############
    wave_crest_x = []  # 波峰x
    wave_crest_y = []  # 波峰y
    for i, j in wave_crest:
        wave_crest_x.append(i)
        wave_crest_y.append(j)

    wave_base_x = []  # 波谷x
    wave_base_y = []  # 波谷y
    for i, j in wave_base:
        wave_base_x.append(i)
        wave_base_y.append(j)

    # 将原始数据和波峰，波谷画到一张图上
    plt.figure(figsize=(20, 10))
    plt.plot(arr)
    plt.plot(wave_base_x, wave_base_y, 'go')  # 红色的点
    plt.plot(wave_crest_x, wave_crest_y, 'ro')  # 蓝色的点
    plt.grid()
    plt.show()





