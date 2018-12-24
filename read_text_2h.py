#！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/12/22 22:01

@desc:

'''
import pandas as pd
import numpy as np

file = open(r'/home/ubuntu/macd_eth_2h.log')

line = file.readline()
arr_list = []
s = set([])
while line:
    str = line.replace('\n', '')
    ss = str.split(' ')
    ss = ss[1:]
    key = ss[6] + '_' + ss[7] + '_' + ss[8] + '_' + ss[9]
    if key not in s:
        arr_list.append(ss)
        s.add(key)
    lines = file.readline()
    if len(s) > 5000:
        s = set([])

file.close()

df = pd.DataFrame(arr_list, columns=['hist_ema', 'hist_signal_ma', 'hist_signal_ma_', 'i', 'j', 'z',
                                  'max_drawdown',
                                  'max_money', 'shooting', 'last_money'])

df.to_csv('result_ETH_2H.csv')