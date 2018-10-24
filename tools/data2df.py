#！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/9/13 9:44

@desc: csv文件转换成DataFrame格式

'''
import pandas as pd
import csv

def csv2df(filename):
    lines = list(csv.reader(open(r'./quantitative/data/' + filename)))
    header, values = lines[0], lines[1:]
    data_dict = {h: v for h, v in zip(header, zip(*values))}
    return pd.DataFrame(data_dict)
