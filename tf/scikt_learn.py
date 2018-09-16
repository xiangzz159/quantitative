#！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/9/16 20:57

@desc: 预测价格

'''

import pandas as pd
import sklearn as skl
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
import time
from tools import data_transform

stock_X  = data_transform.transform('BTC2016-now-1D.csv')
stock_X ['Close'] = stock_X['Close'].astype(float)


stock_y  = pd.Series(stock_X['Close'].values)

stock_X_test = stock_X.iloc[len(stock_X)-1]
# 使用今天的交易价格，13 个指标预测明天的价格。偏移股票数据，今天的数据，目标是明天的价格。
stock_X = stock_X.drop(stock_X.index[len(stock_X)-1]) # 删除最后一条数据
stock_y = stock_y.drop(stock_y.index[0]) # 删除第一条数据
#删除掉close 也就是收盘价格。
del stock_X["Close"]
del stock_X_test["Close"]

#使用最后一个数据做测试。
stock_y_test = stock_y.iloc[len(stock_y)-1]

print(stock_X.tail(5))
print("###########################")
print(stock_y.tail(5)) #
#print(stock_X.values[0])

print("###########################")
print(len(stock_X),",",len(stock_y))

print("###########################")
print(stock_X_test.values,stock_y_test)

model = linear_model.LinearRegression()
model.fit(stock_X.values,stock_y)
print("############## test & target #############")
print(model.predict([stock_X_test.values]))
print(stock_y_test)

print("############## coef_ & intercept_ #############")
print(model.coef_) #系数
print(model.intercept_) #截断
print("score:", model.score(stock_X.values,stock_y)) #评分