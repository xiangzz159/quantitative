# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/11/9 9:17

@desc:

'''
from tools.stockstats import StockDataFrame
import csv
import time
import pandas as pd
import numpy as np


def floatrange(start, stop, num):
    return [start + float(i) * num for i in range(int((stop - start) / num))]


def csv2df(filename):
    lines = list(csv.reader(open(r'./quantitative/data/' + filename)))
    header, values = lines[0], lines[1:]
    data_dict = {h: v for h, v in zip(header, zip(*values))}
    return pd.DataFrame(data_dict)


filename = 'BTC2017-09-01-now-2H'
l = []
for hist_ema in range(3, 12):
    for hist_signal_ma in range(3, 10):
        for hist_signal_ma_ in range(3, 10):
            for ii in floatrange(-5.0, 5.0, 0.1):
                for jj in floatrange(-5.0, 5.0, 0.1):
                    money = 10000
                    df = csv2df(filename + '.csv')
                    df = df.astype(float)
                    df['Timestamp'] = df['Timestamp'].astype(int)
                    stock = StockDataFrame.retype(df)
                    df['date'] = df['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
                    df['date'] = pd.to_datetime(df['date'])

                    hist_ema_name = 'hist_%d_ema' % hist_ema
                    hist_signal_ma_name = 'hist_signal_%d_ma' % hist_signal_ma
                    hist_signal_ma1_name = 'hist_signal_%d_ma_' % hist_signal_ma_
                    stock['macd']
                    stock[hist_ema_name]

                    df[hist_signal_ma_name] = np.round(
                        df['hist_signal'].rolling(min_periods=1, window=hist_signal_ma).mean(), 2)
                    df[hist_signal_ma1_name] = np.round(
                        df[hist_signal_ma_name].rolling(min_periods=1, window=hist_signal_ma_).mean(), 2)
                    df[hist_ema_name] = np.round(df[hist_ema_name], 2)

                    # DIF->macd  DEA->macds  MACD->macdh
                    df['signal'] = np.where(
                        ((abs(df[hist_ema_name]) >= ii) & (df[hist_signal_ma_name] < jj)) & ((
                                (df[hist_signal_ma_name] >= -1) & (df[hist_signal_ma_name].shift(1) < -1))), 'long',
                        'wait')
                    df['signal'] = np.where((df['hist'].shift(1) < 0) & (df['hist'] > 0), 'long', df['signal'])

                    df['signal'] = np.where(
                        ((abs(df[hist_ema_name]) >= ii) & (df[hist_signal_ma_name] >= jj)) & ((
                                (df[hist_signal_ma_name] <= 1) & (df[hist_signal_ma_name].shift(1) > 1))), 'short',
                        df['signal'])
                    df['signal'] = np.where((df['hist'].shift(1) > 0) & (df['hist'] < 0), 'short', df['signal'])

                    df_signal = df.loc[df['signal'] != 'wait']
                    df_signal = df_signal[::1]
                    # 筛选出重复信号
                    unless_signal = df_signal.loc[
                        ((df_signal['signal'] == 'short') & (df_signal['signal'].shift(1) == 'short')) | (
                                (df_signal['signal'] == 'long') & (df_signal['signal'].shift(1) == 'long'))]
                    # 过滤重复信号
                    for index, row in unless_signal.iterrows():
                        df.signal[index] = 'wait'
                    # 去掉前面100条指标不准数据
                    df = df[100:]
                    df_signal = df.loc[df['signal'] != 'wait']

                    # 最大回撤，本金最大值， 正确次数， 总交易次数， 买卖双向手续费
                    max_drawdown = 0.0
                    max_money = 0.0
                    true_times = 0.0
                    total_times = 0.0
                    rate = 0.15 / 100
                    for i in range(1, len(df_signal)):
                        total_times += 1
                        row_ = df_signal.iloc[i - 1]
                        row = df_signal.iloc[i]
                        fragment_df = df.loc[
                            (df['timestamp'] >= row_['timestamp']) & (df['timestamp'] <= row['timestamp'])]
                        max_price = max(fragment_df.close)
                        min_price = min(fragment_df.close)
                        begin_money = money
                        if row_['signal'] == 'long':
                            money = money * (row['close'] / row_['close'] - rate)
                            max_drawdown = max(max_drawdown, round((row_['close'] - min_price) / row_['close'], 3))
                        elif row_['signal'] == 'short':
                            money = money * (row_['close'] / row['close'] - rate)
                            max_drawdown = max(max_drawdown, round((row_['close'] - max_price) / row_['close'], 3))
                        if money > begin_money:
                            true_times += 1
                        max_money = max(max_money, money)

                    print(filename, hist_ema, hist_signal_ma, hist_signal_ma_, ii, jj, max_drawdown, max_money,
                          round(true_times / total_times, 3), money)
                    l.append([filename, hist_ema, hist_signal_ma, hist_signal_ma_, ii, jj, max_drawdown, max_money,
                              round(true_times / total_times, 3), money])

result = pd.DataFrame(l, columns=['filename', 'hist_ema', 'hist_signal_ma', 'hist_signal_ma_', 'i', 'j', 'max_drawdown',
                                  'max_money', 'shooting', 'last_money'])
result.to_csv('result_BTC_2H.csv')
