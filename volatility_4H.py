#！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/12/31 12:27

@desc:

'''

# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/12/31 12:27

@desc:

'''

import pandas as pd
import numpy as np
from tools.stockstats import StockDataFrame
import time
import copy
import csv


def csv2df(filename):
    lines = list(csv.reader(open(r'./quantitative/data/' + filename)))
    header, values = lines[0], lines[1:]
    data_dict = {h: v for h, v in zip(header, zip(*values))}
    return pd.DataFrame(data_dict)


# filename = 'BTC2017-09-01-now-4H'
filename = 'BTC2015-02-19-now-4H'
source_data = csv2df(filename + '.csv')
source_data = source_data.astype(float)
source_data['Timestamp'] = source_data['Timestamp'].astype(int)
stock = StockDataFrame.retype(source_data)
source_data['date'] = source_data['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
source_data['date'] = pd.to_datetime(source_data['date'])

l = []
for late_cycles in range(6, 48, 6):
    for mean_cycles in range(1, 5):
        max_money = 0.0
        true_times = 0.0
        total_times = 0.0
        money = 10000

        df = copy.deepcopy(source_data)
        df['volatility_rate'] = (df['close'] - df['close'].shift(late_cycles)) / df['close'].shift(late_cycles)
        df['volatility_mean'] = df['volatility_rate'].rolling(window=late_cycles).mean()
        df['volatility_mean'] = df['volatility_mean'].rolling(window=mean_cycles).mean()
        df['volatility_std'] = df['volatility_rate'].rolling(window=late_cycles).std()
        df['volatility_abs'] = abs(df['volatility_rate'])
        df = df[50:]

        df['signal'] = np.where((df['volatility_rate'].shift(1) < 0) & (df['volatility_rate'] > 0), 'long', 'wait')
        df['signal'] = np.where((df['volatility_rate'] > 0) & (df['volatility_mean'] > 0) & (
                df['volatility_rate'].shift(1) < df['volatility_mean'].shift(1)) & (
                                        df['volatility_rate'] > df['volatility_mean']), 'long', df['signal'])
        df['signal'] = np.where((df['volatility_rate'].shift(1) > 0) & (df['volatility_rate'] < 0), 'short',
                                df['signal'])
        df['signal'] = np.where((df['volatility_rate'] < 0) & (df['volatility_mean'] < 0) & (
                df['volatility_rate'].shift(1) > df['volatility_mean'].shift(1)) & (
                                        df['volatility_rate'] < df['volatility_mean']), 'short', df['signal'])
        df['signal'] = np.where(
            (df['signal'] == 'wait') & (df['volatility_rate'].shift(1) > df['volatility_mean'].shift(1)) & (
                    df['volatility_rate'] < df['volatility_mean']), 'close_long', df['signal'])
        df['signal'] = np.where(
            (df['signal'] == 'wait') & (df['volatility_rate'].shift(1) < df['volatility_mean'].shift(1)) & (
                    df['volatility_rate'] > df['volatility_mean']), 'close_short', df['signal'])

        df[['close']] = round(df[['close']], 1)
        df[['volatility_rate', 'volatility_mean', 'volatility_std']] = round(
            df[['volatility_rate', 'volatility_mean', 'volatility_std']], 5)
        df_ = df[['date', 'close', 'volatility_rate', 'volatility_mean', 'volatility_std', 'signal']].loc[
            df['signal'] != 'wait']
        df_['signal'] = np.where(df_['signal'].shift(1) == df_['signal'], 'wait', df_['signal'])
        if (df_[:1]['signal'] == 'close_long').bool() or (df_[:1]['signal'] == 'close_short').bool():
            df_ = df_[1:]
        if (df_[-1:]['signal'] == 'long').bool() or (df_[-1:]['signal'] == 'short').bool():
            df_ = df_[:len(df_) - 1]
        for i in range(1, len(df_), 2):
            total_times += 1
            row_ = df_.iloc[i - 1]
            row = df_.iloc[i]
            if row['signal'] == 'close_long' and row_['signal'] == 'long':
                money = money * row['close'] / row_['close']
                if row['close'] > row_['close']:
                    true_times += 1
            elif row['signal'] == 'close_short' and row_['signal'] == 'short':
                money = money * row_['close'] / row['close']
                if row['close'] < row_['close']:
                    true_times += 1
            elif row['signal'] == 'short' and row_['signal'] == 'long':
                money = money * row['close'] / row_['close']
                if row['close'] > row_['close']:
                    true_times += 1
                i -= 1
            elif row['signal'] == 'long' and row['signal'] == 'short':
                money = money * row_['close'] / row['close']
                if row['close'] < row_['close']:
                    true_times += 1
                i -= 1
            max_money = max(money, money)
        print(late_cycles, mean_cycles, round(true_times / total_times, 3), money, max_money)
        l.append([late_cycles, mean_cycles, round(true_times / total_times, 3), money, max_money])

result = pd.DataFrame(l, columns=['late_cycles', 'mean_cycles', 'shooting', 'money', 'max_money'])
result.to_csv('result_4H.csv')

