#！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2019/1/4 9:24

@desc:

'''


import pandas as pd
import numpy as np
from tools.stockstats import StockDataFrame
import time
import matplotlib.ticker as ticker
from tools import data2df
import matplotlib.pyplot as plt
from tools import public_tools
import copy

filename = 'ETH2017-09-01-now-1H'
# filename = 'BTC2017-09-01-now-2H'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)
df['Timestamp'] = df['Timestamp'].astype(int)
stock = StockDataFrame.retype(df)

late_cycles = 23
mean_cycles = 4

df['date'] = df['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
df['date'] = pd.to_datetime(df['date'])

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
df['signal'] = np.where((df['volatility_rate'].shift(1) > 0) & (df['volatility_rate'] < 0), 'short', df['signal'])
df['signal'] = np.where((df['volatility_rate'] < 0) & (df['volatility_mean'] < 0) & (
        df['volatility_rate'].shift(1) > df['volatility_mean'].shift(1)) & (
                                df['volatility_rate'] < df['volatility_mean']), 'short', df['signal'])
df['signal'] = np.where((df['signal'] == 'wait') & (df['volatility_rate'].shift(1) > df['volatility_mean'].shift(1)) & (
        df['volatility_rate'] < df['volatility_mean']), 'close_long', df['signal'])
df['signal'] = np.where((df['signal'] == 'wait') & (df['volatility_rate'].shift(1) < df['volatility_mean'].shift(1)) & (
        df['volatility_rate'] > df['volatility_mean']), 'close_short', df['signal'])

df_ = df[['timestamp', 'date', 'close', 'volatility_rate', 'volatility_mean', 'volatility_std', 'signal']].loc[
    df['signal'] != 'wait']
df_['signal'] = np.where(df_['signal'].shift(1) == df_['signal'], 'wait', df_['signal'])
df_['signal'] = np.where((df_['signal'].shift(1) == 'close_long') & (df_['signal'] == 'close_short'), 'wait', df_['signal'])
df_['signal'] = np.where((df_['signal'].shift(1) == 'close_short') & (df_['signal'] == 'close_long'), 'wait', df_['signal'])
df_ = df_.loc[df_['signal'] != 'wait']
if (df_[:1]['signal'] == 'close_long').bool() or (df_[:1]['signal'] == 'close_short').bool():
    df_ = df_[1:]
if (df_[-1:]['signal'] == 'long').bool() or (df_[-1:]['signal'] == 'short').bool():
    df_ = df_[:len(df_) - 1]

stop_rates = public_tools.floatrange(0.99, 1.001, 0.001, 3)
# max_drawdown = 0.0
for stop_rate in stop_rates:
    l = []
    money = 10000
    max_money = 0.0
    true_times = 0
    total_times = 0
    stop_times = 0
    i = 1
    while i < len(df_):
        row_ = df_.iloc[i - 1]
        row = df_.iloc[i]
        side = row_['signal']
        stop_price = row_['close'] * stop_rate if side == 'long' else row_['close'] / stop_rate
        fragment_df = df.loc[(df['timestamp'] > row_['timestamp']) & (df['timestamp'] <= row['timestamp'])]
        high_price = max(fragment_df.high)
        low_price = max(fragment_df.low)
        total_times += 1
        money_ = money
        if row['signal'] == 'close_long' and row_['signal'] == 'long':
            if stop_price > low_price:  # 触发止损
                money = money * stop_price / row_['close']
                l.append([row['date'], stop_price, money, 'long', 3])
                stop_times += 1
            else:
                money = money * row['close'] / row_['close']
                l.append([row['date'], row['close'], money, 'long', 2])
        elif row['signal'] == 'close_short' and row_['signal'] == 'short':
            if stop_price < high_price:
                money = money * row_['close'] / stop_price
                l.append([row['date'], stop_price, money, 'short', 3])
                stop_times += 1
            else:
                money = money * row_['close'] / row['close']
                l.append([row['date'], row['close'], money, 'short', 2])
        elif row['signal'] == 'short' and row_['signal'] == 'long':
            if stop_price > low_price:  # 触发止损
                money = money * stop_price / row_['close']
                l.append([row['date'], stop_price, money, 'long', 3])
                stop_times += 1
            else:
                money = money * row['close'] / row_['close']
                l.append([row['date'], row['close'], money, 'long', 2])
            i -= 1
        elif row['signal'] == 'long' and row_['signal'] == 'short':
            if stop_price < high_price:
                money = money * row_['close'] / stop_price
                l.append([row['date'], stop_price, money, 'short', 3])
                stop_times += 1
            else:
                money = money * row_['close'] / row['close']
                l.append([row['date'], row['close'], money, 'short', 2])
            i -= 1
        i += 2
        if money > money_:
            true_times += 1
        max_money = max(max_money, money)

    print('true_times:%d, total_times:%d, stop_times:%d, max_money:%.10f, stop_rate:%.10f' % (
    true_times, total_times, stop_times, max_money, stop_rate))

    profits = pd.DataFrame(l, columns=['date', 'close', 'money', 'type', 'status'])
    # profits.to_csv('../data/profits_%.3f.csv' % stop_rate, index=False)
    00

    num = 1 if int(len(profits) / 30) == 0 else int(len(profits) / 30)
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    axes[0].plot(range(profits.shape[0]), profits.close, label='Close Price')
    axes[1].plot(range(profits.shape[0]), profits.money, label='Money')

    axes[0].set_title('数字货币价格变化曲线及投资收益曲线(%.3f止损)' % stop_rate)
    axes[0].set_xticks([])
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(30))
    # 以date为x轴刻度
    axes[1].set_xticklabels(profits.date[::num])
    # 美观x轴刻度
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    # 显示label
    axes[0].legend()
    axes[1].legend()
    plt.show()