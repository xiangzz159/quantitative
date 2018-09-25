# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/9/25 8:37

@desc:

'''

import pandas as pd
import numpy as np
from tools.stockstats import StockDataFrame
from job import poloniex_data_collection
import time
import matplotlib.ticker as ticker
from tools import data2df
import matplotlib.pyplot as plt

df = data2df.csv2df('BTC2018-09-15-now-30M.csv')
df = df.astype(float)
df['Timestamp'] = df['Timestamp'].astype(int)
stock = StockDataFrame.retype(df)

df['date'] = df['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
df['date'] = pd.to_datetime(df['date'])

stock['cci']

df['regime'] = np.where((df['cci'] >= 100) & (df['cci'].shift(1) < 100), 1, 0)
df['regime'] = np.where((df['cci'] <= 100) & (df['cci'].shift(1) > 100), -1, df['regime'])

df['regime'] = np.where((df['cci'] <= -100) & (df['cci'].shift(1) >= -100), -1, df['regime'])
df['regime'] = np.where((df['cci'] >= -100) & (df['cci'].shift(1) <= -100), 1, df['regime'])

df = df.loc[(df['regime'] == 1) | (df['regime'] == -1)]
df = df[5:]
# print(df[['date', 'close', 'cci', 'regime']])
# print('*' * 50)

# 数据清理
l = []
for index, row in df.iterrows():
    # print(row['date'], row['regime'], row['cci'])
    if row['regime'] == 1 and row['cci'] >= 100:
        l.append([row['date'], row['close'], row['cci'], row['regime'], 'long', 'long'])
    elif row['regime'] == -1 and (row['cci'] <= 100 and row['cci'] > -100):
        l.append([row['date'], row['close'], row['cci'], row['regime'], 'close_long', 'long'])
    elif row['regime'] == -1 and row['cci'] <= -100:
        l.append([row['date'], row['close'], row['cci'], row['regime'], 'short', 'short'])
    elif row['regime'] == 1 and (row['cci'] >= -100 and row['cci'] < 100):
        l.append([row['date'], row['close'], row['cci'], row['regime'], 'close_short', 'short'])

df_signals = pd.DataFrame(l, columns=['date', 'price', 'cci', 'regime', 'signal', 'type'])

long_df_signals = df_signals.loc[df_signals['type'] == 'long']
if long_df_signals['signal'][0] == 'close_long':
    long_df_signals = long_df_signals[1:]
if (long_df_signals['signal'][-1:] == 'long').bool():
    long_df_signals = long_df_signals[:len(long_df_signals)]


# 做多——利润
# long_profits = pd.DataFrame({
#     'data': df_signals.loc[(df_signals['type'] == 'long') & (df_signals['signal'] == 'close_long'), 'date'],
#     'price': df_signals.loc[(df_signals['type'] == 'long') & (df_signals['signal'] == 'close_long'), 'price'],
#     'profits': pd.Series(df_signals['price'] - df_signals['price'].shift(1)).loc[
#         df_signals.loc[(df_signals['type'] == 'long') & (df_signals['signal'].shift(1) == 'long')].index
#     ].tolist()
# })
# print(long_profits)

# 做空——利润
# short_df_signals = df_signals.loc[df_signals['type'] == 'short']
# short_profits = pd.DataFrame({
#
# })

# df_signals = pd.concat([
#     pd.DataFrame({
#         'date': df.loc[df['signal'] == 1, 'date'],
#         'price': df.loc[df['signal'] == 1, 'close'],
#         'regime': df.loc[df['signal'] == 1, 'regime'],
#         'signal': df.loc[df['signal'] == 1, 'buy'],
#     })
# ])

# print(df[['regime', 'cci', 'close']])
# print(df[['regime', 'cci', 'close']].loc[(df['regime'] == 1) | (df['regime'] == -1)])

# stock['rsi_6']


# print(df[['date', 'macd', 'macds', 'macdh']].tail(30))
# ax = stock[['close', 'cci_20', 'rsi_12']].plot(figsize=(20, 10), grid=True, xticks=df.index, rot=90, subplots=True, style='b')
#
# # 设置x轴刻度数量
# ax[0].xaxis.set_major_locator(ticker.MaxNLocator(40))
# # 以ts为x轴刻度
# ax[0].set_xticklabels(df.date)
# # 美观x轴刻度
# plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
# plt.show()
