# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/10/23 19:02

@desc:

'''

import pandas as pd
import numpy as np
from tools.stockstats import StockDataFrame
from tools import data2df
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time


def is_true(df, compare_df, compare_time):
    for index, row in df.loc[df['rsi_regime'] == 1].iterrows():
        t = row['timestamp'] - row['timestamp'] % compare_time - compare_time
        compare_row = compare_df.loc[compare_df['timestamp'] == t].iloc[0]
        if compare_row['cci'] < compare_cci or compare_row['cci'] > -1 * compare_cci:
            continue
        else:
            df['rsi_regime'][index] = 0
    return df

tt = {
    '30M': 1800,
    '1H': 3600,
    '2H': 7200,
    '4H': 14400
}

dk,d,cci,rsi,compare_cci = 0,15,-85,20,80

f1, f2 = '15M', '30M'
filename = 'BTC2017-09-01-now-' + f1
compare_filename = 'BTC2017-09-01-now-' + f2
compare_time = tt[f2]
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)
df['Timestamp'] = df['Timestamp'].astype(int)
stock = StockDataFrame.retype(df)
df['date'] = df['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
df['date'] = pd.to_datetime(df['date'])
stock['cci']
stock['stoch_rsi']
# 震荡行情判断
compare_df_ = data2df.csv2df(compare_filename + '.csv')
compare_df_ = compare_df_.astype(float)
compare_df_['Timestamp'] = compare_df_['Timestamp'].astype(int)
compare_stock = StockDataFrame.retype(compare_df_)
compare_df_['date'] = compare_df_['timestamp'].apply(
    lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
compare_df_['date'] = pd.to_datetime(df['date'])
compare_stock['cci']
compare_stock['stoch_rsi']

# 做多时机
df['rsi_regime'] = np.where(
    (df['stoch_k'] - df['stoch_k'].shift(1) + df['stoch_d'].shift(1) - df['stoch_d'] > dk) & (
            df['stoch_d'] < d) & (
            df['cci'] < cci) | (
            df['stoch_k'] - df['stoch_k'].shift(1) + df['stoch_d'].shift(1) - df['stoch_d'] > dk) & (
            df['stoch_rsi'] < rsi) & (
            df['cci'] < cci), 1, 0)

# 做空时机
dk_,d_,cci_,rsi_,compare_cci_ = 0,85,85,80,80
df['rsi_regime'] = np.where(
    (df['stoch_k'] - df['stoch_k'].shift(1) + df['stoch_d'].shift(1) - df['stoch_d'] > dk_) & (
            df['stoch_d'] > d_) & (
            df['cci'] > cci_) | (
            df['stoch_k'] - df['stoch_k'].shift(1) + df['stoch_d'].shift(1) - df['stoch_d'] > dk_) & (
            df['stoch_rsi'] > rsi_) & (
            df['cci'] > cci_), -1, df['rsi_regime'])

df = is_true(df, compare_df_, compare_time)

df[['date', 'open', 'high', 'low', 'close', 'rsi_regime']].loc[df['rsi_regime'] != 0].to_csv('../data/res.csv', index=False)

# df['rsi_regime'] = np.where(df['rsi_regime'].shift(1) == 1, -1, df['rsi_regime'])
# rsi_df = df.loc[(df['rsi_regime'] == 1) | (df['rsi_regime'] == -1)]
#
# if (rsi_df[:1]['rsi_regime'] == -1).bool():
#     rsi_df = rsi_df[1:]
# if (rsi_df[-1:]['rsi_regime'] == 1).bool():
#     rsi_df = rsi_df[:len(rsi_df) - 1]
#
# rsi_df = rsi_df.loc[((rsi_df['rsi_regime'] == -1) & (rsi_df['rsi_regime'].shift(1) == 1)) | (
#         (rsi_df['rsi_regime'] == 1) & (rsi_df['rsi_regime'].shift(-1) == -1))]
#
# # 本息计算
# # 命中次数， 最大回撤
# principal = 10000
# true_times = 0
# total_times = len(rsi_df) / 2
# max_principal = 0
# min_principal = 10000
# ll = []
# for i in range(1, len(rsi_df), 2):
#     row_ = rsi_df.iloc[i - 1]
#     row = rsi_df.iloc[i]
#     buy_price = row['close']
#     sell_price = row_['close']
#     principal = principal * sell_price / buy_price
#     ll.append([row['date'], row['close'], principal])
#     max_principal = max(max_principal, principal)
#     min_principal = min(min_principal, principal)
#     if sell_price > buy_price:
#         true_times += 1
#
# profits = pd.DataFrame(ll, columns=['date', 'close', 'principal'])
# ax = profits[['close', 'principal']].plot(figsize=(20, 10), grid=True, xticks=profits.index, rot=90, subplots=True,
#                                           style='b')
#
# # 设置x轴刻度数量
# ax[0].xaxis.set_major_locator(ticker.MaxNLocator(40))
# # 以date为x轴刻度
# ax[0].set_xticklabels(profits[::20].date)
# # 美观x轴刻度
# plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
# plt.title(filename)
# # plt.savefig('../data/CCI_RSI_profits1.png')
# plt.show()
