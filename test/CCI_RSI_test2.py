# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/9/29 11:09

@desc: CCI超短线（30m)现货策略

# The optimal parameters
# by shooting: dk=13, d=20, cci=-95
# by min principal: dk=12, d=30, cci=-95

'''
import pandas as pd
import numpy as np
from tools.stockstats import StockDataFrame
import time
import matplotlib.ticker as ticker
from tools import data2df
import matplotlib.pyplot as plt

# 本金
principal = 10000.0
filename = 'BTC2017-09-01-now-1H'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)
df['Timestamp'] = df['Timestamp'].astype(int)
stock = StockDataFrame.retype(df)

df['date'] = df['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
df['date'] = pd.to_datetime(df['date'])

stock['cci']
stock['stoch_rsi']
# 去掉前几个指标不准数据H
df = df[5:]

dk, d, cci = 15, 20, -95

ll = []
# 买入时机
df['rsi_regime'] = np.where(
    (df['stoch_k'] - df['stoch_k'].shift(1) + df['stoch_d'].shift(1) - df['stoch_d'] > dk) & (
            df['stoch_d'] < d) & (
            df['cci'] < cci), 1, 0)
# 卖出时机
# df['rsi_regime'] = np.where(df['stoch_d'] >= df['stoch_k'], -1,
#                             df['rsi_regime'])
df['rsi_regime'] = np.where(df['rsi_regime'].shift(1) == 1, -1, df['rsi_regime'])
rsi_df = df.loc[(df['rsi_regime'] == 1) | (df['rsi_regime'] == -1)]

if (rsi_df[:1]['rsi_regime'] == -1).bool():
    rsi_df = rsi_df[1:]
if (rsi_df[-1:]['rsi_regime'] == 1).bool():
    rsi_df = rsi_df[:len(rsi_df) - 1]

rsi_df = rsi_df.loc[((rsi_df['rsi_regime'] == -1) & (rsi_df['rsi_regime'].shift(1) == 1)) | (
        (rsi_df['rsi_regime'] == 1) & (rsi_df['rsi_regime'].shift(-1) == -1))]

rsi_df[['date', 'close', 'rsi_regime']].to_csv('../data/rsi_df.csv', index=None)

# 本息计算
# 命中次数， 最大回撤
principal = 10000
true_times = 0
total_times = len(rsi_df) / 2
max_principal = 0
min_principal = 10000
l = []

# for i in range(1, len(rsi_df), 2):
#     row_ = rsi_df.iloc[i - 1]
#     buy_price = row_['close']
#     row = rsi_df.iloc[i]
#     principal_ = principal
#     l.append([row_['date'], row_['close'], principal, 'buy'])
#     sell_price = row['close']
#     principal = principal * sell_price / buy_price
#     l.append([row['date'], sell_price, principal, 'sell'])
#
#     max_principal = max(max_principal, principal)
#     min_principal = min(min_principal, principal)
#     if sell_price > buy_price:
#         true_times += 1

for i in range(1, len(rsi_df), 2):
    row_ = rsi_df.iloc[i - 1]
    buy_price = row_['close']
    stop_price = max(row_['low'], row_['close'] * 0.985)
    row = rsi_df.iloc[i]
    principal_ = principal
    l.append([row_['date'], row_['close'], principal, 'buy'])
    if stop_price > row['low']:
        sell_price = stop_price
    else:
        sell_price = row['close']
    principal = principal * sell_price / buy_price
    l.append([row['date'], sell_price, principal, 'sell'])

    max_principal = max(max_principal, principal)
    min_principal = min(min_principal, principal)
    if sell_price > buy_price:
        true_times += 1

rsi_profits = pd.DataFrame(l, columns=['date', 'price', 'principal', 'side'])
print('最大回撤：%.10f, 最大本息：%.10f, 命中率：%.10f' % (min_principal / 10000, max_principal, true_times / total_times))
rsi_profits.to_csv('../data/rsi_profits.csv', index=None)

ax = rsi_profits[['price', 'principal']].plot(figsize=(20, 10), grid=True, xticks=rsi_profits.index, rot=90, subplots=True, style='b')
# 设置x轴
# 刻度数量
ax[0].xaxis.set_major_locator(ticker.MaxNLocator(40))
# 以date为x轴刻度
ax[0].set_xticklabels(rsi_profits.date)
# 美观x轴刻度
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
# plt.savefig('../data/' + filename + '.png')
plt.show()
