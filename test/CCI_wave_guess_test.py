#！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/12/27 9:32

@desc:

'''

import pandas as pd
import numpy as np
from tools.stockstats import StockDataFrame
import time
import matplotlib.ticker as ticker
from tools import data2df
import matplotlib.pyplot as plt
from job import wave_guess

# 本金
principal = 10000.0
filename = 'BTC2015-02-19-now-4H'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)
df['Timestamp'] = df['Timestamp'].astype(int)
stock = StockDataFrame.retype(df)

df['date'] = df['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
df['date'] = pd.to_datetime(df['date'])

stock['cci']
stock['stoch_rsi']
# 去掉前几个cci指标不准数据
df = df[5:]
df['dk'] = df['stoch_d'] - df['stoch_k']
dk = 1
arr = pd.Series(df['cci'].values)
l = [[0, 0] for x in range(125)]
for i in range(125, len(df)):
    arr_ = arr[i - 125: i]
    wave_crest, wave_base = wave_guess.wave_return(arr_)
    l.append([wave_crest[1], wave_base[1]])
arr = np.array(l)
signal_rate = 0.8
df['wave_crest'] = arr[:, :1] * signal_rate
df['wave_base'] = arr[:, 1:] * signal_rate

df = df[125:]
# df[['cci', 'wave_crest', 'wave_base']] = round(df[['cci', 'wave_crest', 'wave_base']], 1)
# df[['cci', 'wave_crest', 'wave_base', 'close']].to_csv('../data/wave_cci_result.csv', index=False)


df['signal'] = np.where((df['cci'] >= df['wave_crest']) & (df['cci'].shift(1) < df['wave_crest']), 'long',
                        'wait')
df['signal'] = np.where(
    (df['cci'] <= df['wave_base']) & (df['cci'].shift(1) >= df['wave_base']),
    'short', df['signal'])

df['signal'] = np.where(
    (df['signal'] == 'wait') & (((df['cci'] <= df['wave_crest']) & (df['cci'].shift(1) > df['wave_crest'])) |
                                ((df['cci'] >= df['wave_crest']) & (df['dk'] > dk)) | (
                                        (df['cci'] <= df['wave_crest']) & (df['cci'].shift(1) > df['wave_crest']))),
    'close_long', df['signal'])
df['signal'] = np.where(
    (df['signal'] == 'wait') & (((df['cci'] >= df['wave_base']) & (df['cci'].shift(1) <= df['wave_base'])) | (
            (df['cci'] <= df['wave_base']) & (df['dk'] < dk * -1)) | (
                                            (df['cci'] >= df['wave_base']) & (df['cci'].shift(1) <= df['wave_base']))),
    'close_short', df['signal'])

# df[['close', 'cci', 'wave_crest', 'wave_base']] = round(df[['close', 'cci', 'wave_crest', 'wave_base']], 1)
# df[['date', 'close', 'cci', 'wave_crest', 'wave_base', 'signal']].to_csv('../data/wave_cci_result.csv', index=False)

df_regime = df.loc[df['signal'] != 'wait']
# 逆序
df_regime = df_regime[::-1]
unless_regime = df_regime.loc[
    ((df_regime['signal'] == 'close_short') & (df_regime['signal'].shift(1) == 'close_short')) | (
            (df_regime['signal'] == 'close_long') & (df_regime['signal'].shift(1) == 'close_long')) | (
            (df_regime['signal'] == 'close_short') & (df_regime['signal'].shift(1) == 'close_long')) | (
            (df_regime['signal'] == 'close_long') & (df_regime['signal'].shift(1) == 'close_short'))]

for index, row in unless_regime.iterrows():
    df_regime = df_regime.drop([index])
df_regime = df_regime[::-1]
if (df_regime[:1]['signal'] == 'close_long').bool() or (df_regime[:1]['signal'] == 'close_short').bool():
    df_regime = df_regime[1:]
if (df_regime[-1:]['signal'] == 'long').bool() or (df_regime[-1:]['signal'] == 'short').bool():
    df_regime = df_regime[:len(df_regime) - 1]

# print(df_regime[['date', 'close', 'signal']].tail(50))
# 计算本息
l = []
for i in range(1, len(df_regime), 2):
    row_ = df_regime.iloc[i - 1]
    row = df_regime.iloc[i]
    if row['signal'] == 'close_long' and row_['signal'] == 'long':
        principal = principal * row['close'] / row_['close']
        l.append([row['date'], row['close'], principal, 'long', 1])
    elif row['signal'] == 'close_short' and row_['signal'] == 'short':
        principal = principal * row_['close'] / row['close']
        l.append([row['date'], row['close'], principal, 'short', -1])
profits = pd.DataFrame(l, columns=['date', 'close', 'principal', 'type', 's'])
# print(profits)

ax = profits[['close', 'principal']].plot(figsize=(20, 10), grid=True, xticks=profits.index, rot=90, subplots=True,
                                          style='b')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 设置x轴刻度数量
ax[0].xaxis.set_major_locator(ticker.MaxNLocator(40))
# 以date为x轴刻度
ax[0].set_xticklabels(profits.date)
# 美观x轴刻度
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
ax[0].set_title(filename + ' 新策略')
# plt.savefig('../data/' + filename + '.png')
plt.show()
