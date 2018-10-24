# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/9/29 11:09

@desc: CCI超短线（30m)现货策略

# The optimal parameters
# by shooting: dk=13, d=10, cci=-95, rsi=45
# by min principal: dk=13, d=10, cci=-100, rsi=35
# by total times: dk=13, d=35, cci=-85, rsi=45
54432
'''
import pandas as pd
import numpy as np
from tools.stockstats import StockDataFrame
from tools import data2df
import copy
import asyncio
import time

tt = {
    '30M': 1800,
    '1H': 3600,
    '2H': 7200,
    '4H': 14400
}

ll = []


async def is_true(df, compare_df, compare_time):
    for index, row in df.loc[df['rsi_regime'] == 1].iterrows():
        t = row['timestamp'] - row['timestamp'] % compare_time - compare_time
        compare_row = compare_df.loc[compare_df['timestamp'] == t].iloc[0]
        # 超过这个区间就不操作
        if compare_row['cci'] < compare_cci and compare_row['cci'] > -1 * compare_cci:
            continue
        else:
            df['rsi_regime'][index] = 0
    return df


async def analysis(df, compare_df, dk, d, cci, rsi, compare_cci, compare_time, f1, f2):
    # 做多时机
    df['rsi_regime'] = np.where(
        (df['stoch_k'] - df['stoch_k'].shift(1) + df['stoch_d'].shift(1) - df['stoch_d'] > dk) & (
                df['stoch_d'] < d) & (
                df['cci'] < cci) | (
                df['stoch_k'] - df['stoch_k'].shift(1) + df['stoch_d'].shift(1) - df['stoch_d'] > dk) & (
                df['stoch_rsi'] < rsi) & (
                df['cci'] < cci), 1, 0)

    df = await is_true(df, compare_df, compare_time)

    df['rsi_regime'] = np.where(df['rsi_regime'].shift(1) == 1, -1, df['rsi_regime'])
    rsi_df = df.loc[(df['rsi_regime'] == 1) | (df['rsi_regime'] == -1)]

    if (rsi_df[:1]['rsi_regime'] == -1).bool():
        rsi_df = rsi_df[1:]
    if (rsi_df[-1:]['rsi_regime'] == 1).bool():
        rsi_df = rsi_df[:len(rsi_df) - 1]

    rsi_df = rsi_df.loc[((rsi_df['rsi_regime'] == -1) & (rsi_df['rsi_regime'].shift(1) == 1)) | (
            (rsi_df['rsi_regime'] == 1) & (rsi_df['rsi_regime'].shift(-1) == -1))]

    # 本息计算
    # 命中次数， 最大回撤
    principal = 10000
    true_times = 0
    total_times = len(rsi_df) / 2
    max_principal = 0
    min_principal = 10000
    for i in range(1, len(rsi_df), 2):
        row_ = rsi_df.iloc[i - 1]
        row = rsi_df.iloc[i]
        buy_price = row_['close']
        sell_price = row['close']
        principal = principal * sell_price / buy_price

        max_principal = max(max_principal, principal)
        min_principal = min(min_principal, principal)
        if sell_price > buy_price:
            true_times += 1

    print(dk, d, cci, rsi, compare_cci, min_principal, max_principal, total_times, true_times / total_times,
          principal,
          f1 + '-' + f2)
    ll.append(
        [dk, d, cci, rsi, compare_cci, min_principal, max_principal, total_times, true_times / total_times,
         principal,
         f1 + '-' + f2])


loop = asyncio.get_event_loop()

f1, f2 = '15M', '30M'
filename = 'BTC2017-09-01-now-' + f1
compare_filename = 'BTC2017-09-01-now-' + f2
compare_time = tt[f2]
df_ = data2df.csv2df(filename + '.csv')
df_ = df_.astype(float)
df_['Timestamp'] = df_['Timestamp'].astype(int)
stock = StockDataFrame.retype(df_)
df_['date'] = df_['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
df_['date'] = pd.to_datetime(df_['date'])
stock['cci']
stock['stoch_rsi']
# 震荡行情判断
compare_df_ = data2df.csv2df(compare_filename + '.csv')
compare_df_ = compare_df_.astype(float)
compare_df_['Timestamp'] = compare_df_['Timestamp'].astype(int)
compare_stock = StockDataFrame.retype(compare_df_)
compare_df_['date'] = compare_df_['timestamp'].apply(
    lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
compare_df_['date'] = pd.to_datetime(df_['date'])
compare_stock['cci']
compare_stock['stoch_rsi']
# 去掉前几个指标不准数据H
df_ = df_[5:]
compare_df_ = compare_df_[5:]
for dk in range(-5, 5):
    for d in range(10, 20):
        for cci in range(-95, -80):
            for rsi in (5, 30):
                as_list = []
                for compare_cci in range(50, 95):
                    df = copy.deepcopy(df_)
                    compare_df = copy.deepcopy(compare_df_)
                    as_list.append(analysis(df, compare_df, dk, d, cci, rsi, compare_cci, compare_time, f1, f2))
                asyncio.get_event_loop().run_until_complete(asyncio.gather(*as_list))

result = pd.DataFrame(ll,
                      columns=['dk', 'd', 'cci', 'rsi', 'compare_cci', 'min_principal', 'max_principal', 'total_times',
                               'shooting',
                               'last_principal', 'f'])
result = result.sort_values(by=['max_principal', 'last_principal', 'shooting', 'min_principal', 'total_times'],
                     ascending=(False, False, False, False, False))
result[:200].to_csv('result-' + f1 + '-' + f2 + '.csv', index=None)
