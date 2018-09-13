# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/9/13 9:41

@desc: 均线交叉策略
* 当短期均线越过长期均线时，交易金融资产。
* 当短期均线再一次越过长期均线时，结束交易
当短期均线高于长期均线时，我们应进行多头交易，当短期均线再次越过（低于）长期均线时，结束此类交易。当短期均线低于长期均线时，我们应进行空头交易，当短期均线再次越过（高于）长期均线时，结束此类交易。
也就是说，如果短期均线高于长期均线，那么这是一个牛市行情（牛市规则），如果短期均线低于长期均线，则目前为熊市行情（熊市规则）

'''

import pandas as pd
import numpy as np
from tools import data_transform
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time

df = data_transform.transform('BTC2016-now-1D.csv')
df['ts'] = df['Timestamp'].apply(lambda x: time.strftime("%Y--%m--%d", time.localtime(int(x))))
# 转换datetime格式
df['ts'] = pd.to_datetime(df['ts'])
# print(df.dtypes)
# print(df.head())

df['MA10'] = np.round(df['Close'].rolling(window=10, center=False).mean(), 2)
df['MA30'] = np.round(df['Close'].rolling(window=30, center=False).mean(), 2)

df['MA10-MA30'] = df['MA10'] - df['MA30']

df['Regime'] = np.where(df['MA10-MA30'] > 0, 1, 0)
df['Regime'] = np.where(df['MA10-MA30'] < 0, -1, df['Regime'])

# TODO x轴不为时间轴，需要改进
# df[['Regime']].plot(ylim=(-2, 2), figsize=(50, 25), grid=True).axhline(y=0, color="black", lw=2)

# 分别计算买入和抛出机会 1：买入， -1：抛出
df["Signal"] = np.sign(df["Regime"] - df["Regime"].shift(1))
df.loc[0:0,('Signal')] = 0.0


# df['Signal'].plot(ylim = (-2, 2))
# plt.show()

# print(df['Signal'].value_counts())
# print(df.head(100))

# print(df.loc[df['Signal'] == 1, 'ts', 'Close'])

# print(df['Signal'].value_counts())
# print('*' * 50)
# print(df.loc[df["Signal"] != 0, ['ts', 'Timestamp', 'Close', 'Signal', 'Regime']])

df_signals = pd.concat([
    pd.DataFrame({"Price": df.loc[df["Signal"] == 1, "Close"],
                  "Regime": df.loc[df["Signal"] == 1, "Regime"],
                  "Signal": "Buy"}),
    pd.DataFrame({"Price": df.loc[df["Signal"] == -1, "Close"],
                  "Regime": df.loc[df["Signal"] == -1, "Regime"],
                  "Signal": "Sell"}),
])
df_signals.sort_index(inplace=True)
# 买入卖出时机
# print(df_signals)

# print(df_signals.dtypes)
df_signals[['Price']] = df_signals[['Price']].astype(float)
df_long_profits = pd.DataFrame({
    "Price": df_signals.loc[(df_signals["Signal"] == "Buy") & df_signals["Regime"] == 1, "Price"],
    "Profit": pd.Series(df_signals["Price"] - df_signals["Price"].shift(1)).loc[
        df_signals.loc[(df_signals["Signal"].shift(1) == "Buy") & (df_signals["Regime"].shift(1) == 1)].index
    ].tolist(),
    "End Date": df_signals["Price"].loc[
        df_signals.loc[(df_signals["Signal"].shift(1) == "Buy") & (df_signals["Regime"].shift(1) == 1)].index
    ].index
})
# 输出利润
print(df_long_profits)
