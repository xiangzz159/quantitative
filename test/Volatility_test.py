import pandas as pd
import numpy as np
from tools.stockstats import StockDataFrame
import time
import matplotlib.ticker as ticker
from tools import data2df
import matplotlib.pyplot as plt

filename = 'BTC2018-04-01-now-1H'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)
df['Timestamp'] = df['Timestamp'].astype(int)
df['date'] = pd.to_datetime(df['Timestamp'], unit='s')
df.index = df.date
stock = StockDataFrame.retype(df)

late_cycles_1 = 18
late_cycles_2 = 18
mean_cycles = 3
volatility = 0.005
long_rate = 0.005
short_rate = 0.005

df['date'] = df['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
df['date'] = pd.to_datetime(df['date'])

df['volatility_rate'] = (df['close'] - df['close'].shift(late_cycles_1)) / df['close'].shift(late_cycles_1)
df['volatility_mean'] = df['volatility_rate'].rolling(window=late_cycles_2).mean()
df['volatility_mean'] = df['volatility_mean'].rolling(window=mean_cycles).mean()
df['change'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)

df = df[50:]

df['signal'] = np.where(
    (abs(df['change']) >= volatility) & (df['volatility_rate'].shift(1) < 0) & (df['volatility_rate'] > 0), 'long',
    'wait')
df['signal'] = np.where(
    (abs(df['change']) >= volatility) & (df['volatility_rate'] > 0) & (df['volatility_mean'] > 0) & (
            df['volatility_rate'].shift(1) < df['volatility_mean'].shift(1)) & (
            df['volatility_rate'] > df['volatility_mean']), 'long', df['signal'])
df['signal'] = np.where(
    (abs(df['change']) >= volatility) & (df['volatility_rate'].shift(1) > 0) & (df['volatility_rate'] < 0), 'short',
    df['signal'])
df['signal'] = np.where(
    (abs(df['change']) >= volatility) & (df['volatility_rate'] < 0) & (df['volatility_mean'] < 0) & (
            df['volatility_rate'].shift(1) > df['volatility_mean'].shift(1)) & (
            df['volatility_rate'] < df['volatility_mean']), 'short', df['signal'])

df['signal'] = np.where(
    (df['signal'] == 'wait') & (df['volatility_rate'].shift(1) > df['volatility_mean'].shift(1)) & (
            df['volatility_rate'] < df['volatility_mean']), 'close_long', df['signal'])
df['signal'] = np.where((df['signal'] == 'wait') & (df['change'] < 0) & (df['change'].shift(1) > 0) & (
        abs(df['change'] - df['change'].shift(1)) > long_rate), 'close_long', df['signal'])
df['signal'] = np.where(
    (df['signal'] == 'wait') & (df['volatility_rate'].shift(1) < df['volatility_mean'].shift(1)) & (
            df['volatility_rate'] > df['volatility_mean']), 'close_short', df['signal'])
df['signal'] = np.where((df['signal'] == 'wait') & (df['change'] > 0) & (df['change'].shift(1) < 0) & (
        abs(df['change'] - df['change'].shift(1)) > short_rate),
                        'close_short', df['signal'])

df_ = df[['date', 'close', 'volatility_rate', 'volatility_mean', 'signal', 'timestamp']].loc[
    df['signal'] != 'wait']
df_['signal'] = np.where(df_['signal'].shift(1) == df_['signal'], 'wait', df_['signal'])
df_['signal'] = np.where((df_['signal'].shift(1) == 'close_long') & (df_['signal'] == 'close_short'), 'wait',
                         df_['signal'])
df_['signal'] = np.where((df_['signal'].shift(1) == 'close_short') & (df_['signal'] == 'close_long'), 'wait',
                         df_['signal'])
df_ = df_.loc[df_['signal'] != 'wait']
if (df_[:1]['signal'] == 'close_long').bool() or (df_[:1]['signal'] == 'close_short').bool():
    df_ = df_[1:]
if (df_[-1:]['signal'] == 'long').bool() or (df_[-1:]['signal'] == 'short').bool():
    df_ = df_[:len(df_) - 1]

df_[['close']] = round(df_[['close']], 1)
# df_[['volatility_rate', 'volatility_mean', 'volatility_std']] = round(
#     df_[['volatility_rate', 'volatility_mean', 'volatility_std']], 5)
df_[['date', 'close', 'signal']].loc[df_['signal'] != 'wait'].to_csv('../data/volatility_rate.csv', index=False)

l = []
market_rate = 0.00075
market_yield_ = 0.0  # 记录连续亏损收益
total_yield = 0.0
max_drawdown = 0.0  # 最大回撤
i = 1
while i < len(df_):
    md = 0
    market_yield = 0.0
    row_ = df_.iloc[i - 1]
    row = df_.iloc[i]
    if row['signal'] == 'close_long' and row_['signal'] == 'long':
        market_yield = (row['close'] - row_['close']) / row['close'] - market_rate * 2
        part_df = df.loc[(df['timestamp'] > row_['timestamp']) & (df['timestamp'] <= row['timestamp'])]
        min_price = min(part_df.low)
        md = (row_['close'] - min_price) / row_['close']
        l.append([row['date'], row['close'], market_yield, 'long', 1])
    elif row['signal'] == 'close_short' and row_['signal'] == 'short':
        market_yield = (row_['close'] - row['close']) / row['close'] - market_rate * 2
        part_df = df.loc[(df['timestamp'] > row_['timestamp']) & (df['timestamp'] <= row['timestamp'])]
        max_price = min(part_df.high)
        md = (max_price - row_['close']) / row_['close']
        l.append([row['date'], row['close'], market_yield, 'short', -1])
    elif row['signal'] == 'short' and row_['signal'] == 'long':
        market_yield = (row['close'] - row_['close']) / row['close'] - market_rate * 2
        part_df = df.loc[(df['timestamp'] > row_['timestamp']) & (df['timestamp'] <= row['timestamp'])]
        min_price = min(part_df.low)
        md = (row_['close'] - min_price) / row_['close']
        l.append([row['date'], row['close'], market_yield, 'long', 1])
        i -= 1
    elif row['signal'] == 'long' and row_['signal'] == 'short':
        market_yield = (row_['close'] - row['close']) / row['close'] - market_rate * 2
        part_df = df.loc[(df['timestamp'] > row_['timestamp']) & (df['timestamp'] <= row['timestamp'])]
        max_price = min(part_df.high)
        md = (max_price - row_['close']) / row_['close']
        l.append([row['date'], row['close'], market_yield, 'short', -1])
        i -= 1

    if market_yield < 0:
        md = max(md, md - market_yield_)
        market_yield_ += market_yield
    else:
        market_yield_ = 0.0
    max_drawdown = max(md, max_drawdown)
    total_yield += market_yield
    i += 2

result = 0.5 * (max_drawdown - total_yield)
print(max_drawdown, total_yield)
print(result)
profits = pd.DataFrame(l, columns=['date', 'close', 'money', 'type', 's'])

ax = profits[['close', 'money']].plot(figsize=(20, 10), grid=True, xticks=profits.index, rot=90, subplots=True,
                                      style='b')

# 设置x轴刻度数量
ax[0].xaxis.set_major_locator(ticker.MaxNLocator(40))
# 以date为x轴刻度
ax[0].set_xticklabels(profits.date)
# 美观x轴刻度
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title(filename)
# plt.savefig('../data/' + filename + '.png')
plt.show()
