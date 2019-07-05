# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2019/4/11 15:50

@desc: 

'''

from tools import data2df
from tools.stockstats import StockDataFrame
import numpy as np
import pandas as pd
from tools.BmBackTest import BmBackTest


def public_func(df, signal, signal_key):
    # df['signal_boll'] = 0
    # df['signal_boll_lb'] = 0
    # df['signal_boll_ub'] = 0
    not_wait = len(df.loc[df[signal_key] != 'wait'])
    df['signal_boll'] = np.where(df[signal_key] == signal, df['boll'], 0)
    df['signal_boll_lb'] = np.where(df[signal_key] == signal, df['boll_lb'], 0)
    df['signal_boll_ub'] = np.where(df[signal_key] == signal, df['boll_ub'], 0)

    if not_wait == 0:
        return df

    df_ = df.loc[df[signal_key] == signal]
    for i in range(1, len(df_)):
        row = df_.iloc[i]
        row_ = df_.iloc[i - 1]
        df['signal_boll'] = np.where(
            (df['signal_boll'] == 0) & (df['timestamp'] > row_['timestamp']) & (df['timestamp'] < row['timestamp']),
            row_['boll'], df['signal_boll'])
        df['signal_boll_lb'] = np.where(
            (df['signal_boll_lb'] == 0) & (df['timestamp'] > row_['timestamp']) & (
                    df['timestamp'] < row['timestamp']),
            row_['boll_lb'], df['signal_boll_lb'])
        df['signal_boll_ub'] = np.where(
            (df['signal_boll_ub'] == 0) & (df['timestamp'] > row_['timestamp']) & (
                    df['timestamp'] < row['timestamp']),
            row_['boll_ub'], df['signal_boll_ub'])

    row1 = df.iloc[0]
    row2 = df_.iloc[len(df_) - 1]
    df['signal_boll'] = np.where((df['signal_boll'] == 0) & (df['timestamp'] >= row1['timestamp']), row1['boll'],
                                 df['signal_boll'])
    df['signal_boll_lb'] = np.where((df['signal_boll_lb'] == 0) & (df['timestamp'] >= row1['timestamp']),
                                    row1['boll_lb'], df['signal_boll_lb'])
    df['signal_boll_ub'] = np.where((df['signal_boll_ub'] == 0) & (df['timestamp'] >= row1['timestamp']),
                                    row1['boll_ub'], df['signal_boll_ub'])

    df['signal_boll'] = np.where((df['timestamp'] > row2['timestamp']), row2['boll'],
                                 df['signal_boll'])
    df['signal_boll_lb'] = np.where((df['timestamp'] > row2['timestamp']),
                                    row2['boll_lb'], df['signal_boll_lb'])
    df['signal_boll_ub'] = np.where((df['timestamp'] > row2['timestamp']),
                                    row2['boll_ub'], df['signal_boll_ub'])

    return df


def analysis(kline):
    df = pd.DataFrame(kline, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    stock = StockDataFrame.retype(df)
    # df['timestamp'] = df['timestamp'] / 1000
    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    df.index = df.date
    stock['boll']
    df['signal'] = 'wait'
    # 参数设置
    trend_A1 = 0.3
    trend_A2 = 0.02
    trend_B = 0.04
    trend_C = 0.05
    # trend_D = 0.07
    stop_trade_times = 5
    ts = 3600
    std_percentage = 0.6
    w = 200
    volatility = 0.001

    # 通道宽度
    df['boll_width'] = abs(df['boll_ub'] - df['boll_lb'])
    # 单次涨跌幅
    df['change'] = abs((df['close'] - df['close'].shift(1)) / df['close'].shift(1))
    # 两次涨幅
    df['change_2'] = abs((df['close'] - df['close'].shift(2)) / df['close'].shift(2))
    # 趋势判断A：超过通道一定幅度且单次涨幅超过一定幅度
    df['signal'] = np.where(((df['close'] > df['boll_ub'] + df['boll_width'] * trend_A1) | (
            df['close'] < df['boll_lb'] - df['boll_width'] * trend_A1)) & (df['change'] > trend_A2), 'trend',
                            df['signal'])
    # 趋势判断B：单次涨幅累计超过一定幅度
    df['signal'] = np.where(df['change'] > trend_B, 'trend', df['signal'])
    # 趋势判断C：两次涨幅累计超过一定幅度
    df['signal'] = np.where(df['change_2'] > trend_C, 'trend', df['signal'])

    for i in range(stop_trade_times):
        df['signal'] = np.where((df['signal'] == 'wait') & (df['signal'].shift(1 + i) == 'trend'), 'can_not_trade',
                                df['signal'])
    df['signal'] = np.where(df['signal'] == 'can_not_trade', 'trend', df['signal'])
    df['close_mean'] = df['close'].rolling(
        center=False, window=w).mean()
    df['close_std'] = df['close'].rolling(
        center=False, window=w).std()

    df['close_std'] = df['close_std'] * std_percentage
    df['rate'] = df['close_std'] / df['boll_width']
    # df['close_std'] = np.where(df['rate'] > 0.1, df['boll_width'] * 0.1, df['close_std'])

    # 策略1
    df['signal'] = np.where(
        (df['signal'] == 'wait') & (df['change'] > volatility) & (df['close'] > df['boll_ub'] - df['close_std'] / 4) & (
                df['close'] < df['boll_ub'] + df['close_std'] / 2), 'short', df['signal'])
    df = public_func(df, 'short', 'signal')
    df['signal'] = np.where((df['signal'] == 'wait') & (df['high'] >= df['signal_boll_ub'] + df['close_std'] / 4),
                            'close_short', df['signal'])

    # 策略2
    df['signal'] = np.where(
        (df['signal'] == 'wait') & (df['change'] > volatility) & (df['close'] < df['boll_lb'] + df['close_std'] / 4) & (
                df['close'] > df['boll_lb'] - df['close_std'] / 2), 'long', df['signal'])
    df = public_func(df, 'long', 'signal')
    df['signal'] = np.where((df['signal'] == 'wait') & (df['low'] <= df['signal_boll_lb'] - df['close_std'] / 4),
                            'close_long', df['signal'])

    # 策略3
    df['signal'] = np.where(
        (df['signal'] == 'wait') & (df['change'] > volatility) & (
                df['close'].shift(1) < df['boll'].shift(1) + df['close_std'].shift(1)) & (
                df['close'] > df['boll'] + df['close_std']), 'long', df['signal'])
    df['signal'] = np.where(
        (df['signal'] == 'wait') & (df['close'].shift(1) > df['boll'].shift(1) + df['close_std'].shift(1) / 2) & (
                df['close'] < df['boll'] + df['close_std'] / 2), 'close_long', df['signal'])
    # 策略4
    df['signal'] = np.where(
        (df['signal'] == 'wait') & (df['change'] > volatility) & (
                df['close'].shift(1) > df['boll'].shift(1) - df['close_std'].shift(1)) & (
                df['close'] < df['boll'] - df['close_std']), 'short', df['signal'])
    df['signal'] = np.where(
        (df['signal'] == 'wait') & (df['close'].shift(1) < df['boll'].shift(1) - df['close_std'].shift(1) / 2) & (
                df['close'] > df['boll'] - df['close_std'] / 2), 'close_short', df['signal'])

    return df.iloc[-1]


filename = 'BitMEX-170901-190606-1H'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)

datas = df.values

backtest = BmBackTest({
    'asset': 1
})

level = 1

for i in range(750, len(df)):
    test_df = datas[i - 750: i]
    row = analysis(test_df)

    if row['signal'] in ['long', 'short']:
        amount = int(backtest.asset * row['close'] * level)
        backtest.create_order(row['signal'], "market", row['close'], amount)

    elif row['signal'] == 'trend' or (row['signal'] == 'close_long' and backtest.side == 'long') or (
            row['signal'] == 'close_short' and backtest.side == 'short'):
        backtest.close_positions(row['close'], 'market')

    else:
        backtest.add_data(row['close'])

    print(i, row['close'], row['signal'], backtest.asset, backtest.float_profit, backtest.asset + backtest.float_profit)

backtest.show("BOLL")
