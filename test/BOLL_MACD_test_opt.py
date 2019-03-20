import pandas as pd
import numpy as np
from tools import data2df
from tools.stockstats import StockDataFrame
from tools import boll_macd_ga_tools
import time


def public_func(df, signal, signal_key):
    df['signal_boll'] = 0
    df['signal_boll_lb'] = 0
    df['signal_boll_ub'] = 0
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


bt = time.time()
filename = 'BTC2018-10-15-now-1H'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)
df['Timestamp'] = df['Timestamp'].astype(int)
df['date'] = pd.to_datetime(df['Timestamp'], unit='s')
df.index = df.date
stock = StockDataFrame.retype(df)
df['signal'] = 'wait'


# 参数设置
boll_A = 2  # BOLL 波动率倍数 1.5~3
boll_std_len = 20  # BOLL_STD 样本长度 14~30
macd_fast_len = 12  # MACD 快线长度 8~16
macd_slow_len = 6  # MACD 慢线长度 20~30
boll_width_threshold = 0.05  # BOLL 通道阈值 0.05~0.15 乘100倍
volatility = 0.0005  # 波动率下单 0.0005~0.0015 乘10000倍
stop_limit = 0.01  # 止盈止损阈值 0.01~0.1 乘100倍
# 趋势判断阈值
trend_A1 = 0.1  # 乘100倍
trend_A2 = 0.01  # 乘1000倍
trend_B = 0.02  # 乘1000倍
trend_C = 0.02  # 乘1000倍
stop_trade_times = 3
ts = 3600  # 1小时时间戳

# 获得macd，boll指标
boll_macd_ga_tools._get_boll(df, stock, boll_A, boll_std_len)
boll_macd_ga_tools._get_macd(df, stock, macd_fast_len, macd_slow_len)

# 通道宽度
df['boll_width'] = abs(df['boll_ub'] - df['boll_lb'])
# 单次涨跌幅
df['change'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
# 止盈止损
df['channel_stop_limit'] = df['boll_width'] * stop_limit
# 趋势判断A：超过通道一定幅度且单次涨幅超过一定幅度
df['signal'] = np.where(((df['close'] > df['boll_ub'] + df['boll_width'] * trend_A1) | (
        df['close'] < df['boll_lb'] - df['boll_width'] * trend_A1)) & (abs(df['change']) > trend_A2), 'trend',
                        df['signal'])
# 趋势判断B：单次涨幅累计超过一定幅度
df['signal'] = np.where(abs(df['change']) > trend_B, 'trend', df['signal'])
# 趋势判断A：两次涨幅累计超过一定幅度
df['signal'] = np.where(abs(df['change'] + df['change'].shift(1)) > trend_C, 'trend', df['signal'])

trend_df = df.loc[df['signal'] != 'wait']

for i in range(len(trend_df)):
    row = trend_df.iloc[i]
    # if 'trend' in row['signal']:
    if 'trend' == row['signal']:
        df['signal'] = np.where((df['signal'] == 'wait') & (df['timestamp'] >= row['timestamp']) & (
                df['timestamp'] <= row['timestamp'] + ts * stop_trade_times), 'can_not_trade', df['signal'])

df['signal'] = np.where(df['signal'] == 'can_not_trade', 'trend', df['signal'])

df['channel_limit'] = df['boll_width'] * boll_width_threshold

# 去掉数据错误记录
df = df[30:]

print('clear data used second:%.10f seconds' % (time.time() - bt))
bt = time.time()

# 策略1
df['signal1'] = np.where(
    (df['signal'] == 'wait') & (df['change'] >= volatility) & (df['boll_lb'].shift(1) <= df['close'].shift(1)) & (
            df['close'].shift(1) <= df['boll'].shift(1)) & (df['boll_lb'] <= df['close']) & (
            df['close'] <= df['boll']) & (
            df['close'] <= df['boll_lb'] + df['channel_limit']), 'long', 'wait')

df = public_func(df, 'long', 'signal1')
df['signal1'] = np.where((df['signal'] == 'wait') & (df['signal1'] == 'wait') & (
        (df['high'] >= df['signal_boll'] - df['channel_stop_limit']) | (
        df['low'] <= df['signal_boll_lb'] - df['channel_stop_limit'])), 'close_long', df['signal1'])

# 策略2
df['signal2'] = np.where(
    (df['signal'] == 'wait') & (df['change'] >= volatility) & (df['boll'].shift(1) >= df['close'].shift(1)) & (
            df['close'].shift(1) >= df['boll_lb'].shift(1)) & (
            df['boll'] + df['channel_limit'] * 4 > df['close']) & (
            df['close'] >= df['boll'] + df['channel_limit']), 'short', 'wait')
df = public_func(df, 'short', 'signal2')
df['signal2'] = np.where((df['signal'] == 'wait') & (df['signal2'] == 'wait') & (
        (df['low'] <= df['signal_boll_ub'] - df['channel_stop_limit']) | (
        df['high'] >= df['signal_boll'] + df['channel_stop_limit'])), 'close_short', df['signal2'])

# 策略3
df['signal3'] = np.where(
    (df['signal'] == 'wait') & (df['change'] >= volatility) & (df['macd'].shift(1) <= df['macds'].shift(1)) & (
            df['macd'] > df['macds']) & (
            df['close'].shift(1) < df['boll'].shift(1)) & (df['close'] > df['boll']), 'long',
    'wait')

df = public_func(df, 'long', 'signal3')

df['signal3'] = np.where(
    (df['signal'] == 'wait') & (df['signal3'] == 'wait') & (
            (df['high'] >= df['signal_boll_ub'] - df['channel_stop_limit']) | (
            df['low'] <= df['signal_boll'] - df['channel_stop_limit'])), 'close_long', df['signal3'])

# 策略4
df['signal4'] = np.where(
    (df['signal'] == 'wait') & (df['change'] >= volatility) & (df['macd'].shift(1) > df['macds'].shift(1)) & (
            df['macd'] < df['macds']) & (
            df['close'].shift(1) > df['boll'].shift(1)) & (df['close'] < df['boll']), 'short',
    'wait')

df = public_func(df, 'short', 'signal4')

df['signal4'] = np.where((df['signal'] == 'wait') & (df['signal4'] == 'wait') & (
        (df['low'] <= df['signal_boll_lb'] + df['channel_stop_limit']) | (
        df['high'] >= df['signal_boll'] + df['channel_stop_limit'])), 'close_short', df['signal4'])

# 策略5
df['signal5'] = np.where(
    (df['signal'] == 'wait') & (df['change'] >= volatility) & (df['boll_ub'].shift(1) >= df['close'].shift(1)) & (
            df['close'].shift(1) >= df['boll'].shift(1)) & (
            df['close'] >= df['boll_ub'] + df['channel_limit']), 'short', 'wait')

df = public_func(df, 'short', 'signal5')

df['signal5'] = np.where((df['signal'] == 'wait') & (df['signal5'] == 'wait') & (
        (df['low'] <= df['signal_boll'] + df['channel_stop_limit']) | (
        df['high'] <= df['signal_boll_ub'] + df['channel_stop_limit'])), 'close_short', df['signal5'])

# 策略6
df['signal6'] = np.where(
    (df['signal'] == 'wait') & (df['change'] >= volatility) & (df['boll'].shift(1) <= df['close'].shift(1)) & (
            df['close'].shift(1) <= df['boll_ub'].shift(1)) & (
            df['boll'] + df['channel_limit'] * 4 > df['close']) & (
            df['close'] >= df['boll'] + df['channel_limit']), 'long',
    'wait')

df = public_func(df, 'long', 'signal6')

df['signal6'] = np.where(
    (df['signal'] == 'wait') & (df['signal6'] == 'wait') & (
            (df['high'] >= df['signal_boll_ub'] - df['channel_stop_limit']) | (
            df['low'] <= df['signal_boll'] - df['channel_stop_limit'])), 'close_long', df['signal6'])

print('get signal used second:%.10f seconds' % (time.time() - bt))
bt = time.time()

last_signal = 'wait'
signal_num = -1
for i in range(len(df)):
    row = df.iloc[i]
    if signal_num == -1:
        for j in range(1, 7):
            if row['signal%d' % j] in ['long', 'short']:
                df['signal'].iloc[i] = row['signal%d' % j]
                signal_num = j
                last_signal = row['signal%d' % j]
                break
    elif signal_num > 0:
        if row['signal%d' % signal_num] == 'close_' + last_signal:
            df['signal'].iloc[i] = row['signal%d' % j]
            signal_num = -1
            last_signal = 'wait'
# 删除无用字段
del df['signal1']
del df['signal2']
del df['signal3']
del df['signal4']
del df['signal5']
del df['signal6']
del df['signal_boll_lb']
del df['signal_boll_ub']
del df['signal_boll']
del df['channel_limit']
del df['channel_stop_limit']
del df['change']
del df['boll_width']

df['signal'] = np.where((df['signal'] == 'trend') & (df['signal'].shift(1) == 'short'), 'close_short', df['signal'])
df['signal'] = np.where((df['signal'] == 'trend') & (df['signal'].shift(1) == 'long'), 'close_long', df['signal'])
df['signal'] = np.where(
    ((df['signal'] == 'close_long') | (df['signal'] == 'close_short')) & (df['signal'].shift(1) == 'trend'), 'wait',
    df['signal'])

signal_df = df.loc[(df['signal'] != 'wait') & (df['signal'] != 'trend')]
if (signal_df[:1]['signal'] == 'close_long').bool() or (signal_df[:1]['signal'] == 'close_short').bool():
    signal_df = signal_df[1:]
if (signal_df[-1:]['signal'] == 'long').bool() or (signal_df[-1:]['signal'] == 'short').bool():
    signal_df = signal_df[:len(signal_df) - 1]

print('clear singal used second:%.10f seconds' % (time.time() - bt))
bt = time.time()

# df[['signal']].loc[(df['signal'] != 'wait') & (df['signal'] != 'trend')].to_csv('../data/signal.csv')
# df[['signal', 'close']].loc[(df['signal'] != 'wait')].to_csv('../data/signal.csv')
# signal_df[['signal', 'close']].loc[(df['signal'] != 'wait')].to_csv('../data/signal2.csv')

# 初始化
limit_rate = 0.00025  # 限价单
market_rate = 0.00075  # 市价单
market_yield_ = 0.0  # 记录连续亏损收益
total_yield = 0.0
max_drawdown = 0.0  # 最大回撤

for i in range(1, len(signal_df), 2):
    md = 0  # 本次交易最大回撤
    market_yield = 0.0  # 收益率
    row_ = signal_df.iloc[i - 1]
    row = signal_df.iloc[i]
    if row['signal'] == 'close_long' and row_['signal'] == 'long':
        market_yield = (row['close'] - row_['close']) / row['close'] - market_rate - limit_rate
        part_df = df.loc[(df['timestamp'] > row_['timestamp']) & (df['timestamp'] <= row['timestamp'])]
        min_price = min(part_df.low)
        md = (row_['close'] - min_price) / row_['close']

    elif row['signal'] == 'close_short' and row_['signal'] == 'short':
        market_yield = (row_['close'] - row['close']) / row['close'] - market_rate - limit_rate
        part_df = df.loc[(df['timestamp'] > row_['timestamp']) & (df['timestamp'] <= row['timestamp'])]
        max_price = min(part_df.high)
        md = (max_price - row_['close']) / row_['close']

    if market_yield < 0:
        md = max(md, md - market_yield_)
        market_yield_ += market_yield
    else:
        market_yield_ = 0.0
    max_drawdown = max(md, max_drawdown)
    total_yield += market_yield

fit = 0.5 * (max_drawdown - total_yield)
print('calculate use second:%.10f seconds' % (time.time() - bt))
