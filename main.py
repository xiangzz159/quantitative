from tools import boll_macd_ga_tools
import pandas as pd
import time
from tools.stockstats import StockDataFrame
import csv
import numpy as np
from datetime import datetime


# 模拟算法实盘回测

def get_time():
    timestamp = time.time()
    time_local = time.localtime(timestamp)
    return time.strftime("%Y-%m-%d %H:%M", time_local)


def get_signal(df, params):
    df['signal'] = 'wait'
    stock = StockDataFrame.retype(df)
    boll_A = params[0]
    boll_std_len = params[1]
    macd_fast_len = params[2]
    macd_slow_len = params[3]
    boll_width_threshold = params[4]
    volatility = params[5]
    stop_limit = params[6]
    trend_A1 = 0.3
    trend_A2 = 0.02
    trend_B = 0.04
    trend_C = 0.05
    stop_trade_times = 5
    ts = 3600
    boll_macd_ga_tools._get_boll(df, stock, boll_A, boll_std_len)
    boll_macd_ga_tools._get_macd(df, stock, macd_fast_len, macd_slow_len)

    # 通道宽度
    df['boll_width'] = abs(df['boll_ub'] - df['boll_lb'])
    # 单次涨跌幅
    df['change'] = abs((df['close'] - df['close'].shift(1)) / df['close'].shift(1))
    # 止盈止损
    df['channel_stop_limit'] = df['boll_width'] * stop_limit

    df = df[-168:]

    # 趋势判断A：超过通道一定幅度且单次涨幅超过一定幅度
    df['signal'] = np.where(((df['close'] > df['boll_ub'] + df['boll_width'] * trend_A1) | (
            df['close'] < df['boll_lb'] - df['boll_width'] * trend_A1)) & (df['change'] > trend_A2), 'trend',
                            df['signal'])
    # 趋势判断B：单次涨幅累计超过一定幅度
    df['signal'] = np.where(df['change'] > trend_B, 'trend', df['signal'])
    # 趋势判断A：两次涨幅累计超过一定幅度
    df['signal'] = np.where(df['change'] + df['change'].shift(1) > trend_C, 'trend', df['signal'])

    trend_df = df.loc[df['signal'] != 'wait']

    tts = []
    last_row_timestamp = df.iloc[-1]['timestamp']
    for idx, row in trend_df.iterrows():
        last_timestamp = tts[-1].timestamp() if len(tts) > 0 else 0
        t = idx.timestamp()
        for i in range(stop_trade_times):
            if t + (i + 1) * ts > last_timestamp and t + (i + 1) * ts <= last_row_timestamp:
                tts.append(pd.Timestamp(datetime.utcfromtimestamp(t + (i + 1) * ts)))
    df.loc[tts, 'signal'] = 'can_not_trade'
    df['signal'] = np.where(df['signal'] == 'can_not_trade', 'trend', df['signal'])

    df['channel_limit'] = df['boll_width'] * boll_width_threshold
    df['channel_limit_3'] = df['channel_limit'] * 3

    # 策略1
    df['signal1'] = np.where(
        (df['signal'] == 'wait') & (df['change'] >= volatility) & (df['boll_lb'].shift(1) <= df['close'].shift(1)) & (
                df['close'].shift(1) <= df['boll'].shift(1)) & (
                df['close'] <= df['boll_lb'] + df['channel_limit']) & (
                df['close'] >= df['boll_lb'] - df['channel_limit']), 'long', 'wait')

    df = boll_macd_ga_tools.public_func(df, 'long', 'signal1')
    df['signal1'] = np.where((df['signal'] == 'trend') | ((df['signal1'] == 'wait') & (
            (df['high'] >= df['signal_boll'] - df['channel_stop_limit']) | (
            df['low'] <= df['signal_boll_lb'] - df['channel_stop_limit']))), 'close_long', df['signal1'])

    # 策略2&4
    df['signal2'] = np.where(
        (df['signal'] == 'wait') & (df['change'] >= volatility) & (((df['boll'].shift(1) >= df['close'].shift(1)) & (
                df['close'].shift(1) >= df['boll_lb'].shift(1)) & (
                                                                            df['boll'] + df['channel_limit_3'] > df[
                                                                        'close']) & (
                                                                            df['close'] >= df['boll'])) | (
                                                                               (df['macd'].shift(1) > df[
                                                                                   'macds'].shift(1)) & (
                                                                                       df['macd'] < df[
                                                                                   'macds']) & (
                                                                                       df['close'].shift(
                                                                                           1) > df[
                                                                                           'boll'].shift(
                                                                                   1)) & (df['close'] <
                                                                                          df['boll']))),
        'short', 'wait')

    df = boll_macd_ga_tools.public_func(df, 'short', 'signal2')
    df['signal2'] = np.where((df['signal'] == 'trend') | ((df['signal2'] == 'wait') & (
            (df['low'] <= df['signal_boll_ub'] - df['channel_stop_limit']) | (
            df['high'] >= df['signal_boll'] + df['channel_stop_limit']))), 'close_short', df['signal2'])

    # 策略3&6
    df['signal3'] = np.where(
        (df['signal'] == 'wait') & (df['change'] >= volatility) & (((df['boll'].shift(1) <= df['close'].shift(1)) & (
                df['close'].shift(1) <= df['boll_ub'].shift(1)) & (
                                                                            df['boll'] + df['channel_limit_3'] > df[
                                                                        'close']) & (
                                                                            df['close'] >= df['boll'] - df[
                                                                        'channel_limit'])) | ((df['macd'].shift(1) <=
                                                                                               df['macds'].shift(1)) & (
                                                                                                      df['macd'] > df[
                                                                                                  'macds']) & (
                                                                                                      df['close'].shift(
                                                                                                          1) < df[
                                                                                                          'boll'].shift(
                                                                                                  1)) & (df['close'] >
                                                                                                         df['boll']))),
        'long',
        'wait')

    df = boll_macd_ga_tools.public_func(df, 'long', 'signal3')

    df['signal3'] = np.where(
        (df['signal'] == 'trend') | ((df['signal3'] == 'wait') & (
                (df['high'] >= df['signal_boll_ub'] - df['channel_stop_limit']) | (
                df['low'] <= df['signal_boll'] - df['channel_stop_limit']))), 'close_long', df['signal3'])

    # 策略5
    df['signal4'] = np.where(
        (df['signal'] == 'wait') & (df['change'] >= volatility) & (df['boll_ub'].shift(1) >= df['close'].shift(1)) & (
                df['close'].shift(1) >= df['boll'].shift(1)) & (
                df['close'] >= df['boll_ub']) & (
                df['close'] <= df['boll_ub'] + df['channel_limit']), 'short', 'wait')

    df = boll_macd_ga_tools.public_func(df, 'short', 'signal4')

    df['signal4'] = np.where((df['signal'] == 'trend') | ((df['signal4'] == 'wait') & (
            (df['low'] <= df['signal_boll'] + df['channel_stop_limit']) | (
            df['high'] >= df['signal_boll_ub'] + df['channel_stop_limit']))), 'close_short', df['signal4'])

    return df.iloc[-1]


if __name__ == '__main__':
    pop_size, chromosome_length = 30, 6
    pops = boll_macd_ga_tools.init_pops(pop_size, chromosome_length)
    iter = 10  # TODO 迭代次数 10
    pc = 0.6  # 杂交概率
    pm = 0.01  # 变异概率
    results = []  # 存储每一代的最优解，N个二元组
    filename = 'BTC2018-04-01-now-1H.csv'
    # lines = list(csv.reader(open(r'/home/centos/quantitative/data/' + filename)))
    lines = list(csv.reader(open(r'./data/' + filename)))
    header, values = lines[0], lines[1:]
    data_dict = {h: v for h, v in zip(header, zip(*values))}
    ori_df = pd.DataFrame(data_dict)
    ori_df = ori_df.astype(float)
    ori_df['Timestamp'] = ori_df['Timestamp'].astype(int)
    ori_df['date'] = pd.to_datetime(ori_df['Timestamp'], unit='s')
    ori_df.index = ori_df.date
    re_total_yield = []
    last_signal = 'wait'

    last_individual = None
    last_trade_price = 0.0
    last_yield = 1.0

    i = 721
    signal = 'wait'
    signal_num = 0
    last_trade_price = 0.
    side = None
    while i < len(ori_df):
        if last_individual is None:
            test_df = ori_df[i - 300: i - 1]
            best_individuals = []
            best_fits = []
            for j in range(iter):
                obj_values = boll_macd_ga_tools.cal_fitness(test_df, pops, pop_size, chromosome_length)  # 计算绩效
                fit_values = boll_macd_ga_tools.clear_fit_values(obj_values)
                best_individual, best_fit = boll_macd_ga_tools.find_best(pops, fit_values,
                                                                         chromosome_length)  # 第一个是最优基因序列, 第二个是对应的最佳个体适度
                best_individuals.append(best_individual)
                best_fits.append(best_fit)
                results.append([best_individual, best_fit])
                boll_macd_ga_tools.selection(pops, fit_values)  # 选择
                boll_macd_ga_tools.crossover(pops, pc)  # 染色体交叉（最优个体之间进行0、1互换）
                boll_macd_ga_tools.mutation(pops, pm)  # 染色体变异（其实就是随机进行0、1取反
            # best_fit = min(best_fits)
            # best_fit_idx = best_fits.index(best_fit)
            last_individual = best_individuals[-1]

        df = ori_df[i - 300: i]
        last_row = get_signal(df, last_individual)

        if signal_num > 0:
            signal = last_row['signal%d' % signal_num]
        else:
            for j in range(1, 5):
                if last_row['signal%d' % j] in ['long', 'short'] and signal_num == 0:
                    signal = last_row['signal%d' % j]
                    signal_num = j
                    break

        if last_row['signal'] == 'trend' and last_trade_price > 0:
            earnings = (last_row['close'] - last_trade_price) / last_trade_price if side == 'long' else (
                                                                                                                last_trade_price -
                                                                                                                last_row[
                                                                                                                    'close']) / last_trade_price
            last_yield = last_yield * (1 + earnings - 0.001)
            last_individual = None
            last_trade_price = 0.0
            signal_num = 0
            signal = 'wait'
            side = None

        elif signal == 'wait' and last_trade_price <= 0:
            last_individual = None if last_trade_price <= 0 else last_individual
        elif signal in ['short', 'long'] and last_trade_price <= 0:
            last_trade_price = last_row['close']
            side = signal
        elif signal in ['close_long', 'close_short'] and last_trade_price > 0:
            earnings = (last_row['close'] - last_trade_price) / last_trade_price if side == 'long' else (
                                                                                                                last_trade_price -
                                                                                                                last_row[
                                                                                                                    'close']) / last_trade_price
            last_yield = last_yield * (1 + earnings - 0.001)
            last_individual = None
            last_trade_price = 0.0
            signal_num = 0
            signal = 'wait'
            side = None
        re_total_yield.append(last_yield)
        if i % 50 == 0:
            print('\n', re_total_yield)
        i += 1


try:
    print(1)
except Exception as e:
    print(e)