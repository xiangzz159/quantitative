from tools import volatility_ga_tools
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
    late_cycles_1 = params[0]
    late_cycles_2 = params[1]
    mean_cycles = params[2]
    volatility = params[3]
    long_rate = params[4]
    short_rate = params[5]

    df['volatility_rate'] = (df['close'] - df['close'].shift(late_cycles_1)) / df['close'].shift(late_cycles_1)
    df['volatility_mean'] = df['volatility_rate'].rolling(window=late_cycles_2).mean()
    df['volatility_mean'] = df['volatility_mean'].rolling(window=mean_cycles).mean()
    # 单次涨跌幅
    df['change'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)

    df = df[-168:]

    df['signal'] = np.where(
        (abs(df['change']) >= volatility) & (df['volatility_rate'].shift(1) < 0) & (df['volatility_rate'] > 0), 'long',
        'wait')
    df['signal'] = np.where(
        (abs(df['change']) >= volatility) & (df['volatility_rate'] > 0) & (df['volatility_mean'] > 0) & (
                df['volatility_rate'].shift(1) < df['volatility_mean'].shift(1)) & (
                df['volatility_rate'] > df['volatility_mean']), 'long', df['signal'])
    df['signal'] = np.where(
        (df['signal'] == 'wait') & (df['volatility_rate'].shift(1) > df['volatility_mean'].shift(1)) & (
                df['volatility_rate'] < df['volatility_mean']), 'close_long', df['signal'])
    df['signal'] = np.where((df['signal'] == 'wait') & (df['change'] < 0) & (df['change'].shift(1) > 0) & (
            abs(df['change'] - df['change'].shift(1)) > long_rate), 'close_long', df['signal'])

    df['signal'] = np.where(
        (abs(df['change']) >= volatility) & (df['volatility_rate'].shift(1) > 0) & (df['volatility_rate'] < 0), 'short',
        df['signal'])
    df['signal'] = np.where(
        (abs(df['change']) >= volatility) & (df['volatility_rate'] < 0) & (df['volatility_mean'] < 0) & (
                df['volatility_rate'].shift(1) > df['volatility_mean'].shift(1)) & (
                df['volatility_rate'] < df['volatility_mean']), 'short', df['signal'])

    df['signal'] = np.where(
        (df['signal'] == 'wait') & (df['volatility_rate'].shift(1) < df['volatility_mean'].shift(1)) & (
                df['volatility_rate'] > df['volatility_mean']), 'close_short', df['signal'])
    df['signal'] = np.where((df['signal'] == 'wait') & (df['change'] > 0) & (df['change'].shift(1) < 0) & (
            abs(df['change'] - df['change'].shift(1)) > short_rate),
                            'close_short', df['signal'])

    return df.iloc[-1]


if __name__ == '__main__':
    pop_size, chromosome_length = 50, 10
    pops = volatility_ga_tools.init_pops(pop_size, chromosome_length)
    iter = 10
    pc = 0.6  # 杂交概率
    pm = 0.01  # 变异概率
    results = []  # 存储每一代的最优解，N个二元组
    filename = 'BTC2018-04-01-now-1H.csv'
    lines = list(csv.reader(open(r'/root/quant/data/' + filename)))
    # lines = list(csv.reader(open(r'./data/' + filename)))
    header, values = lines[0], lines[1:]
    data_dict = {h: v for h, v in zip(header, zip(*values))}
    ori_df = pd.DataFrame(data_dict)
    ori_df = ori_df.astype(float)
    stock = StockDataFrame.retype(ori_df)
    ori_df['timestamp'] = ori_df['timestamp'].astype(int)
    ori_df['date'] = pd.to_datetime(ori_df['timestamp'], unit='s')
    ori_df.index = ori_df.date
    re_total_yield = []
    last_signal = 'wait'

    last_trade_price = 0.0
    last_yield = 1.0

    i = 301
    signal = 'wait'
    last_trade_price = 0.
    side = None
    while i < len(ori_df):
        test_df = ori_df[i - 300: i - 1]
        best_individuals = []
        best_fits = []
        for j in range(iter):
            obj_values = volatility_ga_tools.cal_fitness(test_df, pops, pop_size, chromosome_length)  # 计算绩效
            fit_values = volatility_ga_tools.clear_fit_values(obj_values)
            best_individual, best_fit = volatility_ga_tools.find_best(pops, fit_values,
                                                                      chromosome_length)  # 第一个是最优基因序列, 第二个是对应的最佳个体适度
            best_individuals.append(best_individual)
            best_fits.append(best_fit)
            results.append([best_individual, best_fit])
            volatility_ga_tools.selection(pops, fit_values)  # 选择
            volatility_ga_tools.crossover(pops, pc)  # 染色体交叉（最优个体之间进行0、1互换）
            volatility_ga_tools.mutation(pops, pm)  # 染色体变异（其实就是随机进行0、1取反
        last_individual = best_individuals[-1]

        df = ori_df[i - 300: i]
        last_row = get_signal(df, last_individual)
        signal = last_row['signal']

        if signal in ['long', 'short']:
            if side is None:
                side = signal
                last_trade_price = last_row['close']
            else:
                if side != signal:
                    earnings = (last_row['close'] - last_trade_price) / last_trade_price if side == 'long' else (
                                                                                                                        last_trade_price -
                                                                                                                        last_row[
                                                                                                                            'close']) / last_trade_price
                    last_yield = last_yield * (1 + earnings - 0.00075 * 2)
                    last_trade_price = last_row['close']
                    side = signal
        if signal in ['close_long', 'close_short'] and side is not None:
            earnings = (last_row['close'] - last_trade_price) / last_trade_price if side == 'long' else (
                                                                                                                last_trade_price -
                                                                                                                last_row[
                                                                                                                    'close']) / last_trade_price
            last_yield = last_yield * (1 + earnings - 0.00075 * 2)
            last_trade_price = 0.0
            signal = 'wait'
            side = None
        # print(i, signal, last_yield, last_trade_price, side, last_row['close'], last_row['signal'])
        re_total_yield.append(last_yield)
        if i % 50 == 0:
            print('\n', re_total_yield)
        i += 1
