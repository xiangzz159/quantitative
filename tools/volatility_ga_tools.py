import numpy as np
from tools.stockstats import StockDataFrame
import random
import copy
import pandas as pd
from datetime import datetime
from tools import public_tools
import asyncio


# 初始化种群pop_size：种群数量，chromsome_length：基因长度
def init_population(pop_size, chromosome_length):
    # 形如[[0,1,..0,1],[0,1,..0,1]...]
    pop = [[random.randint(0, 1) for i in range(chromosome_length)] for j in range(pop_size)]
    return pop


# 计算2进制序列代表的数值
def binary2decimal(binary, lower_limit, upper_limit, chromosome_length):
    t = 0
    for j in range(len(binary)):
        t += binary[j] * 2 ** j
    t = lower_limit + t * (upper_limit - lower_limit) / (2 ** chromosome_length - 1)
    return t


# 初始化种群
def init_pops(pop_size=50, chromosome_length=6):
    late_cycles_1 = init_population(pop_size, chromosome_length)
    late_cycles_2 = init_population(pop_size, chromosome_length)
    mean_cycles = init_population(pop_size, chromosome_length)
    volatility = init_population(pop_size, chromosome_length)
    short_rate = init_population(pop_size, chromosome_length)
    long_rate = init_population(pop_size, chromosome_length)
    return [
        late_cycles_1,
        late_cycles_2,
        mean_cycles,
        volatility,
        long_rate,
        short_rate
    ]


# 淘汰
def clear_fit_values(obj_values):
    fit_value = []
    # 去掉大于0的值，更改c_min会改变淘汰的上限
    # 比如设成10可以加快收敛
    # 但是如果设置过大，有可能影响了全局最优的搜索
    c_min = 0
    for value in obj_values:
        temp = value if value <= c_min else 0
        fit_value.append(temp)
    # fit_value保存的是活下来的值
    return fit_value


# 轮赌法选择
def selection(pops, fit_value):
    # https://blog.csdn.net/pymqq/article/details/51375522

    p_fit_value = []
    # 适应度总和
    total_fit = sum(fit_value)
    if total_fit == 0:
        return
    # 归一化，使概率总和为1
    for i in range(len(fit_value)):
        p_fit_value.append(fit_value[i] / total_fit)
    # 概率求和排序

    # https://www.cnblogs.com/LoganChen/p/7509702.html
    cum_sum(p_fit_value)
    pop_len = len(pops[0])
    # 类似搞一个转盘吧下面这个的意思
    ms = sorted([random.random() for i in range(pop_len)])
    fitin = 0
    newin = 0
    newpop = pops[:][:]
    # 转轮盘选择法
    while newin < pop_len:
        # 如果这个概率大于随机出来的那个概率，就选这个
        if (ms[newin] < p_fit_value[fitin]):
            for i in range(len(pops)):
                newpop[i][newin] = pops[i][fitin]
            newin = newin + 1
        else:
            fitin = fitin + 1
    # 这里注意一下，因为random.random()不会大于1，所以保证这里的newpop规格会和以前的一样
    # 而且这个pop里面会有不少重复的个体，保证种群数量一样

    # 之前是看另一个人的程序，感觉他这里有点bug，要适当修改
    pop = newpop[:][:]


# 计算累计概率
def cum_sum(fit_value):
    # 输入[1, 2, 3, 4, 5]，返回[1,3,6,10,15]，matlab的一个函数
    # 这个地方遇坑，局部变量如果赋值给引用变量，在函数周期结束后，引用变量也将失去这个值
    temp = fit_value[:]
    for i in range(len(temp)):
        fit_value[i] = (sum(temp[:i + 1]))


# 杂交
def crossover(pops, pc):
    # 一定概率杂交，主要是杂交种群种相邻的两个个体
    pop_len = len(pops[0])
    for i in range(pop_len - 1):
        # 随机看看达到杂交概率没
        if (random.random() < pc):
            for j in range(len(pops)):
                # 随机选取杂交点，然后交换数组
                cpoint = random.randint(0, len(pops[j][i]))
                temp1 = []
                temp2 = []
                temp1.extend(pops[j][i][0:cpoint])
                temp1.extend(pops[j][i + 1][cpoint:len(pops[j][i])])
                temp2.extend(pops[j][i + 1][0:cpoint])
                temp2.extend(pops[j][i][cpoint:len(pops[j][i])])
                pops[j][i] = temp1[:]
                pops[j][i + 1] = temp2[:]


# 基因突变
def mutation(pops, pm):
    px = len(pops[0])
    py = len(pops[0][0])
    # 每条染色体随便选一个杂交
    for i in range(px):
        if (random.random() < pm):
            mpoint = random.randint(0, py - 1)
            for j in range(len(pops)):
                if (pops[j][i][mpoint] == 1):
                    pops[j][i][mpoint] = 0
                else:
                    pops[j][i][mpoint] = 1


# 找出最优解和最优解的基因编码
def find_best(pop, fit_values, chromosome_length=6):
    # 绩效越小越好
    best_fit = min(fit_values)
    best_fit_idx = fit_values.index(best_fit)
    late_cycles_1 = int(binary2decimal(pop[0][best_fit_idx], 12, 24, chromosome_length))
    late_cycles_2 = int(binary2decimal(pop[1][best_fit_idx], 12, 24, chromosome_length))
    mean_cycles = int(binary2decimal(pop[2][best_fit_idx], 1, 9, chromosome_length))
    volatility = binary2decimal(pop[3][best_fit_idx], 0.001, 0.002, chromosome_length)
    long_rate = binary2decimal(pop[4][best_fit_idx], 0.001, 0.002, chromosome_length)
    short_rate = binary2decimal(pop[5][best_fit_idx], 0.001, 0.002, chromosome_length)
    # 用来存最优基因编码
    best_individual = [late_cycles_1, late_cycles_2, mean_cycles, volatility, long_rate, short_rate]
    return best_individual, best_fit


def cal_fitness(df, pops, pop_size=50, chromosome_length=6):
    obj_value = []
    async_funcs = []
    for i in range(pop_size):
        df_ = copy.deepcopy(df)
        late_cycles_1 = int(binary2decimal(pops[0][i], 12, 24, chromosome_length))
        late_cycles_2 = int(binary2decimal(pops[1][i], 12, 24, chromosome_length))
        mean_cycles = int(binary2decimal(pops[2][i], 1, 12, chromosome_length))
        volatility = binary2decimal(pops[3][i], 0.001, 0.002, chromosome_length)
        long_rate = binary2decimal(pops[4][i], 0.001, 0.008, chromosome_length)
        short_rate = binary2decimal(pops[5][i], 0.001, 0.008, chromosome_length)
        async_funcs.append(
            cal_someone_fitness(df_, i, late_cycles_1, late_cycles_2, mean_cycles, volatility, long_rate, short_rate))
    loop = asyncio.get_event_loop()
    re = loop.run_until_complete(asyncio.gather(*async_funcs))
    n = len(re)
    for i in range(n - 1, 0, -1):
        for j in range(i):
            if re[j][0] > re[j + 1][0]:
                re[j], re[j + 1] = re[j + 1], re[j]
    for i in range(n):
        obj_value.append(re[i][1])
    return obj_value


# 计算适应度（绩效）
async def cal_someone_fitness(df,
                              index, late_cycles_1, late_cycles_2, mean_cycles, volatility,
                              long_rate, short_rate, data_len=-168):
    df['volatility_rate'] = (df['close'] - df['close'].shift(late_cycles_1)) / df['close'].shift(late_cycles_1)
    df['volatility_mean'] = df['volatility_rate'].rolling(window=late_cycles_2).mean()
    df['volatility_mean'] = df['volatility_mean'].rolling(window=mean_cycles).mean()
    # 单次涨跌幅
    df['change'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)

    df = df[data_len:]

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
    return index, result
