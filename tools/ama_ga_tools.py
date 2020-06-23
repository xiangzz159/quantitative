# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2020/6/20 9:30

@desc: 需要遗传优化的参数
wn_rate: 波峰波谷比率 （0~1）
cci_w: cci窗口
dir_w：direction窗口
vola_w： 波动长度
ama_std_w： ama_std窗口
percentage：filter百分比
atr_len: atr长度
stop_atr： atr止损
fastest: 快线
slowest：慢线

'''
import pandas as pd
import numpy as np
import heapq
from stockstats import StockDataFrame
from stockstats import StockDataFrame
import random
import copy
from datetime import datetime
import asyncio



def _get_cci(df, stock, cci_w):
    cci_key = 'cci_' + str(cci_w)
    stock[cci_key]
    df['cci'] = df[cci_key]
    del df[cci_key]


def wave_guess(arr, wn_rate):
    wn = int(len(arr) * wn_rate)
    # 计算最大的N个值，认为是波峰
    wave_crest = heapq.nlargest(wn, enumerate(arr), key=lambda x: x[1])

    # 计算最小的N个值，认为是波谷
    wave_base = heapq.nsmallest(wn, enumerate(arr), key=lambda x: x[1])

    wave_crest_x = []  # 波峰x
    wave_crest_y = []  # 波峰y
    for i, j in wave_crest:
        wave_crest_x.append(i)
        wave_crest_y.append(j)

    wave_base_x = []  # 波谷x
    wave_base_y = []  # 波谷y
    for i, j in wave_base:
        wave_base_x.append(i)
        wave_base_y.append(j)

    return wave_crest_x, wave_base_x

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
    wn_rate = init_population(pop_size, chromosome_length)
    cci_w = init_population(pop_size, chromosome_length)
    dir_w = init_population(pop_size, chromosome_length)
    vola_w = init_population(pop_size, chromosome_length)
    ama_std_w = init_population(pop_size, chromosome_length)
    percentage = init_population(pop_size, chromosome_length)
    atr_len = init_population(pop_size, chromosome_length)
    stop_atr = init_population(pop_size, chromosome_length)
    fastest = init_population(pop_size, chromosome_length)
    slowest = init_population(pop_size, chromosome_length)
    return [wn_rate, cci_w, dir_w, vola_w, ama_std_w, percentage, atr_len, stop_atr, fastest, slowest]

# 淘汰
def clear_fit_values(obj_values):
    fit_value = []
    # 去掉小于0的值，更改c_min会改变淘汰的下限
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
    pops = newpop[:][:]

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
def find_best(pops, fit_values, chromosome_length=6):
    # 绩效越小越好
    best_fit = min(fit_values)
    best_fit_idx = fit_values.index(best_fit)
    _wn_rate = binary2decimal(pops[0][best_fit_idx], 0.03, 0.09, chromosome_length)
    _cci_w = int(binary2decimal(pops[1][best_fit_idx], 17, 27, chromosome_length))
    _dir_w = int(binary2decimal(pops[2][best_fit_idx], 10, 30, chromosome_length))
    _vola_w = int(binary2decimal(pops[3][best_fit_idx], 10, 30, chromosome_length))
    _ama_std_w = int(binary2decimal(pops[4][best_fit_idx], 10, 30, chromosome_length))
    _percentage = binary2decimal(pops[5][best_fit_idx], 0.05, 0.2, chromosome_length)
    _atr_len = int(binary2decimal(pops[6][best_fit_idx], 10, 30, chromosome_length))
    _stop_atr = binary2decimal(pops[7][best_fit_idx], 1, 2, chromosome_length)
    _fastest = binary2decimal(pops[8][best_fit_idx], 0.3, 1, chromosome_length)
    _slowest = binary2decimal(pops[9][best_fit_idx], 0.03, 0.1, chromosome_length)

    # 用来存最优基因编码
    best_individual = [_wn_rate, _cci_w, _dir_w, _vola_w, _ama_std_w, _percentage, _atr_len, _stop_atr, _fastest, _slowest]
    return best_individual, best_fit


def cal_fitness(df, pops, pop_size=50, chromosome_length=6):
    obj_value = []
    async_funcs = []
    for i in range(pop_size):
        df_ = copy.deepcopy(df)
        stock = StockDataFrame.retype(df_)
        df_['signal'] = 'wait'
        _wn_rate = binary2decimal(pops[0][i], 0.03, 0.09, chromosome_length)
        _cci_w = int(binary2decimal(pops[1][i], 17, 27, chromosome_length))
        _dir_w = int(binary2decimal(pops[2][i], 10, 30, chromosome_length))
        _vola_w = int(binary2decimal(pops[3][i], 10, 30, chromosome_length))
        _ama_std_w = int(binary2decimal(pops[4][i], 10, 30, chromosome_length))
        _percentage = binary2decimal(pops[5][i], 0.05, 0.2, chromosome_length)
        _atr_len = int(binary2decimal(pops[6][i], 10, 30, chromosome_length))
        _stop_atr = binary2decimal(pops[7][i], 1, 2, chromosome_length)
        _fastest = binary2decimal(pops[8][i], 0.3, 1, chromosome_length)
        _slowest = binary2decimal(pops[9][i], 0.03, 0.1, chromosome_length)
        async_funcs.append(analyze(df_, stock, i, _wn_rate, _cci_w, _dir_w, _vola_w, _ama_std_w, _percentage, _atr_len, _stop_atr, _fastest, _slowest))
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

def analyze(df, stock, index, _wn_rate, _cci_w, _dir_w, _vola_w, _ama_std_w, _percentage, _atr_len, _stop_atr, _fastest, _slowest):
    data_len = len(df)
    _get_cci(df, stock, _cci_w)

    # 参数计算
    # 1. 价格方向
    df['direction'] = df['close'] - df['close'].shift(_dir_w)
    # 2. 波动性
    df['price_diff'] = abs(df['close'] - df['close'].shift(1))
    df['volatility'] = df['price_diff'].rolling(center=False, window=_vola_w).sum()
    # 3. 效率系数 ER
    df['ER'] = df['direction'] / df['volatility']
    # 4. 变换上述系数为趋势速度
    df['smooth'] = df['ER'] * (_fastest - _slowest) + _slowest
    df['smooth'] = df['smooth'] ** 2
    # 删除包含NaN值的任何行 axis：0行，1列;inplace修改原始df
    df.dropna(axis=0, inplace=True)
    ama = 0
    arr = [0]
    for i in range(1, len(df)):
        row = df.iloc[i]
        ama = ama + row['smooth'] * (row['close'] - ama)
        arr.append(ama)
    df['ama'] = np.array(arr)
    df['ama_diff'] = df['ama'] - df['ama'].shift(1)
    df['ama_std'] = df['ama_diff'].rolling(window=_ama_std_w, center=False).std()
    df['filter'] = df['ama_std'] * _percentage
    # ATR计算
    df['tr'] = df['high'] - df['low']
    df['tr'] = np.where(df['tr'] >= abs(df['high'] - df['close'].shift(1)), df['tr'],
                        abs(df['high'] - df['close'].shift(1)))
    df['tr'] = np.where(df['tr'] >= abs(df['low'] - df['close'].shift(1)), df['tr'],
                        abs(df['low'] - df['close'].shift(1)))
    df['atr'] = df['tr'].rolling(window=_atr_len, center=False).mean()


    df['wave'] = 0
    data_len_ = len(df)
    wave_crest_x, wave_base_x = wave_guess(df['close'].values, _wn_rate)
    wave_base_x_ = []
    wave_crest_x_ = []
    for x in wave_crest_x:
        idx = data_len - data_len_ + x
        wave_crest_x_.append(idx)
    for x in wave_base_x:
        idx = data_len - data_len_ + x
        wave_base_x_.append(idx)
    df.loc[wave_crest_x_, 'wave'] = 1
    df.loc[wave_base_x_, 'wave'] = -1

    # 策略1 开多
    df['signal'] = 'wait'
    df['signal'] = np.where((df['signal'] == 'wait') & (
            (df['ama'] - df['ama'].shift(1) > df['filter']) | (df['ama'] - df['ama'].shift(2) > df['filter'])),
                            'long', df['signal'])
    df['signal'] = np.where((df['signal'].shift(1) == 'long') & (df['signal'] == 'wait'), 'close_long', df['signal'])
    df['signal'] = np.where((df['signal'] == df['signal'].shift(1)), 'wait', df['signal'])

    # 策略2 开空
    df['signal'] = np.where((df['signal'] == 'wait') & ((df['ama'] - df['ama'].shift(1) < -1 * df['filter']) | (
            df['ama'] - df['ama'].shift(2) < -1 * df['filter'])), 'short', df['signal'])
    df['signal'] = np.where((df['signal'].shift(1) == 'short') & (df['signal'] == 'wait'), 'close_short', df['signal'])
    df['signal'] = np.where((df['signal'] == df['signal'].shift(1)), 'wait', df['signal'])

    # 策略3开多
    df['signal'] = np.where(
        (df['signal'] == 'wait') & (df['wave'] != 1) & (df['wave'].shift(1) != 1) & (df['wave'].shift(2) != 1) & (
                (df['ama'] - df['ama'].shift(1) > df['filter']) | (df['ama'] - df['ama'].shift(2) > df['filter'])),
        'long', df['signal'])
    df['signal'] = np.where((df['cci'].shift(1) > 120) & (df['cci'] < 120),
                            'close_long', df['signal'])
    df['signal'] = np.where((df['signal'] == df['signal'].shift(1)), 'wait', df['signal'])

    # 策略4开空
    df['signal'] = np.where(
        (df['signal'] == 'wait') & (df['wave'] != -1) & (df['wave'].shift(1) != -1) & (df['wave'].shift(2) != -1) & (
                (df['ama'] - df['ama'].shift(1) < -1 * df['filter']) | (
                df['ama'] - df['ama'].shift(2) < -1 * df['filter'])), 'short', df['signal'])
    df['signal'] = np.where((df['cci'].shift(1) < -120) & (df['cci'] > -120),
                            'close_short', df['signal'])
    df['signal'] = np.where((df['signal'] == df['signal'].shift(1)), 'wait', df['signal'])

    # CCI 信号
    df['signal'] = np.where((df['signal'] == 'wait') & (df['cci'].shift(1) < 100) & (df['cci'] > 100), 'long',
                            df['signal'])

    df['signal'] = np.where((df['signal'] == 'wait') & (df['cci'].shift(1) > -100) & (df['cci'] < -100), 'short',
                            df['signal'])

    # wave 平仓信号
    df['signal'] = np.where((df['wave'].shift(2) == 1) & (df['wave'].shift(1) == 1) & (df['wave'] == 0),
                            'close_long', df['signal'])
    df['signal'] = np.where((df['wave'].shift(2) == -1) & (df['wave'].shift(1) == -1) & (df['wave'] == 0),
                            'close_short', df['signal'])

    df['long_stop_price'] = df['close'] - df['atr'] * _stop_atr
    df['short_stop_price'] = df['close'] + df['atr'] * _stop_atr

    signal_df = df.loc[(df['signal'] != 'wait') & (df['signal'] != 'trend')]
    if len(signal_df) == 0:
        return 0.0, 0.0
    if (signal_df[:1]['signal'] == 'close_long').bool() or (signal_df[:1]['signal'] == 'close_short').bool():
        signal_df = signal_df[1:]
    if (signal_df[-1:]['signal'] == 'long').bool() or (signal_df[-1:]['signal'] == 'short').bool():
        signal_df = signal_df[:len(signal_df) - 1]
    # 初始化
    df = df[-270:]
    market_rate = 0.00075  # 市价单
    market_yield_ = 0.0  # 记录连续亏损收益
    total_yield = 0.0
    max_drawdown = 0.0  # 最大回撤
    i = 1
    while i < len(signal_df):
        md = 0
        market_yield = 0.0
        row_ = signal_df.iloc[i - 1]
        row = signal_df.iloc[i]
        if row['signal'] == 'close_long' and row_['signal'] == 'long':
            market_yield = (row['close'] - row_['close']) / row['close'] - market_rate * 2
            part_df = df.loc[(df['timestamp'] > row_['timestamp']) & (df['timestamp'] <= row['timestamp'])]
            if len(part_df) == 0:
                continue
            min_price = min(part_df.low)
            md = (row_['close'] - min_price) / row_['close']
        elif row['signal'] == 'close_short' and row_['signal'] == 'short':
            market_yield = (row_['close'] - row['close']) / row['close'] - market_rate * 2
            part_df = df.loc[(df['timestamp'] > row_['timestamp']) & (df['timestamp'] <= row['timestamp'])]
            if len(part_df) == 0:
                continue
            max_price = min(part_df.high)
            md = (max_price - row_['close']) / row_['close']
        elif row['signal'] == 'short' and row_['signal'] == 'long':
            market_yield = (row['close'] - row_['close']) / row['close'] - market_rate * 2
            part_df = df.loc[(df['timestamp'] > row_['timestamp']) & (df['timestamp'] <= row['timestamp'])]
            if len(part_df) == 0:
                continue
            min_price = min(part_df.low)
            md = (row_['close'] - min_price) / row_['close']
            i -= 1
        elif row['signal'] == 'long' and row_['signal'] == 'short':
            market_yield = (row_['close'] - row['close']) / row['close'] - market_rate * 2
            part_df = df.loc[(df['timestamp'] > row_['timestamp']) & (df['timestamp'] <= row['timestamp'])]
            if len(part_df) == 0:
                continue
            max_price = min(part_df.high)
            md = (max_price - row_['close']) / row_['close']
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