import numpy as np
from tools.stockstats import StockDataFrame
import random
import copy
import pandas as pd
from datetime import datetime
import asyncio

"""
df: 样本
fl: fast_length
sl: slow_length
"""


def _get_macd(df, stock, fl, sl):
    fast_key = 'close_%d_ema' % fl
    slow_key = 'close_%d_ema' % sl
    stock[fast_key]
    stock[slow_key]

    df['macd'] = df[fast_key] - df[slow_key]
    stock['macd_9_ema']
    # DEA
    df['macds'] = df['macd_9_ema']
    df['hist'] = df['macd'] - df['macds']
    # MACD
    df['macdh'] = 2 * df['hist']

    del df[fast_key]
    del df[slow_key]
    del df['hist']
    del df['macd_9_ema']


"""
df: 样本
A: BOLL 波动率
l: STD 样本长度
"""


def _get_boll(df, stock, A, l):
    avg_key = 'close_%d_sma' % l
    std_key = 'close_%d_mstd' % l
    stock[avg_key]
    stock[std_key]

    moving_avg = df[avg_key]
    moving_std = df[std_key]
    df['boll'] = moving_avg
    moving_avg = list(map(np.float64, moving_avg))
    moving_std = list(map(np.float64, moving_std))
    # noinspection PyTypeChecker
    df['boll_ub'] = np.add(moving_avg,
                           np.multiply(A, moving_std))
    # noinspection PyTypeChecker
    df['boll_lb'] = np.subtract(moving_avg,
                                np.multiply(A,
                                            moving_std))
    del df[avg_key]
    del df[std_key]


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
    boll_A = init_population(pop_size, chromosome_length)  # BOLL 波动率倍数 1.5~3
    boll_std_len = init_population(pop_size, chromosome_length)  # BOLL_STD 样本长度 14~30
    macd_fast_len = init_population(pop_size, chromosome_length)  # MACD 快线长度 8~16
    macd_slow_len = init_population(pop_size, chromosome_length)  # MACD 慢线长度 20~30
    boll_width_threshold = init_population(pop_size, chromosome_length)  # BOLL 通道阈值 0.05~0.15
    volatility = init_population(pop_size, chromosome_length)  # 波动率下单 0.0008~0.0012
    stop_limit = init_population(pop_size, chromosome_length)  # 止盈止损阈值 0.01~0.1
    # 趋势判断阈值
    trend_A1 = init_population(pop_size, chromosome_length)  # 0~0.3
    trend_A2 = init_population(pop_size, chromosome_length)  # 0~0.02
    trend_B = init_population(pop_size, chromosome_length)  # 0~0.04
    trend_C = init_population(pop_size, chromosome_length)  # 0~0.05
    stop_trade_times = init_population(pop_size, chromosome_length)  # 6~24
    return [
        boll_A,
        boll_std_len,
        macd_fast_len,
        macd_slow_len,
        boll_width_threshold,
        volatility,
        stop_limit,
        trend_A1,
        trend_A2,
        trend_B,
        trend_C,
        stop_trade_times
    ]


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
    boll_A_ = binary2decimal(pop[0][best_fit_idx], 1.5, 3, chromosome_length)
    boll_std_len_ = int(binary2decimal(pop[1][best_fit_idx], 14, 30, chromosome_length))
    macd_fast_len_ = int(binary2decimal(pop[2][best_fit_idx], 8, 16, chromosome_length))
    macd_slow_len_ = int(binary2decimal(pop[3][best_fit_idx], 20, 30, chromosome_length))
    boll_width_threshold_ = binary2decimal(pop[4][best_fit_idx], 0.05, 0.15, chromosome_length)
    volatility_ = binary2decimal(pop[5][best_fit_idx], 0.0005, 0.0015, chromosome_length)
    stop_limit_ = binary2decimal(pop[6][best_fit_idx], 0.01, 0.1, chromosome_length)
    trend_A1_ = binary2decimal(pop[7][best_fit_idx], 0.2, 0.35, chromosome_length)
    trend_A2_ = binary2decimal(pop[8][best_fit_idx], 0.01, 0.025, chromosome_length)
    trend_B_ = binary2decimal(pop[9][best_fit_idx], 0.02, 0.045, chromosome_length)
    trend_C_ = binary2decimal(pop[10][best_fit_idx], 0.03, 0.055, chromosome_length)
    stop_trade_times_ = int(binary2decimal(pop[11][best_fit_idx], 6, 12, chromosome_length))
    # 用来存最优基因编码
    best_individual = [boll_A_, boll_std_len_, macd_fast_len_, macd_slow_len_,
                       boll_width_threshold_, volatility_, stop_limit_, trend_A1_, trend_A2_,
                       trend_B_, trend_C_, stop_trade_times_]
    return best_individual, best_fit


def cal_fitness(df, pops, pop_size=50, chromosome_length=6):
    obj_value = []
    async_funcs = []
    for i in range(pop_size):
        df_ = copy.deepcopy(df)
        stock = StockDataFrame.retype(df_)
        df_['signal'] = 'wait'
        boll_A_ = binary2decimal(pops[0][i], 1.5, 3, chromosome_length)
        boll_std_len_ = int(binary2decimal(pops[1][i], 14, 30, chromosome_length))
        macd_fast_len_ = int(binary2decimal(pops[2][i], 8, 16, chromosome_length))
        macd_slow_len_ = int(binary2decimal(pops[3][i], 20, 30, chromosome_length))
        boll_width_threshold_ = binary2decimal(pops[4][i], 0.05, 0.15, chromosome_length)
        volatility_ = binary2decimal(pops[5][i], 0.0009, 0.0011, chromosome_length)
        stop_limit_ = binary2decimal(pops[6][i], 0.01, 0.1, chromosome_length)
        trend_A1_ = binary2decimal(pops[7][i], 0.2, 0.35, chromosome_length)
        trend_A2_ = binary2decimal(pops[8][i], 0.01, 0.025, chromosome_length)
        trend_B_ = binary2decimal(pops[9][i], 0.02, 0.045, chromosome_length)
        trend_C_ = binary2decimal(pops[10][i], 0.03, 0.055, chromosome_length)
        stop_trade_times_ = int(binary2decimal(pops[11][i], 6, 12, chromosome_length))
        async_funcs.append(cal_someone_fitness(df_, stock, i, boll_A_, boll_std_len_, macd_fast_len_,
                                               macd_slow_len_,
                                               boll_width_threshold_, volatility_, stop_limit_, trend_A1_,
                                               trend_A2_,
                                               trend_B_, trend_C_, stop_trade_times_))
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
                              stock,
                              index,
                              boll_A,
                              boll_std_len,
                              macd_fast_len,
                              macd_slow_len,
                              boll_width_threshold,
                              volatility,
                              stop_limit,
                              trend_A1,
                              trend_A2,
                              trend_B,
                              trend_C,
                              stop_trade_times,
                              ts=3600,
                              data_len=-96):
    # 获得macd，boll指标
    _get_boll(df, stock, boll_A, boll_std_len)
    _get_macd(df, stock, macd_fast_len, macd_slow_len)

    # 通道宽度
    df['boll_width'] = abs(df['boll_ub'] - df['boll_lb'])
    # 单次涨跌幅
    df['change'] = abs((df['close'] - df['close'].shift(1)) / df['close'].shift(1))
    # 止盈止损
    df['channel_stop_limit'] = df['boll_width'] * stop_limit
    # 趋势判断A：超过通道一定幅度且单次涨幅超过一定幅度
    df['signal'] = np.where(((df['close'] > df['boll_ub'] + df['boll_width'] * trend_A1) | (
            df['close'] < df['boll_lb'] - df['boll_width'] * trend_A1)) & (df['change'] > trend_A2), 'trend',
                            df['signal'])
    # 趋势判断B：单次涨幅累计超过一定幅度
    df['signal'] = np.where(df['change'] > trend_B, 'trend', df['signal'])
    # 趋势判断C：两次涨幅累计超过一定幅度
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
    if len(tts) > 0:
        df.loc[tts, 'signal'] = 'can_not_trade'
    df['signal'] = np.where(df['signal'] == 'can_not_trade', 'trend', df['signal'])

    df['channel_limit'] = df['boll_width'] * boll_width_threshold
    df['channel_limit_3'] = df['channel_limit'] * 3

    # 去掉数据错误记录
    df = df[data_len:]

    # 策略1
    df['signal1'] = np.where(
        (df['signal'] == 'wait') & (df['change'] >= volatility) & (df['boll_lb'].shift(1) <= df['close'].shift(1)) & (
                df['close'].shift(1) <= df['boll'].shift(1)) & (
                df['close'] <= df['boll_lb'] + df['channel_limit']) & (
                df['close'] >= df['boll_lb'] - df['channel_limit']), 'long', 'wait')

    df['signal1'] = np.where((df['signal'] == 'trend') | ((df['signal1'] == 'wait') & (
            (df['high'] >= df['boll'] - df['channel_stop_limit']) | (
            df['low'] <= df['boll_lb'] - df['channel_stop_limit']))), 'close_long', df['signal1'])

    # 策略2
    df['signal2'] = np.where(
        (df['signal'] == 'wait') & (df['change'] >= volatility) & ((df['boll'].shift(1) >= df['close'].shift(1)) & (
                df['close'].shift(1) >= df['boll_lb'].shift(1)) & (df['boll'] + df['channel_limit_3'] > df['close'])
                                                                   & (df['close'] >= df['boll'])), 'short', 'wait')

    df['signal2'] = np.where((df['signal'] == 'trend') | ((df['signal2'] == 'wait') & (
            (df['low'] <= df['boll_ub'] - df['channel_stop_limit']) | (
            df['high'] >= df['boll'] + df['channel_stop_limit']))), 'close_short', df['signal2'])

    # 策略3
    df['signal3'] = np.where(
        (df['signal'] == 'wait') & (df['change'] >= volatility) & ((df['macd'].shift(1) <= df['macds'].shift(1)) & (
                df['macd'] > df['macds']) & (df['close'].shift(1) < df['boll'].shift(1))
                                                                   & (df['close'] > df['boll'])), 'long', 'wait')

    df['signal3'] = np.where(
        (df['signal'] == 'trend') | ((df['signal3'] == 'wait') & (
                (df['high'] >= df['boll_ub'] - df['channel_stop_limit']) |
                (df['low'] <= df['boll'] - df['channel_stop_limit']))), 'close_long', df['signal3'])

    # 策略4
    df['signal4'] = np.where(
        (df['signal'] == 'wait') & (df['change'] >= volatility) & ((df['close'].shift(1) > df['boll'].shift(1))
                                                                   & (df['close'] < df['boll']) & (
                                                                           df['boll'] - df['close'] > df[
                                                                       'channel_limit'])),
        'short', 'wait')

    df['signal4'] = np.where((df['signal'] == 'trend') | ((df['signal4'] == 'wait') & (
            (df['low'] <= df['boll_ub'] - df['channel_stop_limit']) | (
            df['high'] >= df['boll'] + df['channel_stop_limit']))), 'close_short', df['signal4'])

    # 策略5
    df['signal5'] = np.where(
        (df['signal'] == 'wait') & (df['change'] >= volatility) & (df['boll_ub'].shift(1) >= df['close'].shift(1)) & (
                df['close'].shift(1) >= df['boll'].shift(1)) & (
                df['close'] >= df['boll_ub']) & (
                df['close'] <= df['boll_ub'] + df['channel_limit']), 'short', 'wait')

    df['signal5'] = np.where((df['signal'] == 'trend') | ((df['signal5'] == 'wait') & (
            (df['low'] <= df['boll'] + df['channel_stop_limit']) | (
            df['high'] >= df['boll_ub'] + df['channel_stop_limit']))), 'close_short', df['signal5'])

    # 策略6
    df['signal6'] = np.where(
        (df['signal'] == 'wait') & (df['change'] >= volatility) & ((df['boll'].shift(1) <= df['close'].shift(1)) & (
                df['close'].shift(1) <= df['boll_ub'].shift(1)) & (df['boll'] + df['channel_limit_3'] > df['close'])
                                                                   & (df['close'] >= df['boll'] - df['channel_limit'])),
        'long', 'wait')

    df['signal6'] = np.where(
        (df['signal'] == 'trend') | ((df['signal6'] == 'wait') & (
                (df['high'] >= df['boll_ub'] - df['channel_stop_limit']) | (
                df['low'] <= df['boll'] - df['channel_stop_limit']))), 'close_long', df['signal6'])

    last_signal = 'wait'
    signal_num = -1
    for i in range(len(df)):
        row = df.iloc[i]
        for j in range(1, 5):
            if row['signal%d' % j] in ['long', 'short']:
                if last_signal != row['signal%d' % j]:
                    df['signal'].iloc[i] = row['signal%d' % j]
                    last_signal = row['signal%d' % j]
                    signal_num = j
                    break
            elif row['signal%d' % j] in ['close_long', 'close_short']:
                if signal_num == j and last_signal in row['signal%d' % j]:
                    df['signal'].iloc[i] = row['signal%d' % j]
                    last_signal = 'wait'
                    signal_num = -1
                    break

    # 删除无用字段
    del df['signal1']
    del df['signal2']
    del df['signal3']
    del df['signal4']
    del df['signal5']
    del df['signal6']
    del df['channel_limit']
    del df['channel_stop_limit']
    del df['change']
    del df['boll_width']

    signal_df = df.loc[(df['signal'] != 'wait') & (df['signal'] != 'trend')]
    if len(signal_df) == 0:
        return 0.0, 0.0
    if (signal_df[:1]['signal'] == 'close_long').bool() or (signal_df[:1]['signal'] == 'close_short').bool():
        signal_df = signal_df[1:]
    if (signal_df[-1:]['signal'] == 'long').bool() or (signal_df[-1:]['signal'] == 'short').bool():
        signal_df = signal_df[:len(signal_df) - 1]

    # 初始化
    limit_rate = 0.00025  # 限价单
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
            market_yield = (row['close'] - row_['close']) / row['close'] - market_rate - limit_rate
            part_df = df.loc[(df['timestamp'] > row_['timestamp']) & (df['timestamp'] <= row['timestamp'])]
            if len(part_df) == 0:
                continue
            min_price = min(part_df.low)
            md = (row_['close'] - min_price) / row_['close']
        elif row['signal'] == 'close_short' and row_['signal'] == 'short':
            market_yield = (row_['close'] - row['close']) / row['close'] - market_rate - limit_rate
            part_df = df.loc[(df['timestamp'] > row_['timestamp']) & (df['timestamp'] <= row['timestamp'])]
            if len(part_df) == 0:
                continue
            max_price = min(part_df.high)
            md = (max_price - row_['close']) / row_['close']
        elif row['signal'] == 'short' and row_['signal'] == 'long':
            market_yield = (row['close'] - row_['close']) / row['close'] - limit_rate * 2
            part_df = df.loc[(df['timestamp'] > row_['timestamp']) & (df['timestamp'] <= row['timestamp'])]
            if len(part_df) == 0:
                continue
            min_price = min(part_df.low)
            md = (row_['close'] - min_price) / row_['close']
            i -= 1
        elif row['signal'] == 'long' and row_['signal'] == 'short':
            market_yield = (row_['close'] - row['close']) / row['close'] - limit_rate * 2
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
