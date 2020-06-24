# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2020/6/22 21:26

@desc:

'''

import pandas as pd
import numpy as np
from tools import data2df, ama_ga_tools, public_tools
from stockstats import StockDataFrame
from tools.BmBackTest import BmBackTest
import copy

# @public_tools.fn_timer
def get_params(kline):
    pop_size, chromosome_length = 30, 6
    pops = ama_ga_tools.init_pops(pop_size, chromosome_length)
    iter = 10
    pc = 0.6  # 杂交概率
    pm = 0.01  # 变异概率
    results = []  # 存储每一代的最优解，N个二元组
    best_individuals = []
    best_fits = []
    for j in range(iter):
        obj_values = ama_ga_tools.cal_fitness(kline, pops, pop_size, chromosome_length)  # 计算绩效
        fit_values = ama_ga_tools.clear_fit_values(obj_values)
        best_individual, best_fit = ama_ga_tools.find_best(pops, fit_values,
                                                                 chromosome_length)  # 第一个是最优基因序列, 第二个是对应的最佳个体适度
        best_individuals.append(best_individual)
        best_fits.append(best_fit)
        results.append([best_individual, best_fit])
        ama_ga_tools.selection(pops, fit_values)  # 选择
        ama_ga_tools.crossover(pops, pc)  # 染色体交叉（最优个体之间进行0、1互换）
        ama_ga_tools.mutation(pops, pm)  # 染色体变异（其实就是随机进行0、1取反
    best_fit = min(best_fits)
    best_fit_idx = best_fits.index(best_fit)
    return best_individuals[best_fit_idx]

def get_signal(kline, params):
    df = pd.DataFrame(kline, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data_len = len(df)
    stock = StockDataFrame.retype(df)
    _wn_rate, _cci_w, _dir_w, _vola_w, _ama_std_w, _percentage, _atr_len, _stop_atr, _fastest, _slowest = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9]
    ama_ga_tools._get_cci(df, stock, _cci_w)

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
    wave_crest_x, wave_base_x = ama_ga_tools.wave_guess(df['close'].values, _wn_rate)
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
    return df.iloc[-1]



def analysis(kline):
    best_individual = get_params(kline)
    last_row = get_signal(kline, best_individual)
    return last_row, best_individual

# filename = 'BitMEX-ETH-180803-190817-4H'
filename='BitMEX-170901-191107-4H'
# filename='BitMEX-170901-190606-4H'
df = data2df.csv2df(filename + '.csv')
df = df.astype(float)

datas = df.values

backtest = BmBackTest({
    'asset': 1
})

level = 1

for i in range(370, len(df)):
    test_df = datas[i - 370: i]
    row, best_individual = analysis(test_df)

    if row['signal'] in ['long', 'short']:
        amount = int(backtest.asset * row['close'] * level)
        backtest.create_order(row['signal'], "market", row['close'], amount)
        # backtest.stop_price = row['%s_stop_price' % row['signal']]

    elif row['signal'] == 'trend' or (row['signal'] == 'close_long' and backtest.side == 'long') or (
            row['signal'] == 'close_short' and backtest.side == 'short'):
        backtest.close_positions(row['close'], 'market')

    else:
        backtest.add_data(row['close'], row['high'], row['low'])

    print(i, row['close'], row['signal'], backtest.open_price, backtest.side, backtest.asset, backtest.float_profit,
          backtest.asset + backtest.float_profit, best_individual)

backtest.show("AMA GA")

