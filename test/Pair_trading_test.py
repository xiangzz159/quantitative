# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2019/12/27 17:27

@desc:

'''

import numpy as np
from tools import data2df
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from tools.BmBackTest import BmBackTest


# 检查协整状态
def analysis(data1, data2):
    test_size = 0.7
    limit_rate = 0.8
    # 分割样例测试数据 70%/30%
    df1, test1, df2, test2 = train_test_split(data1, data2, test_size=test_size
                                              , shuffle=False)

    train = pd.DataFrame()
    train['asset1'] = df1['Close']
    train['asset2'] = df2['Close']

    x = sm.add_constant(train['asset1'])
    y = train['asset2']
    model = sm.OLS(y, x).fit()
    resid = model.resid
    result = sm.tsa.stattools.adfuller(resid)
    adf_test_result = result[0]
    pvalue = result[1]
    d = result[4]
    p1 = d['1%']
    p5 = d['5%']
    p10 = d['10%']

    df = pd.DataFrame()
    df['asset1'] = test1['Close']
    df['asset2'] = test2['Close']
    df['s'] = 0

    if adf_test_result > p1 or adf_test_result > p5 or adf_test_result > p10 or pvalue > 0.01:
        return df

    df['fitted'] = np.mat(sm.add_constant(df['asset2'])) * np.mat(model.params).reshape(2, 1)

    df['residual'] = df['asset1'] - df['fitted']

    df['z'] = (df['residual'] - np.mean(df['residual'])) / np.std(df['residual'])

    # use z*0 to get panda series instead of an integer result
    df['z upper limit'] = df['z'] * limit_rate + np.mean(df['z']) + np.std(df['z'])
    df['z lower limit'] = df['z'] * limit_rate + np.mean(df['z']) - np.std(df['z'])


    df['s'] = np.select([df['z'] > df['z upper limit'], \
                         df['z'] < df['z lower limit']], \
                        [1, -1], default=0)
    df['p'] = df['s'].diff()
    df['s'] = df['p']
    df['s'] = np.select([df['s'] > 0, df['s'] < 0], [1, -1], default=0)

    # df['signal'] = 'wait'
    # df['signal'] = np.where((df['s'] == 1) & (df['s'].shift(1) == 0), 'long', df['signal'])
    # df['signal'] = np.where((df['s'] == -1) & (df['s'].shift(1) == 0), 'short', df['signal'])
    # df['signal'] = np.where((df['s'] == 1) & (df['s'].shift(1) == -1), 'long', df['signal'])
    # df['signal'] = np.where((df['s'] == -1) & (df['s'].shift(1) == 1), 'short', df['signal'])
    # df['signal'] = np.where((df['s'] == 0) & (df['s'].shift(1) == -1), 'close_short', df['signal'])
    # df['signal'] = np.where((df['s'] == 0) & (df['s'].shift(1) == 1), 'close_long', df['signal'])

    return df


filename = 'BitMEX-%s-20180803-20190920-15m'
ticker1 = 'BTC'
ticker2 = 'ETH'
df1 = data2df.csv2df(filename % ticker1 + '.csv')
df1 = df1.astype(float)
df2 = data2df.csv2df(filename % ticker2 + '.csv')
df2 = df2.astype(float)
backtest1 = BmBackTest({
    'asset': 0.5
})
backtest2 = BmBackTest({
    'asset': 0.5
})

level = 1

start_idx = 500
plt_data = []


signals = {
    -1: 'short',
    0: 'wait',
    1: 'long'
}
for i in range(start_idx, len(df1)):
    test_df1 = df1[i - start_idx: i]
    test_df2 = df2[i - start_idx: i]
    re = analysis(test_df1, test_df2)
    # positions1==1 Long ticker1 Short ticker2
    # positions1==-1 Short ticker1 Long ticker2
    row = re.iloc[-1]
    s = int(row['s'])

    if s != 0:
        if (s == 1 and backtest1.side == 'short') or (s == -1 and backtest1.side == 'long'):
            # close position
            backtest1.close_positions(row['asset1'], 'market')
            backtest2.close_positions(row['asset2'], 'market')
        else:
            if backtest1.side == 'wait':
                min_asset = backtest1.asset + backtest2.asset
                amount1 = int(min_asset * row['asset1'] * level)
                amount2 = int(min_asset * row['asset2'] * level)
                backtest1.create_order(signals[s], 'market', row['asset1'], amount1)
                backtest2.create_order(signals[-1 * s], 'market', row['asset2'], amount2)
    else:
        backtest1.add_data(row['asset1'])
        backtest2.add_data(row['asset2'])

    print(i, round(row['asset1'], 4), round(row['asset2'], 4), signals[s],
          round(backtest1.asset + backtest1.float_profit, 4),
          round(backtest2.asset + backtest2.float_profit, 4),
          round(backtest1.asset + backtest1.float_profit + backtest2.asset + backtest2.float_profit, 4))
