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


# 检查协整状态
def cointegration(data1, data2):
    # 分割样例测试数据 70%/30%
    df1, test1, df2, test2 = train_test_split(data1, data2, test_size=0.7
                                              , shuffle=False)

    train = pd.DataFrame()
    train['asset1'] = df1['Close']
    train['asset2'] = df2['Close']

    # this is the part where we test the cointegration
    # in this case, i use Engle-Granger two-step method
    # which is invented by the mentor of my mentor!!!
    # generally people use Johanssen test to check the cointegration status
    # the first step for EG is to run a linear regression on both variables
    # next, we do OLS and obtain the residual
    # after that we run unit root test to check the existence of cointegration
    # if it is stationary, we can determine its a drunk man with a dog
    # the first step would be adding a constant vector to asset1

    x = sm.add_constant(train['asset1'])
    y = train['asset2']
    model = sm.OLS(y, x).fit()
    resid = model.resid

    # print(model.summary())
    # print('\n', sm.tsa.stattools.adfuller(resid))

    # this phrase is how we set the trigger conditions
    # first we normalize the residual
    # we would get a vector that follows standard normal distribution
    # generally speaking, most tests use one sigma level as the threshold
    # two sigma level reaches 95% which is relatively difficult to trigger
    # after normalization, we should obtain a white noise follows N(0,1)
    # we set +-1 as the threshold
    # eventually we visualize the result

    signals = pd.DataFrame()
    signals['asset1'] = test1['Close']
    signals['asset2'] = test2['Close']

    signals['fitted'] = np.mat(sm.add_constant(signals['asset2'])) * np.mat(model.params).reshape(2, 1)

    signals['residual'] = signals['asset1'] - signals['fitted']

    signals['z'] = (signals['residual'] - np.mean(signals['residual'])) / np.std(signals['residual'])

    # use z*0 to get panda series instead of an integer result
    signals['z upper limit'] = signals['z'] * 0 + np.mean(signals['z']) + np.std(signals['z'])
    signals['z lower limit'] = signals['z'] * 0 + np.mean(signals['z']) - np.std(signals['z'])

    return signals


# In[2]:


# the signal generation process is very straight forward
# if the normalized residual gets above or below threshold
# we long the bearish one and short the bullish one, vice versa
# i only need to generate trading signal of one asset
# the other one should be the opposite direction
def signal_generation(df1, df2, method):
    signals = method(df1, df2)

    signals['signals1'] = 0

    # as z statistics cannot exceed both upper and lower bounds at the same time
    # this line holds
    signals['signals1'] = np.select([signals['z'] > signals['z upper limit'], \
                                     signals['z'] < signals['z lower limit']], \
                                    [-1, 1], default=0)


    # signals only imply holding
    # we take the first order difference to obtain the execution signal
    signals['positions1'] = signals['signals1'].diff()
    signals['signals2'] = -signals['signals1']
    signals['positions2'] = signals['signals2'].diff()
    print(signals[['z', 'signals1', 'positions1', 'signals2', 'positions2']].tail(10))

    # fix initial positions issue
    if signals['signals1'].iloc[0] != 0:
        signals['positions1'].iloc[0] = signals['signals1'].iloc[0]
        signals['positions2'].iloc[0] = signals['signals2'].iloc[0]

    return signals


filename = 'poloniex-USDT_%s-2019-01-01-2019-12-12-30m'
ticker1 = 'EOS'
ticker2 = 'ETH'
df1 = data2df.csv2df(filename % ticker1 + '.csv')
df1 = df1.astype(float)
df2 = data2df.csv2df(filename % ticker2 + '.csv')
df2 = df2.astype(float)
df1 = df1[-500:]
df2 = df2[-500:]
