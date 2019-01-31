#！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2019/1/31 9:57

@desc:

'''

from WindPy import *
from WindCharts import *
from pandas import DataFrame
import pandas as pd
import numpy as np

w.start(show_welcome=False)


def parse_data(data):
    df = DataFrame(data.Data)
    df = df.T
    df.columns = data.Fields
    df.index = [temp.strftime('%Y-%m-%d') for temp in data.Times]
    df['middle'] = (df['CLOSE'] + df['HIGH'] + df['LOW']) / 3.0
    df['middle_sma'] = df['middle'].rolling(min_periods=1,window=20,center=False).mean()
    md = df['middle'].rolling(
                min_periods=1, center=False, window=20).apply(
                lambda x: np.fabs(x - x.mean()).mean())
    df['cci'] = (df['middle'] - df['middle_sma']) / (.015 * md)
    d = df['CLOSE'] - df['CLOSE'].shift(1)
    df['closepm'] = (d + d.abs()) / 2
    df['closenm'] = (-d + d.abs()) / 2
    df['closepm_'] = df['closepm'].ewm(ignore_na=False, alpha=1.0 / 14, min_periods=0, adjust=True).mean()
    df['closenm_'] = df['closenm'].ewm(ignore_na=False, alpha=1.0 / 14, min_periods=0, adjust=True).mean()
    df['rs'] = rs = df['closepm_'] / df['closenm_']
    df['rsi'] = 100 - 100 / (1.0 + rs)
    df['stoch_rsi'] = df['rsi']
    lowestlow = pd.Series.rolling(df.stoch_rsi, window=14, center=False).min()
    highesthigh = pd.Series.rolling(df.stoch_rsi, window=14, center=False).max()
    K = pd.Series.rolling(100 * ((df.stoch_rsi - lowestlow) / (highesthigh - lowestlow)), window=3).mean()
    D = pd.Series.rolling(K, window=3).mean()
    df['stoch_k'] = K
    df['stoch_d'] = D
    df['dk'] = df['stoch_d'] - df['stoch_k']
    dk = 1
    df['signal'] = np.where((df['cci'] >= 105) & (df['cci'].shift(1) < 100), 'long',
                            'wait')
    df['signal'] = np.where(
        (df['signal'] == 'wait') & (((df['cci'] <= 100) & (df['cci'].shift(1) > 100)) |
                                    ((df['cci'] >= 95) & (df['dk'] > dk)) | (
                                                (df['cci'] <= 95) & (df['cci'].shift(1) > 95))),
        'close_long', df['signal'])
    del df['closepm']
    del df['closepm_']
    del df['closenm']
    del df['closenm_']
    del df['rs']
    del df['middle']
    del df['middle_sma']
    del df['rsi']
    return df

def prepare_data(context):
    # 利用WIND API获取复权后的均线数据
    context.signal = {}
    for code in context.securities:
        data = w.wsd(code, 'close,high,low', context.start_date, context.end_date)
        df = parse_data(data)
        context.signal[code] = df['signal']



def initialize(context):
    context.capital = 1000000
    context.securities = ['000001.SH']
    context.start_date = '20140101'
    context.end_date = '20180101'
    context.period = 'd'
    prepare_data(context)


def handle_data(bar_datetime, context, bar_data):
    bar_datetime_str = bar_datetime.strftime('%Y-%m-%d')
    position = bkt.query_position()
    data = w.wsd('000001.SH', 'close,high,low', context.start_date, context.end_date)
    df = parse_data(data)

    if ('000001.SH' in position.get_field('code')):
        if context.signal['000001.SH'][bar_datetime_str] == 'close_long':
            bkt.batch_order.sell_all()
    else:
        if ('000001.SH' not in position.get_field('code')):
            if context.signal['000001.SH'][bar_datetime_str] == 'long':
                res = bkt.order_percent('000001.SH', 0.9, 'buy')


bkt = BackTest(init_func=initialize, handle_data_func=handle_data)
res = bkt.run(show_progress=True)
nav_df = bkt.summary('nav')
df = parse_data(data)



