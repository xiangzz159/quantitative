#！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2019/1/30 10:06

@desc:  MA策略

'''

from WindAlgo import *
import talib as ta


def initialize(context):
    context.capital = 1000000
    context.securities = ['000001.SZ', '600519.SH']
    context.start_date = '20150101'
    context.end_date = '20170501'
    context.period = 'd'


def handle_data(bar_datetime, context, bar_data):
    his = bkt.history('600519.SH', 30)
    ma5 = ta.MA(np.array(his.get_field('close')), timeperiod=5, matype=0)
    ma20 = ta.MA(np.array(his.get_field('close')), timeperiod=20, matype=0)
    position = bkt.query_position()
    if ('600519.SH' in position.get_field('code')):
        if (ma5[-1] < ma20[-1]):
            bkt.batch_order.sell_all()
    else:
        if ('600519.SH' not in position.get_field('code')):
            if (ma5[-1] > ma20[-1]):
                res = bkt.order('600519.SH', 4000, 'buy')


bkt = BackTest(init_func=initialize, handle_data_func=handle_data)
res = bkt.run(show_progress=True)
nav_df = bkt.summary('nav')