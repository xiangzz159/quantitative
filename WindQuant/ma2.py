#！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2019/1/30 10:07

@desc:

'''

from WindAlgo import *
from datetime import *
from pandas import DataFrame
from WindPy import *

w.start(show_welcome=False)


def prepare_data(context):
    # 利用WIND API获取复权后的均线数据
    context.ma5 = {}
    context.ma15 = {}
    for code in context.securities:
        # 获取5日前复权均线
        wsd_data = w.wsd(code, "MA", context.start_date, context.end_date, "MA_N=5;PriceAdj=F")
        wsd_df = DataFrame(wsd_data.Data)
        wsd_df = wsd_df.T
        wsd_df.columns = wsd_data.Fields
        wsd_df.index = [temp.strftime('%Y-%m-%d') for temp in wsd_data.Times]
        context.ma5[code] = wsd_df

        # 获取15日前复权均线
        wsd_data = w.wsd(code, "MA", context.start_date, context.end_date, "MA_N=15;PriceAdj=F")
        wsd_df = DataFrame(wsd_data.Data)
        wsd_df = wsd_df.T
        wsd_df.columns = wsd_data.Fields
        wsd_df.index = [temp.strftime('%Y-%m-%d') for temp in wsd_data.Times]
        context.ma15[code] = wsd_df


# 定义初始化函数
def initialize(context):
    context.capital = 1000000
    context.securities = ["000333.SZ"]
    context.start_date = "20160201"
    context.end_date = "20170501"
    context.period = 'd'
    prepare_data(context)


# 定义策略函数
def handle_data(bar_datetime, context, bar_data):
    bar_datetime_str = bar_datetime.strftime('%Y-%m-%d')
    position = bkt.query_position()
    if ('000333.SZ' in position.get_field('code')):
        if (context.ma5['000333.SZ']['MA'][bar_datetime_str] < context.ma15['000333.SZ']['MA'][bar_datetime_str]):
            bkt.batch_order.sell_all()
    else:
        if ('000333.SZ' not in position.get_field('code')):
            if (context.ma5['000333.SZ']['MA'][bar_datetime_str] > context.ma15['000333.SZ']['MA'][bar_datetime_str]):
                res = bkt.order_percent('000333.SZ', 0.9, 'buy')


bkt = BackTest(init_func=initialize, handle_data_func=handle_data)
res = bkt.run(show_progress=True)
nav_df = bkt.summary('nav')