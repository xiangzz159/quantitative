# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/6/26 9:38

@desc: poloniex 数据抓取

'''

import json
import numpy as np
import os
import pandas as pd
from urllib import request
import time

'''

@begin: 开始时间

@end: 结束时间

@period: K线周期

@return: DataFrame数据

'''
def get_poloniex_kline(symbol, begin, end, period):
    try:
        url = 'https://poloniex.com/public?command=returnChartData&currencyPair=%s&start=%d&end=%d&period=%d' % (
            symbol, begin, end, period)
        print('URL: %s' % url)
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
        }

        req = request.Request(url=url, headers=headers, method='GET')
        openUrl = request.urlopen(req)

        r = openUrl.read()
        d = json.loads(r.decode())
        df = pd.DataFrame(d)

        original_columns = [u'date', u'high', u'low', u'open', u'close', u'volume', u'weightedAverage']
        new_columns = ['Timestamp', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
        df = df.loc[:, original_columns]
        df.columns = new_columns
        return df
    except BaseException as e:
        print(e)

if __name__ == '__main__':
    ts = {
        '2015': 1420041600,
        '2016': 1451577600,
        '2017': 1483200000,
        '2018': 1514736000,
    }
    # 一年时间戳
    t = 31536000
    period = 14400  # 300, 900, 1800, 7200, 14400, and 86400

    end = int(time.time())
    begin = 1514736000

    date = time.strftime("%Y-%m-%d", time.localtime(begin))
    df = get_poloniex_kline('USDT_BTC', begin, end, period)
    fileName = '../data/BTC%s-%s.csv' % (date, "now-4H")
    df.to_csv(fileName, index=None)
