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
    timeframe = {
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '2H': 7200,
        '4H': 14400,
        '1d': 86400
    }
    period = '30m'
    begin = 1546272000
    end = 1576080000
    symbol = 'USDT_BTC'
    date1 = time.strftime("%Y-%m-%d", time.localtime(begin))
    date2 = time.strftime("%Y-%m-%d", time.localtime(end))
    df = get_poloniex_kline(symbol, begin, end, timeframe[period])

    fileName = '../data/poloniex-%s-%s-%s-%s.csv' % (symbol, date1, date2, period)
    df.to_csv(fileName, index=None)
