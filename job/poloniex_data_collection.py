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

if __name__ == '__main__':
    ts = {
        '2015': 1420041600,
        '2016': 1451577600,
        '2017': 1483200000,
        '2018': 1514736000,
    }
    # 一年时间戳
    t = 31536000
    period = 1800  # 300, 900, 1800, 7200, 14400, and 86400

    # 获取2016年到现在4小时BTCK线数据
    try:
        url = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=%d&end=%d&period=%d' % (
            1504195200, int(time.time()), period)
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
        fileName = '../data/BTC%s-%s.csv' % ("2017.9.1", "now-30M")
        df.to_csv(fileName, index=None)
    except BaseException as e:
        print(e)
