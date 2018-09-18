#！/usr/bin/env python
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
    t = 31536000
    period = 86400  # 300, 900, 1800, 7200, 14400, and 86400

    # 获取2016年到现在4小时BTCK线数据
    try:
        url = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=%d&end=%d&period=%d' % (
            ts['2016'], int(time.time()), period)
        print('URL: %s' % url)
        headers = {
            'user-agent': 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)',
            'content-type': 'application/json',
            'cookie': '__cfduid=d627bf9882e83f26102e92c2a464e11d71529976213; _ga=GA1.2.937486957.1529980048; mp_fb00f1e678521d803202045e854f467e_mixpanel=%7B%22distinct_id%22%3A%20%221657ef8c0c7748-0b1341e6c18b12-2711639-144000-1657ef8c0c89f0%22%2C%22%24initial_referrer%22%3A%20%22%24direct%22%2C%22%24initial_referring_domain%22%3A%20%22%24direct%22%7D'
        }

        req = request.Request(url=url, headers=headers, method='GET')
        openUrl = request.urlopen(req)

        r = openUrl.read()
        d = json.loads(r.decode())
        df = pd.DataFrame(d)

        original_columns = [u'date', u'high', u'low', u'open', u'close', u'volume', u'weightedAverage']
        new_columns = ['Date', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
        df = df.loc[:, original_columns]
        df.columns = new_columns
        fileName = '../data/BTC%s-%s.csv' % ("2016", "now-1D")
        df.to_csv(fileName, index=None)
    except BaseException as e:
        print(e)
