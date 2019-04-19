# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2019/4/19 10:26

@desc: 

'''

import json
import numpy as np
import os
import pandas as pd
from urllib import request
import time

symbol = 'tBTCUSD'
# Available values: '1m', '5m', '15m', '30m', '1h', '3h', '6h', '12h', '1D', '7D', '14D', '1M'
timeframe = '5m'
# max:5000
limit = 5000
# ms
start = 1514736000000
try:
    url = 'https://api-pub.bitfinex.com/v2/candles/trade:%s:%s/hist?limit=%d&start=%d&sort=1' % (
        timeframe, symbol, limit, start)
    print('URL: %s' % url)
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
    }

    req = request.Request(url=url, headers=headers, method='GET')
    openUrl = request.urlopen(req)

    r = openUrl.read()
    d = json.loads(r.decode())
    df = pd.DataFrame(d, columns=['Timestamp', 'Open', 'Close', 'High', 'Low', 'Volume'])

    df['Timestamp'] = df['Timestamp'] / 1000
    date = time.strftime("%Y-%m-%d", time.localtime(start / 1000))
    fileName = '../data/bitfinex_5M.csv'
    df.to_csv(fileName, index=False)
except BaseException as e:
    print(e)
