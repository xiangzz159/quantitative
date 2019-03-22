# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2019/3/22 17:38

@desc: https://cryptowat.ch/docs/api#intro

'''

import json
import numpy as np
import os
import pandas as pd
from urllib import request
import time

def get_ohlcvc(url):
    try:
        print('URL: %s' % url)
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
        }

        req = request.Request(url=url, headers=headers, method='GET')
        openUrl = request.urlopen(req)

        r = openUrl.read()
        d = json.loads(r.decode())
        df = pd.DataFrame(d['result']['60'], columns=['Timestamp', 'High', 'Low', 'Open', 'Close', 'Volume', 'Cost'])
        return df
    except BaseException as e:
        print(e)

if __name__ == '__main__':
    df = get_ohlcvc('https://api.cryptowat.ch/markets/poloniex/btcusdt/ohlc?periods=60&after=1514736000')
    fileName = '../data/test.csv'
    df.to_csv(fileName, index=None)
