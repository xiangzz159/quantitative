#！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/11/28 10:59

@desc:

'''

import ccxt
import time
import pandas as pd

limit = 750
ex = ccxt.bitmex()
since = int(time.time()) * 1000 - 3600000 * (limit - 1)
k = ex.fetch_ohlcv('BTC/USD', '1h', since, limit)
df = pd.DataFrame(k, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
df['Timestamp'] = df['Timestamp'] / 1000
fileName = '../data/bitmex.csv'
df.to_csv(fileName, index=None)