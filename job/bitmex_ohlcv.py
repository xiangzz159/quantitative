# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2019/3/22 17:51

@desc:

'''

import ccxt
import time
import numpy as np
import pandas as pd

limit = 500
symbol = 'BTC/USD'
periods = 60 * 1000 * 60
begin = 1514736000000
end = 1551412740000
count = int((end - begin) / limit / periods)
periods_str = '1h'
ex = ccxt.bitmex()
klines = []
for i in range(count):
    try:
        kline = ex.fetch_ohlcv(symbol, periods_str, begin, limit)
        klines.append(kline)
        begin += limit * periods
        time.sleep(1)
        if i % 100 == 0:
            print(i, klines)
    except Exception as e:
        print(e)
    finally:
        time.sleep(2)

data = np.array(klines[0])
for i in range(1, len(klines)):
    data = np.append(data, klines[i], axis=0)
df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df.to_csv('/home/centos/BitMEX-180101-190227-1H.csv', index=False)
