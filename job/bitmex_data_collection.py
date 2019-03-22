# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/11/28 10:59

@desc:

'''

import ccxt.async_support as ccxt
import asyncio


async def get_kline(symbol, periods_str, since, limit):
    ex = ccxt.bitmex()
    try:
        kline = await ex.fetch_ohlcv(symbol, periods_str, since, limit)
        await ex.close()
        return kline
    except ccxt.BaseError as e:
        print(type(e).__name__, str(e), str(e.args))
        raise e


limit = 750
symbol = 'BTC/USD'
periods = 60 * 1000
begin = 1514736000000
end = begin + limit * periods * 3
periods_str = '1m'
async_funcs = []
tt = end - begin
count = int(tt / periods / limit)
begin_ = begin
for i in range(count):
    async_funcs.append(get_kline(symbol, periods_str, begin_, limit))
    begin_ += periods * limit
loop = asyncio.get_event_loop()
klines = loop.run_until_complete(asyncio.gather(*async_funcs))
for kline in klines:
    print(kline)

# df = pd.DataFrame(k, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
# df['Timestamp'] = df['Timestamp'] / 1000
# fileName = '../data/bitmex.csv'
# df.to_csv(fileName, index=None)
