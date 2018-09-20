# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/9/20 8:46

@desc:

'''

import logging
import time
import sys
import asyncio
from btfxwss import BtfxWss
import time

# config
log = logging.getLogger(__name__)
fh = logging.FileHandler('../test.log')
fh.setLevel(logging.DEBUG)
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.DEBUG)

log.addHandler(sh)
log.addHandler(fh)
logging.basicConfig(level=logging.DEBUG, handlers=[fh, sh])

# 国内设置代理
wss = BtfxWss(http_proxy_host='127.0.0.1', http_proxy_port='1080')


# wss = BtfxWss()

# 订阅websocket
def subscribe():
    while not wss.conn.connected.is_set():
        time.sleep(1)
    # Subscribe to some channels
    wss.subscribe_to_ticker('BTCUSD')
    wss.subscribe_to_order_book('BTCUSD')
    wss.subscribe_to_candles('BTCUSD', '1h')
    wss.subscribe_to_candles('BTCUSD', '1D')
    wss.subscribe_to_trades('BTCUSD')
    wss.subscribe_to_raw_order_book('BTCUSD')


async def get_order_book():
    book = wss.books('BTCUSD')
    while not book.empty():
        re = parse_order_book(book.get())
        ts = int(re['timestamp'])
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        print('%s: Order Book: %s' % (date, str(re['data'])))


def parse_order_book(book):
    list = []
    ts = book[1]
    for x in book[0]:
        if type(x[0]) == type([]):
            for y in x:
                list.append({
                    'PRICE': y[0],
                    'AMOUNT': y[2]
                })
        else:
            list.append({
                'PRICE': x[0],
                'AMOUNT': x[2]
            })

    return {
        'data': list,
        'timestamp': ts
    }


async def get_candles_1h():
    candles_1h = wss.candles('BTCUSD', '1h')
    while not candles_1h.empty():
        ts = int(time.time())
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        print('%s: 1H candles: %s' % (date, candles_1h.get()))


async def get_candles_1D():
    candles_1D = wss.candles('BTCUSD', '1D')
    while not candles_1D.empty():
        ts = int(time.time())
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        print('%s: 1D candles: %s' % (date, candles_1D.get()))


async def get_ticker():
    ticker = wss.tickers('BTCUSD')
    while not ticker.empty():
        re = parse_ticker(ticker.get())
        ts = int(re['timestamp'])
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        print('%s: Ticker: %s' % (date, str(re['data'])))


def parse_ticker(ticker):
    list = []
    ts = ticker[1]
    for x in ticker[0]:
        list.append({
            'BID': x[0],
            'BID_SIZE': x[1],  # Size of the last highest bid
            'ASK': x[2],
            'ASK_SIZE': x[3],
            'DAILY_CHANGE': x[4],  # Amount that the last price has changed since yesterday
            'DAILY_CHANGE_PERC': x[5],  # Amount that the price has changed expressed in percentage terms
            'LAST_PRICE': x[6],
            'VOLUME': x[7],  # Daily volume
            'HIGH': x[8],  # Daily high
            'LOW': x[9]  # Daily low
        })
    re = {
        'data': list,
        'timestamp': ts
    }
    return re


async def get_trade():
    trade = wss.trades('BTCUSD')
    while not trade.empty():
        re = parse_trade(trade.get())
        ts = int(re['timestamp'])
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        print('%s: Trade: %s' % (date, str(re['data'])))


def parse_trade(trade):
    list = []
    ts = trade[1]
    for i in range(len(trade[0])):
        x = trade[0][i]
        if type(x) == type([]):
            if i == 0:
                for y in x:
                    list.append({
                        'PRICE': y[3],
                        'AMOUNT': y[2]
                    })
            else:
                list.append({
                    'PRICE': x[3],
                    'AMOUNT': x[2]
                })

    re = {
        'data': list,
        'timestamp': ts
    }
    return re

# These are the most granular books.
async def get_raw_book():
    raw_book = wss.raw_books('BTCUSD')
    while not raw_book.empty():
        re = parse_raw_book(raw_book.get())
        ts = int(re['timestamp'])
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        print('%s: Raw Book: %s' % (date, re['data']))


def parse_raw_book(book):
    list = []
    ts = book[1]
    for x in book[0]:
        if type(x[0]) == type([]):
            for y in x:
                list.append({
                    'PRICE': y[1],
                    'AMOUNT': y[2]
                })
        else:
            list.append({
                'PRICE': x[1],
                'AMOUNT': x[2]
            })

    return {
        'data': list,
        'timestamp': ts
    }


async def print_data():
    while True:
        # await get_order_book()
        await get_raw_book()
        await get_trade()
        await get_ticker()
        # await get_candles_1h()
        # await get_candles_1D()


def main():
    wss.start()
    subscribe()

    loop = asyncio.get_event_loop()
    while wss.conn.connected.is_set():
        loop.run_until_complete(print_data())
        time.sleep(1)

    print('Connect Close')

    # Unsubscribing from channels:
    # wss.unsubscribe_from_ticker('BTCUSD')
    # wss.unsubscribe_from_order_book('BTCUSD')

    # Shutting down the client:
    # wss.stop()


if __name__ == '__main__':
    main()

