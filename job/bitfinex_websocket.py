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


def get_order_book():
    book = wss.books('BTCUSD')
    while not book.empty():
        print('Order Book: %s' % str(book.get()))


def get_candles():
    candles_1h = wss.candles('BTCUSD', '1h')
    # candles_1D = wss.candles('BTCUSD', '1D')
    while not candles_1h.empty():
        print('1H candles: %s' % str(candles_1h.get()))
        # print('1D candles: %s' % str(candles_1D.get()))


def get_ticker():
    ticker = wss.tickers('BTCUSD')
    while not ticker.empty():
        print('Ticker: %s' % str(ticker.get()))


def get_trade():
    trade = wss.trades('BTCUSD')
    while not trade.empty():
        print('Trade: %s' % str(trade.get()))


def get_raw_book():
    raw_book = wss.raw_books('BTCUSD')
    while not raw_book.empty():
        print('Raw Book: %s' % str(raw_book.get()))


if __name__ == '__main__':
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

    wss.start()
    subscribe()

    while wss.conn.connected.is_set():
        get_order_book()
        get_raw_book()
        get_trade()
        get_ticker()
        get_candles()
        time.sleep(1)

    print('Connect Close')



    # Unsubscribing from channels:
    # wss.unsubscribe_from_ticker('BTCUSD')
    # wss.unsubscribe_from_order_book('BTCUSD')

    # Shutting down the client:
    # wss.stop()
