#！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/9/17 20:25

@desc:

'''

from bitmex_websocket import BitMEXWebsocket
import logging
from time import sleep


# Basic use of websocket.
def run():
    logger = setup_logger()

    # Instantiating the WS will make it connect. Be sure to add your api_key/api_secret.
    ws = BitMEXWebsocket(endpoint="https://www.bitmex.com/api/v1", symbol="XBTUSD",
                         api_key=None, api_secret=None).ws.run_forever(http_proxy_host='127.0.0.1', http_proxy_port='1080')

    logger.info("Instrument data: %s" % ws.get_instrument())

    # Run forever
    ws.ws.run_forever(http_proxy_host='127.0.0.1', http_proxy_port='1080')
    while(ws.ws.sock.connected):
        logger.info("Ticker: %s" % ws.get_ticker())
        if ws.api_key:
            logger.info("Funds: %s" % ws.funds())
        logger.info("Market Depth: %s" % ws.market_depth())
        logger.info("Recent Trades: %s\n\n" % ws.recent_trades())
        sleep(10)


def setup_logger():
    # Prints logger info to terminal
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Change this to DEBUG if you want a lot more info
    ch = logging.StreamHandler()
    # create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # add formatter to ch
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


if __name__ == "__main__":
    run()