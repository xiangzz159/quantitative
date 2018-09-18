#！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/9/18 17:18

@desc:

'''

from stockstats import StockDataFrame
import pandas as pd

stock = StockDataFrame.retype(pd.read_csv(r'../data/BTC2016-now-1D.csv'))
stock['volume_delta']

# open delta against next 2 day
stock['open_2_d']

# open price change (in percent) between today and the day before yesterday
# 'r' stands for rate.
stock['open_-2_r']

# CR indicator, including 5, 10, 20 days moving average
stock['cr']
stock['cr-ma1']
stock['cr-ma2']
stock['cr-ma3']

# volume max of three days ago, yesterday and two days later
stock['volume_-3,2,-1_max']

# volume min between 3 days ago and tomorrow
stock['volume_-3~1_min']

# KDJ, default to 9 days
stock['kdjk']
stock['kdjd']
stock['kdjj']

# three days KDJK cross up 3 days KDJD
stock['kdj_3_xu_kdjd_3']

# 2 days simple moving average on open price
stock['open_2_sma']

# MACD
stock['macd']
# MACD signal line
stock['macds']
# MACD histogram
stock['macdh']

# bolling, including upper band and lower band
stock['boll']
stock['boll_ub']
stock['boll_lb']

# close price less than 10.0 in 5 days count
stock['close_10.0_le_5_c']

# CR MA2 cross up CR MA1 in 20 days count
stock['cr-ma2_xu_cr-ma1_20_c']

# 6 days RSI
stock['rsi_6']
# 12 days RSI
stock['rsi_12']

# 10 days WR
stock['wr_10']
# 6 days WR
stock['wr_6']

# CCI, default to 14 days
stock['cci']
# 20 days CCI
stock['cci_20']

# TR (true range)
stock['tr']
# ATR (Average True Range)
stock['atr']

# DMA, difference of 10 and 50 moving average
stock['dma']

# DMI
# +DI, default to 14 days
stock['pdi']
# -DI, default to 14 days
stock['mdi']
# DX, default to 14 days of +DI and -DI
stock['dx']
# ADX, 6 days SMA of DX, same as stock['dx_6_ema']
stock['adx']
# ADXR, 6 days SMA of ADX, same as stock['adx_6_ema']
stock['adxr']

# TRIX, default to 12 days
stock['trix']
# MATRIX is the simple moving average of TRIX
stock['trix_9_sma']

# VR, default to 26 days
stock['vr']
# MAVR is the simple moving average of VR
stock['vr_6_sma']