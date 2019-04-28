#！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2019/4/28 10:31

@desc:

'''

from pytrends.request import TrendReq
import time


pytrend = TrendReq()

pytrend.build_payload(kw_list=['BTC USD', 'Buy Bitcoin'], timeframe='now 7-d')

# Interest Over Time
interest_over_time_df = pytrend.interest_over_time()
print('\n', interest_over_time_df.head())
# time.sleep(5)


# interest_by_region_df = pytrend.interest_by_region()
# print('\n', interest_by_region_df.head())
# time.sleep(5)


# Related Queries, returns a dictionary of dataframes
# related_queries_dict = pytrend.related_queries()
# print('\n', related_queries_dict)
# time.sleep(5)


# Get Google Hot Trends data
# trending_searches_df = pytrend.trending_searches()
# print('\n', trending_searches_df.head())
# time.sleep(5)


# Get Google Top Charts
# top_charts_df = pytrend.top_charts(cid='actors', date=201611)
# print('\n', top_charts_df.head())