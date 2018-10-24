# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/9/28 8:46

@desc:

'''

import config
import pymysql
import time


# K线拟合：1h->4h
def kline_fitting_1H_4H(kline):
    # 毫秒时间戳转化为秒
    begin_time = kline[0][0] / 1000 if len(str(kline[0][0])) == 13 else kline[0][0]
    begin_day = int(begin_time - begin_time % 14400) + 14400
    index = 0
    for i in range(len(kline)):
        t = kline[i][0] / 1000 if len(str(kline[0][0])) == 13 else kline[i][0]
        if int(t) == begin_day:
            index = i

    kline = kline[index:]
    num = len(kline) % 4
    kline_ = kline[len(kline) - num:]
    kline = kline[:len(kline) - num]

    l = []
    # 存在第4小时k线没走完的情况，例如00：00 - 03：02，这时候03：00-04：00仍然有k线数据
    for i in range(0, len(kline), 4):
        timestamp = int(kline[i][0] / 1000) if len(str(kline[0][0])) == 13 else kline[i][0]
        open = kline[i][1]
        high = max(kline[i][2], kline[i + 1][2], kline[i + 2][2], kline[i + 3][2])
        low = min(kline[i][3], kline[i + 1][3], kline[i + 2][3], kline[i + 3][3])
        close = kline[i + 3][4]
        volumn = kline[i][5] + kline[i + 1][5] + kline[i + 2][5] + kline[i + 3][5]
        l.append([timestamp, open, high, low, close, volumn])
    if len(kline_) > 2:
        timestamp = int(kline_[0][0] / 1000) if len(str(kline_[0][0])) == 13 else kline_[0][0]
        open = kline_[0][1]
        close = kline_[len(kline_) - 1][4]
        volumn = 0
        high = 0
        low = 2 << 16
        for k in kline_:
            volumn += k[5]
            high = max(high, k[2])
            low = min(low, k[3])
        l.append([timestamp, open, high, low, close, volumn])
    return l


# K线拟合
def kline_fitting(kline, n, fitting_time):
    # 毫秒时间戳转化为秒
    begin_time = kline[0][0] / 1000 if len(str(kline[0][0])) == 13 else kline[0][0]
    begin_day = int(begin_time - begin_time % fitting_time) + fitting_time
    index = 0
    for i in range(len(kline)):
        t = kline[i][0] / 1000 if len(str(kline[0][0])) == 13 else kline[i][0]
        if int(t) == begin_day:
            index = i
            break

    kline = kline[index:]
    num = len(kline) % n
    kline = kline[:len(kline) - num]

    l = []
    for i in range(0, len(kline), n):
        timestamp = int(kline[i][0] / 1000) if len(str(kline[0][0])) == 13 else kline[i][0]
        open = kline[i][1]
        high = kline[i][2]
        for j in range(i + 1, i + n):
            high = max(high, kline[j][2])
        low = kline[i][3]
        for j in range(i + 1, i + n):
            low = min(low, kline[j][3])
        close = kline[i + n - 1][4]
        volumn = 0
        for j in range(i, i + n):
            volumn += kline[j][5]
        l.append([timestamp, open, high, low, close, volumn])
    return l

def fetch_data(sql):
    data = None
    flag = False
    try:
        db = pymysql.connect(host=config.DB_HOST, user=config.DB_USER, password=config.DB_PASSWORD, db=config.DB_NAME)
        cursor = db.cursor()
        cursor.execute(sql)
        data = cursor.fetchone()
        flag = True
    except BaseException as e:
        print(get_time(), 'fetch data failed, error:%s, sql:%s' % (str(e), sql))
    finally:
        db.close()
        return data, flag


def fetch_datas(sql):
    data = None
    flag = False
    try:
        db = pymysql.connect(host=config.DB_HOST, user=config.DB_USER, password=config.DB_PASSWORD, db=config.DB_NAME)
        cursor = db.cursor()
        cursor.execute(sql)
        data = cursor.fetchall()
        flag = True
    except BaseException as e:
        print(get_time(), 'fetch datas failed, error:%s, sql:%s' % (str(e), sql))
    finally:
        db.close()
        return data, flag


def execute_sql(sql):
    flag = False
    id = None
    try:
        db = pymysql.connect(host=config.DB_HOST, user=config.DB_USER, password=config.DB_PASSWORD, db=config.DB_NAME)
        cursor = db.cursor()
        cursor.execute(sql)
        db.commit()
        id = cursor.lastrowid
        flag = True
    except BaseException as e:
        db.rollback()
        print(get_time(), 'execute sql failed, error:%s, sql:%s' % (str(e), sql))
    finally:
        db.close()
        return id, flag


def get_time():
    timestamp = time.time() + 8 * 3600  # 处理时间戳的时区
    time_local = time.localtime(timestamp)
    return time.strftime("%Y-%m-%d %H:%M", time_local)


def insert_quantitative_order(params={}):
    sql = 'INSERT INTO `quantitative_order` (`position_id`, `account_id`, `username`, `exchange_code`, `symbol`,`order_id`, `price`, `amount`, `average_price`, `filled`, `side`, `type`, `status`, `info`, `create_time`) VALUES ("%d", "%s", "%s", "%s", "%s", "%s", "%.10f", "%.10f", "%.10f", "%.10f", "%s", "%s", "%s", "%s", "%d")' % (
        params['position_id'], params['account_id'], params['username'], params['exchange_code'], params['symbol'],
        params['id'],
        params['price'], params['amount'], params['price'], params['filled'], params['side'], params['type'],
        params['status'], params['info'], int(params['timestamp'] / 1000))
    print(get_time(), 'insert into quantitative order sql:', sql)
    return sql


def insert_positions(params={}):
    sql = 'INSERT INTO `positions` (`account_id`, `username`, `exchange_code`, `symbol`, `open_price`, `open_amount`, `open_time`, `stop_price`, `strategies`, `side`) VALUES ("%s", "%s", "%s", "%s", "%.10f", "%.10f", "%d", "%.10f", "%s", "%s") ' % (
        params['account_id'], params['username'], params['exchange_code'], params['symbol'],
        params['price'], params['amount'], params['timestamp'],
        params['stop_price'], params['strategies'], params['side'])
    print(get_time(), 'insert into positions sql:', sql)
    return sql
