#！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/10/10 19:54

@desc:

'''

# 克隆自聚宽文章：https://www.joinquant.com/post/469
# 标题：【量化缠论】应用之维克多1-2-3法则
# 作者：莫邪的救赎

import pandas as pd
from datetime import timedelta, date


def initialize(context):
    g.security = ['600307.XSHG']
    set_universe(g.security)
    set_benchmark('600307.XSHG')
    set_commission(PerTrade(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))
    set_slippage(PriceRelatedSlippage())
    g.HighAfterEntry = {}  # 存放持仓股票历史最高价
    g.n = 5  # 获取几分钟k线
    endtime = context.current_dt.date()  # 结束时间
    starttime = endtime - 10 * timedelta(days=1)  # 开始时间
    ## 获取前几日的趋势
    ''' temp_data 包含处理后最后一个k线的数据
        zoushi 包含关系处理中关于走势的记录
        after_baohan 合并后的k线'''
    g.temp_data, g.zoushi, g.after_baohan = k_initialization(security=g.security[0], start=starttime, end=endtime,
                                                             n=g.n)
    ## 每日运行一次"根据跌幅止损"、"根据大盘跌幅止损"及更新订单
    run_daily(dapan_stoploss)  # 如不想加入大盘止损，注释此句即可
    run_daily(sell)
    run_daily(add_HighAfterEntry, time='after_close')  # 收盘后，根据成交订单将新买入的股票加入g.HighAfterEntry


def handle_data(context, data):
    security = g.security
    Cash = context.portfolio.cash

    ## 更新 g.HighAfterEntry,用于移动分级止盈
    if len(g.HighAfterEntry) > 0:
        stock = g.security[0]
        per_price = data[stock].pre_close
        if g.HighAfterEntry[stock] < per_price:
            g.HighAfterEntry[stock] = per_price
        else:
            pass

    ## 获取n分钟k线
    hour = context.current_dt.hour
    minute = context.current_dt.minute
    print
    'time: %s:%s' % (hour, minute)
    n = g.n
    if (hour == 9 and minute == 30) or (hour == 13 and minute == 00):
        pass
    else:
        if minute % n == 0:
            # 获得前n分钟的k线
            temp_hist = attribute_history(security[0], 5, '1m', ('open', 'close', 'high', 'low'))
            # 合成n分钟k线
            Fnk = temp_hist[-1:]
            Fnk.open = temp_hist.open[0]
            Fnk.high = max(temp_hist.high)
            Fnk.low = min(temp_hist.low)
            # 包含关系处理
            Input_k = pd.concat([g.temp_data, Fnk], axis=0)
            g.temp_data, g.zoushi, g.after_baohan = recognition_baohan(Input_k, g.zoushi, g.after_baohan)
            # 分型
            fenxing_type, fenxing_time, fenxing_plot, fenxing_data = recognition_fenxing(g.after_baohan)
            '''
            fenxing_type 记录分型点的类型，1为顶分型，-1为底分型
            fenxing_time 记录分型点的时间
            fenxing_plot 记录点的数值，为顶分型去high值，为底分型去low值
            fenxing_data 分型点的DataFrame值
            '''
            # print fenxing_type, fenxing_time, fenxing_plot, fenxing_data
            # 判断趋势是否反转，并买进
            if fenxing_type[0] == -1 and len(fenxing_type) > 7:
                location_1 = [i for i, a in enumerate(fenxing_type) if a == 1]  # 找出1在列表中的所有位置
                location_2 = [i for i, a in enumerate(fenxing_type) if a == -1]  # 找出-1在列表中的所有位置
                # 线段破坏
                case1 = fenxing_data.low[location_2[0]] > fenxing_data.low[location_2[1]]
                # 线段形成
                case2 = fenxing_data.high[location_1[1]] < fenxing_data.high[location_1[2]] < fenxing_data.high[
                    location_1[3]]
                case3 = fenxing_data.low[location_2[1]] < fenxing_data.low[location_2[2]] < fenxing_data.low[
                    location_2[3]]
                # 第i笔中底比第i+2笔顶高(扶助判断条件，根据实测加入之后条件太苛刻，很难有买入机会)
                case4 = fenxing_data.low[location_2[1]] > fenxing_data.high[location_1[3]]
                if case1 and case2 and case3:
                    # 买入
                    order_value(security[0], Cash)
                g.lowprice = fenxing_data.low[location_2[0]]

            # ## 构成假突破，则卖出
            # if len(context.portfolio.positions) > 0:
            #     price = data[security[0]].pre_close
            #     if price < g.lowprice :
            #         order_target(security[0], 0)

    # ## 构成假突破，则卖出
    # if len(context.portfolio.positions) > 0:
    #     price = data[security[0]].pre_close
    #     if price < g.lowprice :
    #         order_target(security[0], 0)

    ## 亏损率止损
    if len(context.portfolio.positions) > 0:
        price = data[security[0]].pre_close
        avg_cost = context.portfolio.positions[security[0]].avg_cost
        if (price - avg_cost) / avg_cost <= -0.1:
            order_target(security[0], 0)

    ## 分级移动止盈
    if len(context.portfolio.positions) > 0:
        stock = security[0]
        avg_cost = context.portfolio.positions[stock].avg_cost  # 获取成本
        high_price = g.HighAfterEntry[stock]
        profit_level = high_price / avg_cost - 1  # 收益率
        # 根据盈利设定stopLoss_price
        if profit_level < 0.2:
            stopLoss_price = high_price * 0.7
        elif 0.5 > profit_level > 0.2:
            stopLoss_price = high_price * 0.8
        elif profit_level >= 0.5:
            stopLoss_price = high_price * 0.9
        # 判断，并卖出
        per_price = data[stock].pre_close
        if per_price <= stopLoss_price:
            order_target(stock, 0)


def dapan_stoploss(context):
    ## 根据局大盘止损，具体用法详见dp_stoploss函数说明
    stock = g.security[0]
    stoploss = dp_stoploss(kernel=2, n=10, zs=0.03)
    if stoploss:
        if len(context.portfolio.positions) > 0:
            for stock in list(context.portfolio.positions.keys()):
                order_target(stock, 0)
        # return


def sell(context):
    # 根据跌幅卖出
    stock = g.security[0]
    hist = attribute_history(stock, 3, '1d', 'close', df=False)
    if ((1 - float(hist['close'][-1] / hist['close'][0])) >= 0.12):
        order_target(stock, 0)


def recognition_fenxing(after_baohan):
    '''
    从后往前找
    返回值：
    fenxing_type 记录分型点的类型，1为顶分型，-1为底分型
    fenxing_time 记录分型点的时间
    fenxing_plot 记录点的数值，为顶分型去high值，为底分型去low值
    fenxing_data 分型点的DataFrame值
    '''
    ## 找出顶和底
    temp_num = 0  # 上一个顶或底的位置
    temp_high = 0  # 上一个顶的high值
    temp_low = 0  # 上一个底的low值
    temp_type = 0  # 上一个记录位置的类型
    end = len(after_baohan)
    i = end - 2
    fenxing_type = []  # 记录分型点的类型，1为顶分型，-1为底分型
    fenxing_time = []  # 记录分型点的时间
    fenxing_plot = []  # 记录点的数值，为顶分型去high值，为底分型去low值
    fenxing_data = pd.DataFrame()  # 分型点的DataFrame值
    while (i >= 1):
        if len(fenxing_type) > 8:
            break
        else:
            case1 = after_baohan.high[i - 1] < after_baohan.high[i] and after_baohan.high[i] > after_baohan.high[
                i + 1]  # 顶分型
            case2 = after_baohan.low[i - 1] > after_baohan.low[i] and after_baohan.low[i] < after_baohan.low[
                i + 1]  # 底分型
            if case1:
                if temp_type == 1:  # 如果上一个分型为顶分型，则进行比较，选取高点更高的分型
                    if after_baohan.high[i] <= temp_high:
                        i -= 1
                    else:
                        temp_high = after_baohan.high[i]
                        temp_num = i
                        temp_type = 1
                        i -= 4
                elif temp_type == 2:  # 如果上一个分型为底分型，则记录上一个分型，用当前分型与后面的分型比较，选取同向更极端的分型
                    if temp_low >= after_baohan.high[i]:  # 如果上一个底分型的底比当前顶分型的顶高，则跳过当前顶分型。
                        i -= 1
                    else:
                        fenxing_type.append(-1)
                        fenxing_time.append(after_baohan.index[temp_num].strftime("%Y-%m-%d %H:%M:%S"))
                        fenxing_data = pd.concat([fenxing_data, after_baohan[temp_num:temp_num + 1]], axis=0)
                        fenxing_plot.append(after_baohan.high[i])
                        temp_high = after_baohan.high[i]
                        temp_num = i
                        temp_type = 1
                        i -= 4
                else:
                    if (after_baohan.low[i - 2] > after_baohan.low[i - 1] and after_baohan.low[i - 1] <
                            after_baohan.low[i]):
                        temp_low = after_baohan.low[i]
                        temp_num = i - 1
                        temp_type = 2
                        i -= 4
                    else:
                        temp_high = after_baohan.high[i]
                        temp_num = i
                        temp_type = 1
                        i -= 4

            elif case2:
                if temp_type == 2:  # 如果上一个分型为底分型，则进行比较，选取低点更低的分型
                    if after_baohan.low[i] >= temp_low:
                        i -= 1
                    else:
                        temp_low = after_baohan.low[i]
                        temp_num = i
                        temp_type = 2
                        i -= 4
                elif temp_type == 1:  # 如果上一个分型为顶分型，则记录上一个分型，用当前分型与后面的分型比较，选取同向更极端的分型
                    if temp_high <= after_baohan.low[i]:  # 如果上一个顶分型的底比当前底分型的底低，则跳过当前底分型。
                        i -= 1
                    else:
                        fenxing_type.append(1)
                        fenxing_time.append(after_baohan.index[temp_num].strftime("%Y-%m-%d %H:%M:%S"))
                        fenxing_data = pd.concat([fenxing_data, after_baohan[temp_num:temp_num + 1]], axis=0)
                        fenxing_plot.append(after_baohan.low[i])
                        temp_low = after_baohan.low[i]
                        temp_num = i
                        temp_type = 2
                        i -= 4
                else:
                    if (after_baohan.high[i - 2] < after_baohan.high[i - 1] and after_baohan.high[i - 1] >
                            after_baohan.high[i]):
                        temp_high = after_baohan.high[i]
                        temp_num = i - 1
                        temp_type = 1
                        i -= 4
                    else:
                        temp_low = after_baohan.low[i]
                        temp_num = i
                        temp_type = 2
                        i -= 4
            else:
                i -= 1
    return fenxing_type, fenxing_time, fenxing_plot, fenxing_data


def recognition_baohan(Input_k, zoushi, after_baohan):
    '''
    判断两根k线的包含关系
    temp_data 包含处理后最后一个k线的数据
    zoushi 包含关系处理中关于走势的记录
    Input_k 是temp_data与新n分钟k线的合集
    zoushi： 3-持平 4-向下 5-向上
    after_baohan 处理之后的k线
    '''
    import pandas as pd
    temp_data = Input_k[:1]
    case1_1 = temp_data.high[-1] > Input_k.high[1] and temp_data.low[-1] < Input_k.low[1]  # 第1根包含第2根
    case1_2 = temp_data.high[-1] > Input_k.high[1] and temp_data.low[-1] == Input_k.low[1]  # 第1根包含第2根
    case1_3 = temp_data.high[-1] == Input_k.high[1] and temp_data.low[-1] < Input_k.low[1]  # 第1根包含第2根
    case2_1 = temp_data.high[-1] < Input_k.high[1] and temp_data.low[-1] > Input_k.low[1]  # 第2根包含第1根
    case2_2 = temp_data.high[-1] < Input_k.high[1] and temp_data.low[-1] == Input_k.low[1]  # 第2根包含第1根
    case2_3 = temp_data.high[-1] == Input_k.high[1] and temp_data.low[-1] > Input_k.low[1]  # 第2根包含第1根
    case3 = temp_data.high[-1] == Input_k.high[1] and temp_data.low[-1] == Input_k.low[1]  # 第1根等于第2根
    case4 = temp_data.high[-1] > Input_k.high[1] and temp_data.low[-1] > Input_k.low[1]  # 向下趋势
    case5 = temp_data.high[-1] < Input_k.high[1] and temp_data.low[-1] < Input_k.low[1]  # 向上趋势
    if case1_1 or case1_2 or case1_3:
        if zoushi[-1] == 4:
            temp_data.high[-1] = Input_k.high[1]
        else:
            temp_data.low[-1] = Input_k.low[1]

    elif case2_1 or case2_2 or case2_3:
        temp_temp = temp_data[-1:]
        temp_data = Input_k[1:]
        if zoushi[-1] == 4:
            temp_data.high[-1] = temp_temp.high[0]
        else:
            temp_data.low[-1] = temp_temp.low[0]

    elif case3:
        zoushi.append(3)
        pass

    elif case4:
        zoushi.append(4)
        after_baohan = pd.concat([after_baohan, temp_data], axis=0)
        temp_data = Input_k[1:]

    elif case5:
        zoushi.append(5)
        after_baohan = pd.concat([after_baohan, temp_data], axis=0)
        temp_data = Input_k[1:]

    return temp_data, zoushi, after_baohan


def k_initialization(security, start, end, n=5):
    '''
    读入回测日期之前的多日k线用以判断之前的趋势
    返回值：
        temp_data 包含处理后最后一个k线的数据
        zoushi 包含关系处理中关于走势的记录
        after_baohan 合并后的k线
    '''
    import pandas as pd
    k_data = get_price(security, start_date=start, end_date=end, frequency='minute',
                       fields=['open', 'close', 'high', 'low'])
    ## 获取n分钟k线
    # 去除9:00与13:00的数据
    for i in range(len(k_data) / 242):
        team = list(k_data.index)
        x = [s.strftime("%Y-%m-%d %H:%M:%S") for s in team]
        y = filter(lambda t: "09:30:00" in t, x)
        k_data = k_data.drop(k_data.index[x.index(y[0])])
        del x[x.index(y[0])]
        y = filter(lambda t: "13:00:00" in t, x)
        k_data = k_data.drop(k_data.index[x.index(y[0])])
        del x[x.index(y[0])]
    # 计算n分钟K线
    Fnk = pd.DataFrame()
    for i in xrange(n, len(k_data) + 1, n):
        temp = k_data[i - n: i]
        temp_open = temp.open[0]
        temp_high = max(temp.high)
        temp_low = min(temp.low)
        temp_k = temp[-1:]
        temp_k.open = temp_open
        temp_k.high = temp_high
        temp_k.low = temp_low
        Fnk = pd.concat([Fnk, temp_k], axis=0)

    ## 判断包含关系
    after_baohan = pd.DataFrame()
    temp_data = Fnk[:1]
    zoushi = [3]  # 3-持平 4-向下 5-向上
    for i in xrange(len(Fnk)):
        case1_1 = temp_data.high[-1] > Fnk.high[i] and temp_data.low[-1] < Fnk.low[i]  # 第1根包含第2根
        case1_2 = temp_data.high[-1] > Fnk.high[i] and temp_data.low[-1] == Fnk.low[i]  # 第1根包含第2根
        case1_3 = temp_data.high[-1] == Fnk.high[i] and temp_data.low[-1] < Fnk.low[i]  # 第1根包含第2根
        case2_1 = temp_data.high[-1] < Fnk.high[i] and temp_data.low[-1] > Fnk.low[i]  # 第2根包含第1根
        case2_2 = temp_data.high[-1] < Fnk.high[i] and temp_data.low[-1] == Fnk.low[i]  # 第2根包含第1根
        case2_3 = temp_data.high[-1] == Fnk.high[i] and temp_data.low[-1] > Fnk.low[i]  # 第2根包含第1根
        case3 = temp_data.high[-1] == Fnk.high[i] and temp_data.low[-1] == Fnk.low[i]  # 第1根等于第2根
        case4 = temp_data.high[-1] > Fnk.high[i] and temp_data.low[-1] > Fnk.low[i]  # 向下趋势
        case5 = temp_data.high[-1] < Fnk.high[i] and temp_data.low[-1] < Fnk.low[i]  # 向上趋势
        if case1_1 or case1_2 or case1_3:
            if zoushi[-1] == 4:
                temp_data.high[-1] = Fnk.high[i]
            else:
                temp_data.low[-1] = Fnk.low[i]

        elif case2_1 or case2_2 or case2_3:
            temp_temp = temp_data[-1:]
            temp_data = Fnk[i:i + 1]
            if zoushi[-1] == 4:
                temp_data.high[-1] = temp_temp.high[0]
            else:
                temp_data.low[-1] = temp_temp.low[0]

        elif case3:
            zoushi.append(3)
            pass

        elif case4:
            zoushi.append(4)
            after_baohan = pd.concat([after_baohan, temp_data], axis=0)
            temp_data = Fnk[i:i + 1]

        elif case5:
            zoushi.append(5)
            after_baohan = pd.concat([after_baohan, temp_data], axis=0)
            temp_data = Fnk[i:i + 1]
    return temp_data, zoushi, after_baohan


def dp_stoploss(kernel=2, n=10, zs=0.03):
    '''
    方法1：当大盘N日均线(默认60日)与昨日收盘价构成“死叉”，则发出True信号
    方法2：当大盘N日内跌幅超过zs，则发出True信号
    '''
    # 止损方法1：根据大盘指数N日均线进行止损
    if kernel == 1:
        t = n + 2
        hist = attribute_history('000300.XSHG', t, '1d', 'close', df=False)
        temp1 = sum(hist['close'][1:-1]) / float(n)
        temp2 = sum(hist['close'][0:-2]) / float(n)
        close1 = hist['close'][-1]
        close2 = hist['close'][-2]
        if (close2 > temp2) and (close1 < temp1):
            return True
        else:
            return False
    # 止损方法2：根据大盘指数跌幅进行止损
    elif kernel == 2:
        hist1 = attribute_history('000300.XSHG', n, '1d', 'close', df=False)
        if ((1 - float(hist1['close'][-1] / hist1['close'][0])) >= zs):
            return True
        else:
            return False


def add_HighAfterEntry(context):
    # 将新买入的股票加入g.HighAfterEntry
    trades = get_orders()
    for t in trades.values():
        if t.is_buy and t.filled > 0:
            x = str(t.security)
            g.HighAfterEntry[x] = t.price
        elif not t.is_buy and t.filled > 0:
            xx = str(t.security)
            try:
                del g.HighAfterEntry[xx]
            except:
                g.HighAfterEntry[xx] = 0
    pass





