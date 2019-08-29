# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2019/7/5 10:10

@desc: BitMEX 量化回测

'''

import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import random
from datetime import datetime

class BmBackTest(object):
    asset = 0  # 资产
    float_profit = 0    # 浮动盈亏
    open_price = 0  # 开仓价格
    open_amount = 0  # 开仓数量
    open_qty = 0  # 开仓价值（BTC）
    stop_price = 0
    side = 'wait'
    market_rate = 0.00075
    limit_rate = -0.00025
    data_arr = []

    def __init__(self, config={}):
        for key in config:
            if hasattr(self, key) and isinstance(getattr(self, key), dict):
                setattr(self, key, self.deep_extend(getattr(self, key), config[key]))
            else:
                setattr(self, key, config[key])

    def create_order(self, side, order_type, price, amount):
        if (self.side == 'long' and side == 'short') | (self.side == 'short' and side == 'long'):
            self.close_positions(price, 'market')

        if self.side == 'wait':
            self.side = side
            self.open_price = price
            self.open_amount = amount
            qty = amount / price
            self.open_qty = qty
            fee = 0
            if order_type == "market":
                fee = qty * self.market_rate
            else:
                fee = qty * self.limit_rate
            self.asset = self.asset - fee

        elif self.side in ['long', 'short'] and self.side == side:  # 加仓
            self.open_price = (self.open_amount * self.open_price + price * amount) / (amount + self.open_amount)
            self.open_amount += amount
            qty = amount / price
            self.open_qty += qty
            fee = 0
            if order_type == "market":
                fee = qty * self.market_rate
            else:
                fee = qty * self.limit_rate
            self.asset = self.asset - fee

        self.add_plt_data(self.asset + self.float_profit, price)

    def close_positions(self, price, order_type):
        if self.side != 'wait':
            close_qty = self.open_amount / price
            close_fee = 0
            if order_type == 'market':
                close_fee = close_qty * self.market_rate
            else:
                close_fee = close_qty * self.limit_rate

            self.asset -= close_fee
            profit = self.open_qty - close_qty if self.side == 'long' else close_qty - self.open_qty
            self.asset += profit
            # init
            self.float_profit = 0  # 浮动盈亏
            self.open_price = 0  # 开仓价格
            self.open_amount = 0  # 开仓数量
            self.open_qty = 0  # 开仓价值（BTC）
            self.stop_price = 0
            self.side = 'wait'
        self.add_plt_data(self.asset + self.float_profit, price)

    def add_data(self, price, high=None, low=None):
        if self.side != 'wait' and self.stop_price > 0:
            if (self.side == 'long' and low and self.stop_price > low) or (self.side == 'short' and high and self.stop_price < high):
                self.close_positions(self.stop_price, 'market')
                return

        close_qty = self.open_amount / price
        profit = self.open_qty - close_qty if self.side == 'long' else close_qty - self.open_qty
        self.float_profit = profit
        self.add_plt_data(self.asset + self.float_profit, price)

    def add_plt_data(self, asset, price):
        self.data_arr.append([asset, price])

    def show(self, title='Back Test'):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        df = pd.DataFrame(self.data_arr, columns=['assets', 'price'])
        ax = df[['price', 'assets']].plot(figsize=(20, 10), grid=True, xticks=df.index, rot=90, subplots=True,
                                          style='b')
        interval = int(len(df) / (40 - 1))
        # 设置x轴刻度数量
        ax[0].xaxis.set_major_locator(ticker.MaxNLocator(40))
        # 以date为x轴刻度
        ax[0].set_xticklabels(df.index[::interval])
        # 美观x轴刻度
        plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
        ax[0].set_title(title)
        plt.show()


    def real_time_show(self, title = 'Back Test'):
        # 动态实时绘图：
        # https://blog.csdn.net/u013950379/article/details/87936999
        ax = []  # 保存图1数据
        ay = []
        bx = []  # 保存图2数据
        by = []
        num = 0  # 计数
        plt.ion()  # 开启一个画图的窗口进入交互模式，用于实时更新数据
        # plt.rcParams['savefig.dpi'] = 200 #图片像素
        # plt.rcParams['figure.dpi'] = 200 #分辨率
        # plt.rcParams['figure.figsize'] = (10, 10)  # 图像显示大小
        # plt.rcParams['lines.linewidth'] = 0.5  # 设置曲线线条宽度
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 防止中文标签乱码，还有通过导入字体文件的方法
        plt.rcParams['axes.unicode_minus'] = False
        while num < len(self.data_arr):
            plt.clf()  # 清除刷新前的图表，防止数据量过大消耗内存
            plt.suptitle(title, fontsize=20)  # 添加总标题，并设置文字大小
            # 图表1
            ax.append(num)  # 追加x坐标值
            ay.append(self.data_arr[num][1])  # 追加y坐标值
            agraphic = plt.subplot(2, 1, 1)
            # agraphic.set_title('价格')  # 添加子标题
            agraphic.set_ylabel('BTC价格', fontsize=15)
            plt.plot(ax, ay)  # 等于agraghic.plot(ax,ay,'g-')

            # 图表2
            bx.append(num)
            by.append(self.data_arr[num][0])
            bgraghic = plt.subplot(2, 1, 2)
            bgraghic.set_ylabel("资产变化", fontsize=15)
            bgraghic.set_xlabel('x轴', fontsize=10)  # 添加轴标签
            bgraghic.plot(bx, by)

            plt.pause(0.05)  # 设置暂停时间，太快图表无法正常显示
            num = num + 1

        plt.ioff()  # 关闭画图的窗口，即关闭交互模式
        plt.show()  # 显示图片，防止闪退
