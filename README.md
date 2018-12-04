# quantitative 量化交易

开发语言：python3.6

推荐使用Anaconda集合包：https://www.anaconda.com/

或者通过下面方式安装项目包：

pip install --upgrade pip

pip install -r requirements.txt



项目以poloniex数据为准
时间周期：30m, 1h, 2h, 4h

**项目结构** 
```
quantitative
│
├─docs 文件包，含数据库设计，接口设计等
│  ├─db sql文件夹
│  └─requirements.txt 项目依赖包
│
├─data 存放k线数据及图标等文件
│
├─job 抓取数据并存储实现类
│  ├─周期内买卖单成交量
│  ├─k线上下挂单量
│  └─周期K线
│
├─strategies 交易策略
│  
├─tf 机器学习文件夹，推测价格
│
├─test 测试文件夹
│
├─config.py 配置文件
│ 
└─tools 通用工具类

```
<br> 

nohup python3 -u ./quantitative/MACD_ETH_1H_test.py > macd_eth_1h.log 2>&1 &
nohup python3 -u ./quantitative/MACD_ETH_2H_test.py > macd_eth_2h.log 2>&1 &
nohup python3 -u ./quantitative/MACD_ETH_4H_test.py > macd_eth_4h.log 2>&1 &

nohup python3 -u ./quantitative/MACD_BTC_1H_test.py > macd_btc_1h.log 2>&1 &
nohup python3 -u ./quantitative/MACD_BTC_2H_test.py > macd_btc_2h.log 2>&1 &
nohup python3 -u ./quantitative/MACD_BTC_4H_test.py > macd_btc_4h.log 2>&1 &

nohup python3 -u ./quantitative/bitmex_macd_print.py > bitmex_macd_print.py.log 2>&1 &
