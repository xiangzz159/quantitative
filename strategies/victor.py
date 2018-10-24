#！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/10/12 15:55

@desc:

'''

import pandas as pd
from tools.stockstats import StockDataFrame
import time


def k_analysis(k):
    # 转DataFrame格式
    k_data = pd.DataFrame(k, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    t = k[0][0]
    # 毫秒级
    if len(str(t)) == 13:
        k_data['timestamp'] = k_data['timestamp'] / 1000
    stock = StockDataFrame.retype(k_data)
    stock['middle_hl']
    k_data['date'] = k_data['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
    k_data['date'] = pd.to_datetime(k_data['date'])

    after_fenxing = pd.DataFrame()
    temp_data = k_data.iloc[0]
    zoushi = []  # 0:持平 -1:向下 1:向上

    for i, row in k_data.iterrows():
        # 第一根包含第二根
        case1_1 = temp_data['high'] > row['high'] and temp_data['low'] < row['low']
        case1_2 = temp_data['high'] > row['high'] and temp_data['low'] == row['low']
        case1_3 = temp_data['high'] == row['high'] and temp_data['low'] < row['low']
        # 第二根包含第一根
        case2_1 = temp_data['high'] < row['high'] and temp_data['low'] > row['low']
        case2_2 = temp_data['high'] < row['high'] and temp_data['low'] == row['low']
        case2_3 = temp_data['high'] == row['high'] and temp_data['low'] > row['low']
        # 第一根等于第二根
        case3 = temp_data['high'] == row['high'] and temp_data['low'] == row['low']
        # 向下趋势
        case4 = temp_data['high'] > row['high'] and temp_data['low'] > row['low']
        # 向上趋势
        case5 = temp_data['high'] < row['high'] and temp_data['low'] < row['low']

        if case1_1 or case1_2 or case1_3:
            if zoushi[-1] == -1:
                temp_data['high'] = row['high']
            else:
                temp_data['low'] = row['low']
        elif case2_1 or case2_2 or case2_3:
            temp_temp = temp_data
            temp_data = row
            if zoushi[-1] == -1:
                temp_data['high'] = temp_temp['high']
            else:
                temp_data['low'] = temp_temp['low']
        elif case3:
            zoushi.append(0)
            pass
        elif case4:
            zoushi.append(-1)
            # 使用默认index: ignore_index=True:
            after_fenxing = pd.concat([after_fenxing, temp_data.to_frame().T], ignore_index=True)
            temp_data = row
        elif case5:
            zoushi.append(1)
            after_fenxing = pd.concat([after_fenxing, temp_data.to_frame().T], ignore_index=True)
            temp_data = row

    # 因为使用candlestick2函数，要求输入open、close、high、low。为了美观，处理k线的最大最小值、开盘收盘价，之后k线不显示影线。
    for i, row in after_fenxing.iterrows():
        if row['open'] > row['close']:
            row['open'] = row['high']
            row['close'] = row['low']
        else:
            row['open'] = row['low']
            row['close'] = row['high']

    # 找出顶和底
    temp_num = 0  # 上一个顶或底的位置
    temp_high = 0  # 上一个顶的high值
    temp_low = 0  # 上一个底的low值
    temp_type = 0  # 上一个记录位置的类型 1-顶分型， 2-底分型
    i = 1
    fenxing_type = []  # 记录分型点的类型，1为顶分型，-1为底分型
    fenxing_time = []  # 记录分型点的时间
    fenxing_plot = []  # 记录点的数值，为顶分型去high值，为底分型去low值
    fenxing_data = pd.DataFrame()  # 分型点的DataFrame值
    interval = 4
    while (i < len(after_fenxing) - 1):
        # 顶分型
        case1 = after_fenxing.high[i - 1] < after_fenxing.high[i] and after_fenxing.high[i] > after_fenxing.high[i + 1]
        # 底分型
        case2 = after_fenxing.low[i - 1] > after_fenxing.low[i] and after_fenxing.low[i] < after_fenxing.low[i + 1]
        if case1:
            # 如果上一个分型为顶分型，则进行比较，选取更高的分型
            if temp_type == 1:
                if after_fenxing.high[i] <= temp_high:
                    i += 1
                    continue
                else:
                    temp_high = after_fenxing.high[i]
                    temp_num = i
                    temp_type = 1
                    i += interval
            # 如果上一个分型为底分型，则记录上一个分型，用当前分型与后面的分型比较，选取通向更极端的分型
            elif temp_type == 2:
                if temp_low >= after_fenxing.high[i]:
                    i += 1
                else:
                    fenxing_type.append(-1)
                    fenxing_time.append(after_fenxing.date[temp_num])
                    fenxing_data = pd.concat([fenxing_data, after_fenxing[temp_num: temp_num + 1]], axis=0)
                    fenxing_plot.append(after_fenxing.high[i])
                    temp_high = after_fenxing.high[i]
                    temp_num = i
                    temp_type = 1
                    i += interval
            else:
                temp_high = after_fenxing.high[i]
                temp_num = i
                temp_type = 1
                i += interval
        elif case2:
            # 如果上一个分型为底分型，则进行比较，选取低点更低的分型
            if temp_type == 2:
                if after_fenxing.low[i] >= temp_low:
                    i += 1
                    continue
                else:
                    temp_low = after_fenxing.low[i]
                    temp_num = i
                    temp_type = 2
                    i += interval
            # 如果上一个分型为顶分型，则记录上一个分型，用当前分型与后面的分型比较，选取通向更极端的分型
            elif temp_type == 1:
                # 如果上一个顶分型的底比当前底分型的底低，则跳过当前的底分型
                if temp_high <= after_fenxing.low[i]:
                    i += 1
                else:
                    fenxing_type.append(1)
                    fenxing_time.append(after_fenxing.date[temp_num])
                    fenxing_data = pd.concat([fenxing_data, after_fenxing[temp_num: temp_num + 1]], axis=0)
                    fenxing_plot.append(after_fenxing.low[i])
                    temp_low = after_fenxing.low[i]
                    temp_num = i
                    temp_type = 2
                    i += interval
            else:
                temp_low = after_fenxing.low[i]
                temp_num = i
                temp_type = 2
                i += interval
        else:
            i += 1
    return fenxing_type, fenxing_time, fenxing_plot, fenxing_data

def analysis(k):
    fenxing_type, fenxing_time, fenxing_plot, fenxing_data = k_analysis(k)
    if len(fenxing_type) > 7:
        if fenxing_type[0] == -1:
            location_1 = [i for i, a in enumerate(fenxing_type) if a == 1] # 找出1在列表中的所有位置
            location_2 = [i for i, a in enumerate(fenxing_type) if a == -1] # 找出-1在列表中的所有位置
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
                print('create long order')


if __name__ == '__main__':
    # 1. 获取k线数据
    # ex = ccxt.bitmex()
    # limit = 500
    # since = int(time.time()) * 1000 - 3600000 * (limit - 1)
    # kline = ex.fetch_ohlcv('BTC/USD', '1h', since, limit)

    # timestamp(ms), open, high, low, close, volumn
    k = [[1536235200000, 6399, 6405.5, 6376.5, 6387, 98364415], [1536238800000, 6387, 6500, 6382, 6471.5, 276723108],
         [1536242400000, 6471.5, 6498, 6435.5, 6453.5, 178547220],
         [1536246000000, 6453.5, 6454, 6381, 6409.5, 196321773],
         [1536249600000, 6409.5, 6436.5, 6394.5, 6415.5, 89198297],
         [1536253200000, 6415.5, 6426.5, 6362, 6362, 70321160],
         [1536256800000, 6362, 6440, 6314.5, 6415, 129225982], [1536260400000, 6415, 6460, 6412.5, 6455, 77726206],
         [1536264000000, 6455, 6455.5, 6418, 6437, 62732114], [1536267600000, 6437, 6448, 6408, 6429, 51770466],
         [1536271200000, 6429, 6454, 6421.5, 6451, 38977358], [1536274800000, 6451, 6527.5, 6451, 6487, 199630602],
         [1536278400000, 6487, 6487.5, 6450, 6470, 84839720], [1536282000000, 6470, 6506.5, 6469.5, 6501.5, 66355857],
         [1536285600000, 6501.5, 6523, 6481, 6487, 116546484], [1536289200000, 6487, 6497, 6486, 6486.5, 36193463],
         [1536292800000, 6486.5, 6487, 6466.5, 6476.5, 60391386],
         [1536296400000, 6476.5, 6497, 6476, 6496.5, 36817046],
         [1536300000000, 6496.5, 6519.5, 6446, 6458.5, 159595620],
         [1536303600000, 6458.5, 6479, 6458, 6477.5, 55766267],
         [1536307200000, 6477.5, 6477.5, 6423, 6447, 199661855], [1536310800000, 6447, 6454, 6425, 6450, 80773567],
         [1536314400000, 6450, 6450.5, 6323.5, 6366.5, 407797492],
         [1536318000000, 6366.5, 6386.5, 6342.5, 6366.5, 120993069],
         [1536321600000, 6366.5, 6439, 6366, 6424, 199946185],
         [1536325200000, 6424, 6449, 6418, 6435, 95877802], [1536328800000, 6435, 6472, 6392, 6413, 200158523],
         [1536332400000, 6413, 6436, 6344, 6375, 153921613], [1536336000000, 6375, 6494.5, 6358, 6441, 302682854],
         [1536339600000, 6441, 6446.5, 6377, 6413.5, 179668788], [1536343200000, 6413.5, 6430, 6398.5, 6411, 53292003],
         [1536346800000, 6411, 6426.5, 6402.5, 6417, 34293174], [1536350400000, 6417, 6450, 6411, 6448, 48033530],
         [1536354000000, 6448, 6447.5, 6399, 6418.5, 47299401], [1536357600000, 6418.5, 6419, 6390, 6397, 52325977],
         [1536361200000, 6397, 6413.5, 6385, 6398, 37787043], [1536364800000, 6398, 6440, 6397.5, 6426, 62275634],
         [1536368400000, 6426, 6436, 6414, 6416.5, 26987352], [1536372000000, 6416.5, 6438.5, 6416, 6429, 34187589],
         [1536375600000, 6429, 6434, 6400, 6410.5, 54306340], [1536379200000, 6410.5, 6420, 6395, 6413, 35709781],
         [1536382800000, 6413, 6434, 6412.5, 6426, 33094224], [1536386400000, 6426, 6451, 6425.5, 6429.5, 71168288],
         [1536390000000, 6429.5, 6434.5, 6427, 6432, 20905636], [1536393600000, 6432, 6468, 6431.5, 6448, 78849396],
         [1536397200000, 6448, 6448.5, 6425.5, 6446, 61705684], [1536400800000, 6446, 6448, 6437, 6441.5, 22699561],
         [1536404400000, 6441.5, 6445.5, 6421, 6427.5, 52153950], [1536408000000, 6427.5, 6428, 6350, 6411, 188021598],
         [1536411600000, 6411, 6418, 6391.5, 6413.5, 57686194], [1536415200000, 6413.5, 6414, 6385, 6402.5, 71474384],
         [1536418800000, 6402.5, 6427, 6391.5, 6424, 86098137], [1536422400000, 6424, 6426, 6371.5, 6377, 89982203],
         [1536426000000, 6377, 6400, 6100, 6205, 386844194], [1536429600000, 6205, 6210, 6130, 6187.5, 308972667],
         [1536433200000, 6187.5, 6197, 6151.5, 6174.5, 92887270],
         [1536436800000, 6174.5, 6180.5, 6100, 6164.5, 166926728],
         [1536440400000, 6164.5, 6180, 6151, 6178, 59335956], [1536444000000, 6178, 6188, 6155.5, 6172, 60817664],
         [1536447600000, 6172, 6192, 6165.5, 6180.5, 46063469], [1536451200000, 6180.5, 6185.5, 6165, 6175, 41824094],
         [1536454800000, 6175, 6197, 6174.5, 6197, 49291512], [1536458400000, 6197, 6197, 6155, 6164.5, 56588229],
         [1536462000000, 6164.5, 6172.5, 6140, 6169, 58095428], [1536465600000, 6169, 6189.5, 6167, 6170, 51618809],
         [1536469200000, 6170, 6181.5, 6169.5, 6175, 21014090], [1536472800000, 6175, 6187, 6175, 6177, 26666448],
         [1536476400000, 6177, 6184, 6175, 6181, 17050501], [1536480000000, 6181, 6188, 6180, 6183, 25718484],
         [1536483600000, 6183, 6483.5, 6178, 6382, 342124183], [1536487200000, 6382, 6389.5, 6353.5, 6354.5, 98866056],
         [1536490800000, 6354.5, 6372.5, 6354.5, 6372, 63318178],
         [1536494400000, 6372, 6388.5, 6360, 6376.5, 71828499],
         [1536498000000, 6376.5, 6437.5, 6376, 6378.5, 98906250],
         [1536501600000, 6378.5, 6398.5, 6364, 6368.5, 81072546],
         [1536505200000, 6368.5, 6373.5, 6352, 6360.5, 76253407],
         [1536508800000, 6360.5, 6371, 6357.5, 6369, 37217017],
         [1536512400000, 6369, 6374, 6363, 6366.5, 26659378], [1536516000000, 6366.5, 6380.5, 6366, 6367, 33577732],
         [1536519600000, 6367, 6382.5, 6363.5, 6376.5, 31304802],
         [1536523200000, 6376.5, 6384.5, 6373, 6383.5, 22272141],
         [1536526800000, 6383.5, 6384.5, 6361, 6363, 31826957], [1536530400000, 6363, 6363, 6208, 6248, 342081246],
         [1536534000000, 6248, 6284.5, 6227, 6240, 144419778], [1536537600000, 6240, 6257.5, 6226.5, 6248, 85142358],
         [1536541200000, 6248, 6385, 6248.5, 6317.5, 178325815],
         [1536544800000, 6317.5, 6328.5, 6303, 6311.5, 64202384],
         [1536548400000, 6311.5, 6311.5, 6287.5, 6303, 49914267],
         [1536552000000, 6303, 6303.5, 6270.5, 6273, 49971031],
         [1536555600000, 6273, 6289.5, 6273, 6284, 42665534], [1536559200000, 6284, 6310, 6279, 6310, 50449894],
         [1536562800000, 6310, 6317.5, 6294.5, 6305.5, 67886197],
         [1536566400000, 6305.5, 6310.5, 6275, 6282.5, 69344556],
         [1536570000000, 6282.5, 6285.5, 6223, 6279.5, 125840549],
         [1536573600000, 6279.5, 6300, 6235, 6288, 119475515],
         [1536577200000, 6288, 6310, 6266, 6299.5, 89582659], [1536580800000, 6299.5, 6300, 6279, 6292, 50274545],
         [1536584400000, 6292, 6305.5, 6262, 6289, 124502587], [1536588000000, 6289, 6346, 6288, 6297.5, 156442079],
         [1536591600000, 6297.5, 6301.5, 6276, 6287, 70934156], [1536595200000, 6287, 6289.5, 6225, 6253, 146410249],
         [1536598800000, 6253, 6268.5, 6232, 6257, 103030606], [1536602400000, 6257, 6300, 6242.5, 6276, 76333970],
         [1536606000000, 6276, 6305, 6271.5, 6289, 51958915], [1536609600000, 6289, 6300, 6265, 6300, 58007091],
         [1536613200000, 6300, 6302, 6285, 6287, 40384472], [1536616800000, 6287, 6287.5, 6261, 6282.5, 41847716],
         [1536620400000, 6282.5, 6310, 6280.5, 6300, 39854710], [1536624000000, 6300, 6341, 6281, 6309, 109389635],
         [1536627600000, 6309, 6394, 6308.5, 6394, 140962083],
         [1536631200000, 6394, 6434.5, 6337.5, 6345.5, 166847428],
         [1536634800000, 6345.5, 6364.5, 6341, 6343.5, 58744899],
         [1536638400000, 6343.5, 6346, 6314.5, 6318, 68349340],
         [1536642000000, 6318, 6325, 6311, 6320.5, 47078411], [1536645600000, 6320.5, 6333.5, 6320, 6324, 30416780],
         [1536649200000, 6324, 6324, 6288, 6299.5, 107286853], [1536652800000, 6299.5, 6310.5, 6290, 6310, 41891575],
         [1536656400000, 6310, 6329.5, 6310, 6323.5, 50699862], [1536660000000, 6323.5, 6324, 6301, 6305, 48857289],
         [1536663600000, 6305, 6309, 6226, 6262.5, 245964798], [1536667200000, 6262.5, 6278.5, 6252, 6266, 76759246],
         [1536670800000, 6266, 6283.5, 6265, 6271.5, 56932729],
         [1536674400000, 6271.5, 6275.5, 6140, 6204.5, 149156265],
         [1536678000000, 6204.5, 6248, 6180, 6247.5, 207192977],
         [1536681600000, 6247.5, 6260, 6235.5, 6245.5, 80790390],
         [1536685200000, 6245.5, 6245.5, 6223, 6231, 46391771], [1536688800000, 6231, 6241.5, 6192, 6237.5, 106907946],
         [1536692400000, 6237.5, 6280, 6229.5, 6262.5, 70358296],
         [1536696000000, 6262.5, 6262.5, 6243, 6250.5, 42647897],
         [1536699600000, 6250.5, 6315.5, 6234, 6307, 123993961], [1536703200000, 6307, 6317, 6283.5, 6287, 74450144],
         [1536706800000, 6287, 6292, 6268.5, 6278.5, 56430458], [1536710400000, 6278.5, 6282, 6252, 6252, 65511797],
         [1536714000000, 6252, 6265, 6246, 6256, 45417815], [1536717600000, 6256, 6263.5, 6234, 6248, 109523733],
         [1536721200000, 6248, 6264, 6220, 6257.5, 88801455], [1536724800000, 6257.5, 6279, 6252.5, 6261.5, 50441354],
         [1536728400000, 6261.5, 6294, 6261, 6268.5, 64690302], [1536732000000, 6268.5, 6269, 6238, 6247, 66730400],
         [1536735600000, 6247, 6253, 6220.5, 6238, 103255249], [1536739200000, 6238, 6244, 6181, 6235, 164079037],
         [1536742800000, 6235, 6247, 6232, 6240, 66544719], [1536746400000, 6240, 6252.5, 6228, 6228, 62662946],
         [1536750000000, 6228, 6241.5, 6212.5, 6231, 96255543], [1536753600000, 6231, 6277, 6230.5, 6241.5, 121506214],
         [1536757200000, 6241.5, 6286.5, 6241, 6286.5, 115619449],
         [1536760800000, 6286.5, 6289, 6245, 6264.5, 115003070],
         [1536764400000, 6264.5, 6273, 6252, 6260.5, 49705740],
         [1536768000000, 6260.5, 6270, 6255.5, 6260.5, 38616186],
         [1536771600000, 6260.5, 6307, 6258.5, 6271.5, 116135240],
         [1536775200000, 6271.5, 6345.5, 6271.5, 6292, 122297682],
         [1536778800000, 6292, 6314, 6287.5, 6298.5, 79486621], [1536782400000, 6298.5, 6299, 6286, 6298, 32010512],
         [1536786000000, 6298, 6320, 6293.5, 6300.5, 51820853],
         [1536789600000, 6300.5, 6335, 6297.5, 6334.5, 58241868],
         [1536793200000, 6334.5, 6335.5, 6313.5, 6322, 63411073],
         [1536796800000, 6322, 6437.5, 6321.5, 6404.5, 376271513],
         [1536800400000, 6404.5, 6406.5, 6381, 6387, 71180080], [1536804000000, 6387, 6387.5, 6370, 6386, 43180438],
         [1536807600000, 6386, 6396.5, 6375.5, 6379.5, 51314916],
         [1536811200000, 6379.5, 6396.5, 6379.5, 6387.5, 40982229],
         [1536814800000, 6387.5, 6391.5, 6356.5, 6373.5, 61983908],
         [1536818400000, 6373.5, 6380, 6368, 6377, 34537365],
         [1536822000000, 6377, 6391.5, 6376.5, 6377.5, 40645725],
         [1536825600000, 6377.5, 6425, 6377.5, 6405, 106520073],
         [1536829200000, 6405, 6495, 6401, 6427.5, 218681812], [1536832800000, 6427.5, 6448, 6416.5, 6423, 117248952],
         [1536836400000, 6423, 6468, 6416, 6453, 97771638], [1536840000000, 6453, 6473, 6425, 6442, 211313668],
         [1536843600000, 6442, 6530, 6436, 6493.5, 212000228],
         [1536847200000, 6493.5, 6508.5, 6476, 6480.5, 138883617],
         [1536850800000, 6480.5, 6484, 6464.5, 6477, 70677760], [1536854400000, 6477, 6515, 6476.5, 6496, 127570051],
         [1536858000000, 6496, 6509.5, 6474.5, 6496, 94295065], [1536861600000, 6496, 6503.5, 6475, 6482, 57737315],
         [1536865200000, 6482, 6488, 6419.5, 6446.5, 124387203],
         [1536868800000, 6446.5, 6469, 6411.5, 6461, 103949558],
         [1536872400000, 6461, 6463, 6448, 6454.5, 30459362], [1536876000000, 6454.5, 6499, 6454, 6475, 63770246],
         [1536879600000, 6475, 6509.5, 6475, 6487.5, 99678700], [1536883200000, 6487.5, 6494, 6454.5, 6459, 75399665],
         [1536886800000, 6459, 6489, 6459, 6481, 38478732], [1536890400000, 6481, 6577, 6480.5, 6543, 238504302],
         [1536894000000, 6543, 6585, 6541, 6546.5, 146867770],
         [1536897600000, 6546.5, 6555.5, 6535.5, 6542.5, 74456882],
         [1536901200000, 6542.5, 6560, 6542, 6547, 43544678], [1536904800000, 6547, 6552, 6537, 6552, 38801990],
         [1536908400000, 6552, 6580, 6545, 6545, 89766600], [1536912000000, 6545, 6553, 6350, 6416.5, 335447267],
         [1536915600000, 6416.5, 6456, 6401.5, 6422, 218328860], [1536919200000, 6422, 6459.5, 6421, 6437, 87610016],
         [1536922800000, 6437, 6485, 6436, 6458.5, 91544677], [1536926400000, 6458.5, 6468, 6442, 6466, 74343440],
         [1536930000000, 6466, 6466, 6420, 6443, 101769717], [1536933600000, 6443, 6450, 6425.5, 6446.5, 62852292],
         [1536937200000, 6446.5, 6478.5, 6446.5, 6473, 77423472], [1536940800000, 6473, 6510, 6473, 6503, 160769888],
         [1536944400000, 6503, 6503, 6460, 6460, 85027913], [1536948000000, 6460, 6477.5, 6455, 6470.5, 43780266],
         [1536951600000, 6470.5, 6538, 6470.5, 6528.5, 95695426], [1536955200000, 6528.5, 6540, 6512, 6530, 76017453],
         [1536958800000, 6530, 6536, 6497, 6507.5, 44901054], [1536962400000, 6507.5, 6534.5, 6494, 6511.5, 65127013],
         [1536966000000, 6511.5, 6523.5, 6450.5, 6478, 85516895],
         [1536969600000, 6478, 6493.5, 6464.5, 6479.5, 41718762],
         [1536973200000, 6479.5, 6519, 6478.5, 6513.5, 43223701], [1536976800000, 6513.5, 6514, 6492, 6492, 28602251],
         [1536980400000, 6492, 6498, 6474.5, 6491.5, 36091887], [1536984000000, 6491.5, 6507, 6486.5, 6505, 33248313],
         [1536987600000, 6505, 6507.5, 6478, 6489.5, 39760663],
         [1536991200000, 6489.5, 6489.5, 6463.5, 6479.5, 48365781],
         [1536994800000, 6479.5, 6499, 6479.5, 6492.5, 32727365],
         [1536998400000, 6492.5, 6533, 6492.5, 6511, 88236362],
         [1537002000000, 6511, 6527.5, 6510.5, 6525.5, 29284042],
         [1537005600000, 6525.5, 6550, 6510, 6517.5, 61030360],
         [1537009200000, 6517.5, 6524, 6495.5, 6516.5, 60058126],
         [1537012800000, 6516.5, 6531, 6508.5, 6524.5, 55897129],
         [1537016400000, 6524.5, 6562.5, 6524, 6536, 133611876], [1537020000000, 6536, 6568, 6526, 6534.5, 80263400],
         [1537023600000, 6534.5, 6538.5, 6513, 6532, 47260872], [1537027200000, 6532, 6551, 6529, 6545, 36023583],
         [1537030800000, 6545, 6556.5, 6523.5, 6545.5, 46867999],
         [1537034400000, 6545.5, 6545.5, 6505.5, 6518, 61702035],
         [1537038000000, 6518, 6532, 6506, 6530, 43668034], [1537041600000, 6530, 6537, 6517.5, 6531, 31219267],
         [1537045200000, 6531, 6531.5, 6465, 6499, 86028051], [1537048800000, 6499, 6524.5, 6499, 6515, 28874263],
         [1537052400000, 6515, 6525.5, 6514.5, 6515.5, 21250886],
         [1537056000000, 6515.5, 6516.5, 6496.5, 6508, 32021973],
         [1537059600000, 6508, 6515.5, 6484, 6493.5, 42764717], [1537063200000, 6493.5, 6503, 6476, 6497, 64028158],
         [1537066800000, 6497, 6499, 6482.5, 6488.5, 30073752],
         [1537070400000, 6488.5, 6489, 6330.5, 6447.5, 239263697],
         [1537074000000, 6447.5, 6475, 6444, 6465, 89900506], [1537077600000, 6465, 6465, 6449, 6454.5, 32601161],
         [1537081200000, 6454.5, 6455.5, 6442.5, 6444.5, 30800561],
         [1537084800000, 6444.5, 6500, 6443, 6479, 95124812],
         [1537088400000, 6479, 6488, 6471, 6485.5, 30391395], [1537092000000, 6485.5, 6516, 6478.5, 6505, 55032328],
         [1537095600000, 6505, 6506, 6487.5, 6493.5, 43350243], [1537099200000, 6493.5, 6498, 6481, 6495, 33488292],
         [1537102800000, 6495, 6497, 6477, 6481, 34917228], [1537106400000, 6481, 6481.5, 6457.5, 6468.5, 89523864],
         [1537110000000, 6468.5, 6472.5, 6461, 6472, 30436874], [1537113600000, 6472, 6484.5, 6472, 6478, 38990066],
         [1537117200000, 6478, 6481.5, 6401, 6466.5, 100211543], [1537120800000, 6466.5, 6490, 6466.5, 6483, 45457245],
         [1537124400000, 6483, 6509.5, 6482.5, 6497, 41837960], [1537128000000, 6497, 6503.5, 6492, 6493, 22450985],
         [1537131600000, 6493, 6497.5, 6481.5, 6484, 23621244], [1537135200000, 6484, 6490, 6472, 6490, 25519945],
         [1537138800000, 6490, 6496.5, 6487, 6493, 13932918], [1537142400000, 6493, 6528.5, 6492.5, 6519, 89161308],
         [1537146000000, 6519, 6521, 6511.5, 6514.5, 18322305], [1537149600000, 6514.5, 6515, 6505, 6512, 19508104],
         [1537153200000, 6512, 6512, 6502, 6509, 18335482], [1537156800000, 6509, 6511.5, 6453, 6470.5, 138235083],
         [1537160400000, 6470.5, 6482, 6460, 6478.5, 58472513],
         [1537164000000, 6478.5, 6483.5, 6469.5, 6478, 25893239],
         [1537167600000, 6478, 6478.5, 6462.5, 6474, 47895881], [1537171200000, 6474, 6474.5, 6464, 6473, 35718539],
         [1537174800000, 6473, 6509, 6405, 6485.5, 185343846], [1537178400000, 6485.5, 6492, 6467.5, 6469, 47459471],
         [1537182000000, 6469, 6478, 6462, 6469.5, 32729365], [1537185600000, 6469.5, 6469.5, 6425.5, 6442, 176887365],
         [1537189200000, 6442, 6451, 6365, 6420.5, 150210041], [1537192800000, 6420.5, 6420.5, 6321, 6348, 357167654],
         [1537196400000, 6348, 6348, 6224, 6256.5, 424043399], [1537200000000, 6256.5, 6285, 6250, 6252, 132562390],
         [1537203600000, 6252, 6284, 6251.5, 6283.5, 64052790],
         [1537207200000, 6283.5, 6306.5, 6270, 6289.5, 139313594],
         [1537210800000, 6289.5, 6296.5, 6253, 6264.5, 62129457],
         [1537214400000, 6264.5, 6271, 6201, 6213.5, 199500836],
         [1537218000000, 6213.5, 6253, 6211.5, 6239.5, 74156356], [1537221600000, 6239.5, 6257, 6216, 6242, 57746528],
         [1537225200000, 6242, 6266.5, 6241.5, 6247, 50593659], [1537228800000, 6247, 6248, 6226.5, 6242.5, 51662547],
         [1537232400000, 6242.5, 6248, 6239, 6246.5, 19657291], [1537236000000, 6246.5, 6247, 6230, 6240.5, 40586652],
         [1537239600000, 6240.5, 6246, 6230, 6232.5, 39690561], [1537243200000, 6232.5, 6261, 6232.5, 6256, 51760433],
         [1537246800000, 6256, 6275, 6253.5, 6264.5, 50308077],
         [1537250400000, 6264.5, 6269.5, 6258.5, 6258.5, 31525977],
         [1537254000000, 6258.5, 6274, 6258.5, 6273.5, 39168690],
         [1537257600000, 6273.5, 6280.5, 6251, 6253, 62897855],
         [1537261200000, 6253, 6253, 6236, 6252, 57318875], [1537264800000, 6252, 6256.5, 6236, 6242, 38256435],
         [1537268400000, 6242, 6287, 6233.5, 6280, 84141669], [1537272000000, 6280, 6329, 6263, 6311.5, 248527055],
         [1537275600000, 6311.5, 6375, 6311, 6360.5, 280438524], [1537279200000, 6360.5, 6365, 6343.5, 6356, 83457400],
         [1537282800000, 6356, 6380.5, 6332.5, 6353, 130696810],
         [1537286400000, 6353, 6358.5, 6343.5, 6344.5, 47396659],
         [1537290000000, 6344.5, 6344.5, 6310, 6323, 116614475], [1537293600000, 6323, 6337.5, 6318, 6333, 48196126],
         [1537297200000, 6333, 6333.5, 6285, 6304, 55956218], [1537300800000, 6304, 6334.5, 6284, 6317, 62903119],
         [1537304400000, 6317, 6325, 6311.5, 6318, 24762925], [1537308000000, 6318, 6354.5, 6317.5, 6343.5, 72082077],
         [1537311600000, 6343.5, 6347, 6328, 6328.5, 33966826], [1537315200000, 6328.5, 6329, 6300, 6312, 74362935],
         [1537318800000, 6312, 6338.5, 6311.5, 6329.5, 46571340],
         [1537322400000, 6329.5, 6334.5, 6321.5, 6333, 27925048],
         [1537326000000, 6333, 6360, 6322, 6346, 94682862], [1537329600000, 6346, 6346.5, 6320, 6329, 46902474],
         [1537333200000, 6329, 6344, 6328.5, 6336.5, 25353458],
         [1537336800000, 6336.5, 6350, 6335.5, 6343.5, 30490647],
         [1537340400000, 6343.5, 6349, 6320.5, 6348, 55643064], [1537344000000, 6348, 6356, 6332.5, 6338, 49729217],
         [1537347600000, 6338, 6338, 6285, 6316.5, 130353466], [1537351200000, 6316.5, 6326, 6314, 6314, 28016047],
         [1537354800000, 6314, 6319, 6257, 6282, 131394417], [1537358400000, 6282, 6291.5, 6250, 6284.5, 162270081],
         [1537362000000, 6284.5, 6293.5, 6276.5, 6281.5, 63132157],
         [1537365600000, 6281.5, 6348, 6281.5, 6322.5, 170685496],
         [1537369200000, 6322.5, 6335.5, 6321, 6325, 59171980],
         [1537372800000, 6325, 6337, 6316, 6323, 49031522], [1537376400000, 6323, 6330, 6317, 6319.5, 34998269],
         [1537380000000, 6319.5, 6320.5, 6066, 6115, 448921973], [1537383600000, 6115, 6580, 6090, 6402, 730797561],
         [1537387200000, 6402, 6450, 6395, 6415.5, 134314469],
         [1537390800000, 6415.5, 6442.5, 6375.5, 6392, 111290218],
         [1537394400000, 6392, 6395, 6365.5, 6382, 73103529], [1537398000000, 6382, 6385.5, 6372, 6384.5, 42706696],
         [1537401600000, 6384.5, 6417, 6384, 6411.5, 68692711],
         [1537405200000, 6411.5, 6417.5, 6399.5, 6402.5, 39079263],
         [1537408800000, 6402.5, 6403, 6382.5, 6386, 36552636], [1537412400000, 6386, 6404.5, 6384, 6398, 34941415],
         [1537416000000, 6398, 6401, 6388, 6388.5, 25596650], [1537419600000, 6388.5, 6393.5, 6387, 6390, 16595347],
         [1537423200000, 6390, 6409, 6389.5, 6398.5, 35116568], [1537426800000, 6398.5, 6405, 6396, 6402.5, 23417359],
         [1537430400000, 6402.5, 6402.5, 6376, 6385.5, 60769853], [1537434000000, 6385.5, 6390, 6380, 6387, 26854300],
         [1537437600000, 6387, 6407.5, 6383.5, 6407, 33488926], [1537441200000, 6407, 6435, 6407, 6421, 110849909],
         [1537444800000, 6421, 6445, 6412.5, 6413.5, 87782787], [1537448400000, 6413.5, 6418.5, 6387, 6418, 80079114],
         [1537452000000, 6418, 6419.5, 6409, 6411.5, 34052338], [1537455600000, 6411.5, 6421.5, 6397, 6410, 52843111],
         [1537459200000, 6410, 6420, 6405.5, 6413.5, 33478748], [1537462800000, 6413.5, 6418, 6391, 6398, 60609942],
         [1537466400000, 6398, 6402, 6387, 6401.5, 36634533], [1537470000000, 6401.5, 6432, 6401.5, 6415.5, 52976974],
         [1537473600000, 6415.5, 6430, 6336.5, 6377, 186959269], [1537477200000, 6377, 6496, 6352.5, 6482, 186832895],
         [1537480800000, 6482, 6520, 6455, 6503.5, 181827687], [1537484400000, 6503.5, 6518.5, 6475, 6482, 109993135],
         [1537488000000, 6482, 6537, 6481.5, 6495.5, 168334058],
         [1537491600000, 6495.5, 6499.5, 6488.5, 6496.5, 35821521],
         [1537495200000, 6496.5, 6558, 6495, 6538.5, 108753543], [1537498800000, 6538.5, 6555.5, 6528, 6544, 70884802],
         [1537502400000, 6544, 6548, 6530.5, 6537, 33385717], [1537506000000, 6537, 6537.5, 6510, 6526, 43707209],
         [1537509600000, 6526, 6547, 6525.5, 6539, 37290486], [1537513200000, 6539, 6707, 6536.5, 6640.5, 267013813],
         [1537516800000, 6640.5, 6748.5, 6640, 6704, 423210724], [1537520400000, 6704, 6720, 6680.5, 6710, 105930379],
         [1537524000000, 6710, 6716, 6692.5, 6696, 65023483], [1537527600000, 6696, 6778, 6696, 6739, 149409168],
         [1537531200000, 6739, 6739.5, 6710, 6726.5, 95121039],
         [1537534800000, 6726.5, 6769.5, 6723, 6727.5, 113697430],
         [1537538400000, 6727.5, 6737.5, 6701.5, 6722, 127836736],
         [1537542000000, 6722, 6722.5, 6680, 6684, 136686620],
         [1537545600000, 6684, 6729.5, 6683.5, 6717, 85216069], [1537549200000, 6717, 6717, 6693, 6695.5, 42845366],
         [1537552800000, 6695.5, 6742.5, 6689.5, 6734, 55673134],
         [1537556400000, 6734, 6773.5, 6725.5, 6747.5, 105391148],
         [1537560000000, 6747.5, 6777.5, 6700, 6700, 113308568], [1537563600000, 6700, 6739, 6690, 6714, 77893996],
         [1537567200000, 6714, 6730, 6713.5, 6730, 25928594], [1537570800000, 6730, 6753.5, 6724, 6746.5, 47876039],
         [1537574400000, 6746.5, 6830, 6746.5, 6787, 193917106], [1537578000000, 6787, 6788, 6736, 6754, 96910349],
         [1537581600000, 6754, 6759.5, 6730, 6730.5, 34957954], [1537585200000, 6730.5, 6744, 6707.5, 6730, 71709321],
         [1537588800000, 6730, 6747, 6729.5, 6741.5, 34094130], [1537592400000, 6741.5, 6742, 6721, 6730.5, 21924504],
         [1537596000000, 6730.5, 6730.5, 6696, 6718.5, 65529608],
         [1537599600000, 6718.5, 6719, 6711, 6717.5, 30511235],
         [1537603200000, 6717.5, 6717.5, 6626, 6645, 237688145], [1537606800000, 6645, 6693.5, 6638, 6688, 78653063],
         [1537610400000, 6688, 6713, 6683, 6688, 77284340], [1537614000000, 6688, 6696.5, 6658.5, 6659, 66177658],
         [1537617600000, 6659, 6697, 6658, 6673, 57626296], [1537621200000, 6673, 6696.5, 6665.5, 6684, 40790064],
         [1537624800000, 6684, 6684.5, 6653, 6664, 58242691], [1537628400000, 6664, 6679.5, 6646, 6675, 60734654],
         [1537632000000, 6675, 6675.5, 6660.5, 6662.5, 25114823],
         [1537635600000, 6662.5, 6674.5, 6662.5, 6674.5, 24272207],
         [1537639200000, 6674.5, 6746, 6634.5, 6726, 185511706], [1537642800000, 6726, 6726, 6707, 6714, 30997854],
         [1537646400000, 6714, 6721, 6706, 6708, 25186683], [1537650000000, 6708, 6708.5, 6677, 6691.5, 44019240],
         [1537653600000, 6691.5, 6702, 6683, 6698, 22268952], [1537657200000, 6698, 6717, 6688, 6709, 26725006],
         [1537660800000, 6709, 6717.5, 6694.5, 6694.5, 31637743],
         [1537664400000, 6694.5, 6696.5, 6676, 6683.5, 30745690],
         [1537668000000, 6683.5, 6698, 6683, 6691, 20294343], [1537671600000, 6691, 6697.5, 6690, 6697.5, 16215949],
         [1537675200000, 6697.5, 6720.5, 6697, 6715, 41625749], [1537678800000, 6715, 6723, 6708, 6713, 24315563],
         [1537682400000, 6713, 6713.5, 6696.5, 6696.5, 21231577],
         [1537686000000, 6696.5, 6708.5, 6696.5, 6708, 13532300],
         [1537689600000, 6708, 6730, 6707.5, 6719, 31243505], [1537693200000, 6719, 6779, 6719, 6758, 134776654],
         [1537696800000, 6758, 6758, 6736.5, 6750.5, 46566939], [1537700400000, 6750.5, 6757, 6737.5, 6741, 37330150],
         [1537704000000, 6741, 6741, 6720, 6727.5, 63084053], [1537707600000, 6727.5, 6729, 6720.5, 6724.5, 26046071],
         [1537711200000, 6724.5, 6733, 6653, 6680, 128515432], [1537714800000, 6680, 6692, 6656, 6678.5, 100179192],
         [1537718400000, 6678.5, 6691.5, 6667.5, 6685, 40619254], [1537722000000, 6685, 6690.5, 6676, 6679, 26694394],
         [1537725600000, 6679, 6698, 6672, 6698, 33318949], [1537729200000, 6698, 6698, 6677.5, 6679.5, 24714359],
         [1537732800000, 6679.5, 6687.5, 6661, 6675, 27915242], [1537736400000, 6675, 6682.5, 6666, 6682.5, 26519664],
         [1537740000000, 6682.5, 6719, 6677, 6714, 50860448], [1537743600000, 6714, 6713.5, 6702.5, 6703, 19097360],
         [1537747200000, 6703, 6703.5, 6686, 6693, 28109478], [1537750800000, 6693, 6720, 6692.5, 6713.5, 36126034],
         [1537754400000, 6713.5, 6714, 6699, 6702, 26306528], [1537758000000, 6702, 6708.5, 6697.5, 6706.5, 17918825],
         [1537761600000, 6706.5, 6706.5, 6681.5, 6696, 35025367], [1537765200000, 6696, 6696, 6673, 6686, 46188434],
         [1537768800000, 6686, 6686.5, 6612, 6646, 124495435], [1537772400000, 6646, 6679, 6636.5, 6672.5, 85466679],
         [1537776000000, 6672.5, 6675, 6632, 6641.5, 69591281],
         [1537779600000, 6641.5, 6642, 6577.5, 6605.5, 200478630],
         [1537783200000, 6605.5, 6620, 6596.5, 6612.5, 86657090],
         [1537786800000, 6612.5, 6613, 6577.5, 6582, 114790855],
         [1537790400000, 6582, 6637, 6579.5, 6620.5, 115807873], [1537794000000, 6620.5, 6621, 6607, 6617.5, 58390149],
         [1537797600000, 6617.5, 6620.5, 6592, 6607, 60646008], [1537801200000, 6607, 6613, 6595, 6600.5, 38825955],
         [1537804800000, 6600.5, 6618.5, 6596, 6610, 64892454], [1537808400000, 6610, 6639, 6609.5, 6637.5, 75686634],
         [1537812000000, 6637.5, 6654, 6636, 6637.5, 59262879],
         [1537815600000, 6637.5, 6639, 6621.5, 6621.5, 33734299],
         [1537819200000, 6621.5, 6629, 6615, 6622, 33851245], [1537822800000, 6622, 6625, 6613.5, 6615.5, 21796272],
         [1537826400000, 6615.5, 6616, 6580, 6592.5, 87138596],
         [1537830000000, 6592.5, 6592.5, 6555, 6578.5, 128821148],
         [1537833600000, 6578.5, 6578.5, 6504, 6513.5, 202481675],
         [1537837200000, 6513.5, 6514, 6402, 6423.5, 286972664],
         [1537840800000, 6423.5, 6428.5, 6382, 6418, 177249001],
         [1537844400000, 6418, 6486.5, 6418, 6457.5, 195083698],
         [1537848000000, 6457.5, 6474, 6455.5, 6455.5, 57272048],
         [1537851600000, 6455.5, 6464.5, 6448, 6461, 34717800],
         [1537855200000, 6461, 6462, 6408.5, 6418.5, 87872360], [1537858800000, 6418.5, 6437, 6357, 6375.5, 182715366],
         [1537862400000, 6375.5, 6414, 6361.5, 6404, 103111329], [1537866000000, 6404, 6417, 6391, 6406, 73757880],
         [1537869600000, 6406, 6408.5, 6389.5, 6405, 55905866], [1537873200000, 6405, 6405, 6372, 6392, 83300095],
         [1537876800000, 6392, 6448, 6381, 6437, 142293713], [1537880400000, 6437, 6437.5, 6407, 6411.5, 66858100],
         [1537884000000, 6411.5, 6422, 6378, 6415, 99914533], [1537887600000, 6415, 6415.5, 6330, 6400.5, 169129541],
         [1537891200000, 6400.5, 6415, 6385, 6385, 65450550], [1537894800000, 6385, 6400, 6358, 6387.5, 65230531],
         [1537898400000, 6387.5, 6390.5, 6311.5, 6347, 226089388], [1537902000000, 6347, 6375, 6336, 6358, 68913747],
         [1537905600000, 6358, 6406, 6358, 6382, 91177350], [1537909200000, 6382, 6394, 6370, 6386, 41867889],
         [1537912800000, 6386, 6399, 6374, 6388.5, 53989272], [1537916400000, 6388.5, 6433.5, 6385, 6429, 94113374],
         [1537920000000, 6429, 6429, 6397.5, 6402.5, 51527501],
         [1537923600000, 6402.5, 6404.5, 6381, 6391.5, 45327992],
         [1537927200000, 6391.5, 6398.5, 6390, 6390.5, 21933615], [1537930800000, 6390.5, 6393, 6377, 6378, 29959219],
         [1537934400000, 6378, 6393, 6377.5, 6389.5, 24610897], [1537938000000, 6389.5, 6459, 6389, 6433.5, 186372647],
         [1537941600000, 6433.5, 6440, 6419, 6424.5, 47541144],
         [1537945200000, 6424.5, 6453, 6424.5, 6446.5, 60068846],
         [1537948800000, 6446.5, 6446.5, 6428, 6442, 44329760],
         [1537952400000, 6442, 6458.5, 6429.5, 6446.5, 56664251],
         [1537956000000, 6446.5, 6469, 6446, 6455.5, 76986885],
         [1537959600000, 6455.5, 6519.5, 6454, 6510.5, 209168374],
         [1537963200000, 6510.5, 6528, 6483.5, 6483.5, 150463394],
         [1537966800000, 6483.5, 6498, 6465.5, 6481, 94330712],
         [1537970400000, 6481, 6527.5, 6480.5, 6511.5, 87886883],
         [1537974000000, 6511.5, 6540, 6500.5, 6522, 114040720],
         [1537977600000, 6522, 6527.5, 6503.5, 6516, 64938030],
         [1537981200000, 6516, 6529.5, 6512.5, 6521.5, 38179897],
         [1537984800000, 6521.5, 6523.5, 6504.5, 6510, 41395370],
         [1537988400000, 6510, 6510.5, 6478.5, 6485.5, 82499550],
         [1537992000000, 6485.5, 6502, 6481, 6499.5, 32612907], [1537995600000, 6499.5, 6500, 6443, 6465.5, 105174253],
         [1537999200000, 6465.5, 6465.5, 6425, 6459, 83511994], [1538002800000, 6459, 6475, 6451, 6452.5, 53962224],
         [1538006400000, 6452.5, 6494.5, 6452, 6489.5, 72394239],
         [1538010000000, 6489.5, 6510.5, 6485, 6485, 67601910],
         [1538013600000, 6485, 6497, 6476, 6495.5, 44429835], [1538017200000, 6495.5, 6501, 6486, 6495.5, 33487611],
         [1538020800000, 6495.5, 6503.5, 6464, 6474.5, 58317897],
         [1538024400000, 6474.5, 6478.5, 6455, 6460.5, 33325486],
         [1538028000000, 6460.5, 6465.5, 6444, 6462, 62197830]]

    fenxing_type, fenxing_time, fenxing_plot, fenxing_data = k_analysis(k)
    print(len(fenxing_type), fenxing_type)
    print('*' * 50)
    print(len(fenxing_time), fenxing_time)
    print('*' * 50)
    print(len(fenxing_plot), fenxing_plot)
    print('*' * 50)
    print(len(fenxing_data), fenxing_data[['date', 'timestamp', 'open', 'close', 'high', 'low']])
    print('*' * 50)



