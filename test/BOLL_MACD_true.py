from tools import boll_macd_ga_tools
from tools import data2df, public_tools
import pandas as pd
import time
from tools.stockstats import StockDataFrame

# 模拟算法实盘回测

if __name__ == '__main__':
    now = time.time()
    pop_size, chromosome_length = 20, 6
    pops = boll_macd_ga_tools.init_pops(pop_size, chromosome_length)
    iter = 5  # TODO 迭代次数
    pc = 0.6  # 杂交概率
    pm = 0.01  # 变异概率
    results = []  # 存储每一代的最优解，N个二元组
    filename = 'BTC2018-10-15-now-1H'
    ori_df = data2df.csv2df(filename + '.csv')
    ori_df = ori_df.astype(float)
    ori_df['Timestamp'] = ori_df['Timestamp'].astype(int)
    ori_df['date'] = pd.to_datetime(ori_df['Timestamp'], unit='s')
    ori_df.index = ori_df.date
    re_max_drawdown, re_total_yield = [], []

    i = 1000
    while i < len(ori_df):
        df = ori_df[i - 500: i - 50]
        best_individuals = []
        best_fits = []
        for j in range(iter):
            obj_values = boll_macd_ga_tools.cal_fitness(df, pops, pop_size, chromosome_length)  # 计算绩效
            fit_values = boll_macd_ga_tools.clear_fit_values(obj_values)
            best_individual, best_fit = boll_macd_ga_tools.find_best(pops, fit_values,
                                                                     chromosome_length)  # 第一个是最优基因序列, 第二个是对应的最佳个体适度
            best_individuals.append(best_individual)
            best_fits.append(best_fit)
            results.append([best_individual, best_fit])
            boll_macd_ga_tools.selection(pops, fit_values)  # 选择
            boll_macd_ga_tools.crossover(pops, pc)  # 染色体交叉（最优个体之间进行0、1互换）
            boll_macd_ga_tools.mutation(pops, pm)  # 染色体变异（其实就是随机进行0、1取反
        best_fit = min(best_fits)
        best_fit_idx = best_fits.index(best_fit)
        best_individual = best_individuals[best_fit_idx]
        df = ori_df[i - 500: i]
        stock = StockDataFrame.retype(df)
        df['signal'] = 'wait'
        max_drawdown, total_yield = boll_macd_ga_tools.cal_someone_fitness(df, stock, best_individual[0],
                                                                           best_individual[1], best_individual[2],
                                                                           best_individual[3],
                                                                           best_individual[4], best_individual[5],
                                                                           best_individual[6], best_individual[7],
                                                                           best_individual[8],
                                                                           best_individual[9], best_individual[10],
                                                                           best_individual[11])
        re_max_drawdown.append(max_drawdown)
        re_total_yield.append(total_yield)
        i += 10
        print(public_tools.get_time(), i)

    print(re_max_drawdown)
    print(re_total_yield)
