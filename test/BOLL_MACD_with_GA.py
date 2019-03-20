from tools import boll_macd_ga_tools
from tools import data2df
import pandas as pd
import time

if __name__ == '__main__':
    now = time.time()
    pop_size, chromosome_length = 50, 6
    pops = boll_macd_ga_tools.init_pops(pop_size, chromosome_length)
    iter = 5  # TODO 迭代次数
    pc = 0.6  # 杂交概率
    pm = 0.01  # 变异概率
    results = []  # 存储每一代的最优解，N个二元组
    filename = 'BTC2018-10-15-now-1H'
    df = data2df.csv2df(filename + '.csv')
    df = df.astype(float)
    df['Timestamp'] = df['Timestamp'].astype(int)
    df['date'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.index = df.date
    best_individuals = []
    best_fits = []
    for i in range(iter):
        obj_values = boll_macd_ga_tools.cal_fitness(df, pops, pop_size, chromosome_length)  # 计算绩效
        print(obj_values)
        fit_values = boll_macd_ga_tools.clear_fit_values(obj_values)
        best_individual, best_fit = boll_macd_ga_tools.find_best(pops, fit_values, chromosome_length)  # 第一个是最优基因序列, 第二个是对应的最佳个体适度
        best_individuals.append(best_individual)
        best_fits.append(best_fit)
        results.append([best_individual, best_fit])
        boll_macd_ga_tools.selection(pops, fit_values)  # 选择
        boll_macd_ga_tools.crossover(pops, pc)  # 染色体交叉（最优个体之间进行0、1互换）
        boll_macd_ga_tools.mutation(pops, pm)  # 染色体变异（其实就是随机进行0、1取反）

    print(results)
    print(best_individuals)
    print(best_fits)
    print(time.time() - now)
