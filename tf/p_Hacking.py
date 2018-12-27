#！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/12/25 21:33

@desc: https://mp.weixin.qq.com/s?__biz=MzAxNTc0Mjg0Mg==&mid=2653289820&idx=1&sn=d3fee74ba1daab837433e4ef6b0ab4d9&chksm=802e3f49b759b65f422d20515942d5813aead73231da7d78e9f235bdb42386cf656079e69b8b&scene=0&xtrack=1#rd

'''

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tools import data2df

stock = data2df.csv2df('BTC2017-09-01-now-1D.csv')
stock = stock.astype(float)
cutoff = len(stock)//2
prices = pd.Series(stock.Close)
log_prices = np.log(prices)
deltas = pd.Series(np.diff(prices), index=stock.index[1:])
log_deltas = pd.Series(np.diff(log_prices), index=stock.index[1:])
latest_prices = stock.Close[cutoff:]
latest_log_prices = np.log(latest_prices)
latest_log_deltas = deltas[cutoff:]
prior_log_deltas = log_deltas[:cutoff]
prior_log_mean = np.mean(prior_log_deltas)
prior_log_std = np.std(prior_log_deltas)
# f, axes = plt.subplots(ncols=2, figsize=(15,5))
# prices.plot(ax=axes[0])
# deltas.hist(bins=50, ax=axes[1])
# f.autofmt_xdate()
# f.tight_layout()
# plt.show()


def predict(mean, std, size, seed=None):
    np.random.seed(seed)
    return np.random.normal(loc=mean, scale=std, size=size)

def apply_returns(start, returns):
    cur = start
    prices = []
    for r in returns:
        cur += r
        prices.append(cur)
    return prices

def score(actual, prediction):
    return np.mean(np.square(actual - prediction))

def compare(prediction, actual):
    plt.plot(prediction, label="prediction")
    plt.plot(actual, label="actual")
    plt.legend()

predict_deltas = predict(prior_log_mean, prior_log_std, len(latest_prices), seed=0)
start = latest_log_prices.iloc[0]
# prediction = apply_returns(start, predict_deltas)
# print("MSE: {:0.08f}".format(score(latest_log_prices, prediction)))
# compare(prediction=prediction, actual=latest_log_prices.values)
# plt.show()

predict_partial = lambda s: predict(mean=prior_log_mean, std = prior_log_std, size=len(latest_prices), seed=s)
def find_best_seed(actual, predict_partial, score, start_seed, end_seed):
    best_so_far = None
    best_score = float("inf")
    start = actual.iloc[0]
    for s in range(start_seed, end_seed):
        print('\r{} / {}'.format(s, end_seed), end="")
        predict_deltas = predict_partial(s)
        predict_prices = apply_returns(start, predict_deltas)
        predict_score = score(actual, predict_prices)
        if predict_score < best_score:
            best_score = predict_score
            best_so_far = s
    return best_so_far, best_score

best_seed, best_score = find_best_seed(latest_log_prices, predict_partial, score, start_seed=0, end_seed=500000)
print("best seed: {} best MSE: {:0.08f}".format(best_seed, best_score))


returns = predict(mean=prior_log_mean, std=prior_log_std, size=400, seed=best_seed)
prediction = apply_returns(start, returns)
# compare(prediction, latest_log_prices.values)
compare(prediction, log_prices.values)
plt.show()