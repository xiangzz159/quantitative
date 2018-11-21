# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/11/16 15:42

@desc:

'''

from tools import data2df, public_tools
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

df = data2df.csv2df('BTC2014-01-01-now-30M.csv')
df = df.astype(float)
df['Timestamp'] = df['Timestamp'].astype(int)

df['Date'] = df['Timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values('Date')

# k线图显示
# plt.figure(figsize=(18, 9))
# plt.plot(range(df.shape[0]), (df['Low'] + df['High']) / 2.0)
# plt.xticks(range(0, df.shape[0], 1500), df['Date'].loc[::1500], rotation=45)
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Mid Price', fontsize=18)
# plt.show()

# 计算最高和最低价的平均值来计算中间价格
high_prices = df.loc[:, 'High'].values
low_prices = df.loc[:, 'Low'].values
mid_prices = (high_prices + low_prices) / 2.0

# 分离训练数据和测试数据
data_count = len(mid_prices)
train_count = int(data_count * 0.95)
train_data = mid_prices[:train_count]
test_data = mid_prices[train_count:]

scaler = MinMaxScaler()
train_data = train_data.reshape(-1, 1)
test_data = test_data.reshape(-1, 1)

smoothing_window_size = 2500
for di in range(0, train_count - smoothing_window_size, smoothing_window_size):
    scaler.fit(train_data[di:di + smoothing_window_size, :])
    train_data[di:di + smoothing_window_size, :] = scaler.transform(train_data[di:di + smoothing_window_size, :])

# You normalize the last bit of remining data
scaler.fit(train_data[di + smoothing_window_size:, :])
train_data[di + smoothing_window_size:, :] = scaler.transform(train_data[di + smoothing_window_size:, :])

# 将数据重新塑造为[data_size]的Shape
train_data = train_data.reshape(-1)
test_data = scaler.transform(test_data).reshape(-1)

all_mid_data_ = np.concatenate([train_data, test_data], axis=0)
# 平滑训练数据1
# EMA = 0.0
# ganma = 0.1
# for ti in range(train_count):
#     EMA = ganma * train_data[ti] + (1 - ganma) * EMA
#     train_data[ti] = EMA
# all_mid_data = np.concatenate([train_data, test_data], axis=0)
# plt.figure(figsize=(20, 10))
# plt.plot(all_mid_data_, label='True', color='yellow')
# plt.plot(all_mid_data, label='Prediction', color='blue')
# plt.show()

# 平滑训练数据2
# window_size = 100
# N = train_data.size
#
# run_avg_predictions = []
# run_avg_x = []
#
# mse_errors = []
#
# running_mean = 0.0
# run_avg_predictions.append(running_mean)
#
# decay = 0.5
#
# for pred_idx in range(1, N):
#     running_mean = running_mean * decay + (1.0 - decay) * train_data[pred_idx - 1]
#     run_avg_predictions.append(running_mean)
#     mse_errors.append(run_avg_predictions[-1] - train_data[pred_idx] ** 2)
#     run_avg_x.append(pred_idx)

# LSTM
# 定义超参数
D = 1
num_unrollings = 50
batch_size = 500
num_nodes = [200, 200, 150]
n_layers = len(num_nodes)
dropout = 0.2

tf.reset_default_graph()

# 定义输入和输出
train_inputs, train_outputs = [], []
for ui in range(num_unrollings):
    train_inputs.append(tf.placeholder(tf.float32, shape=[batch_size, D], name='train_inputs_%d' % ui))
    train_outputs.append(tf.placeholder(tf.float32, shape=[batch_size, 1], name='train_outputs_%d' % ui))

# 定义LSTM和回归层参数, 用w和b表示
lstm_cells = [
    tf.contrib.rnn.LSTMCell(num_units=num_nodes[li],
                            state_is_tuple=True,
                            initializer=tf.contrib.layers.xavier_initializer()
                            )
    for li in range(n_layers)]

drop_lstm_cells = [tf.contrib.rnn.DropoutWrapper(
    lstm, input_keep_prob=1.0, output_keep_prob=1.0 - dropout, state_keep_prob=1.0 - dropout
) for lstm in lstm_cells]
drop_multi_cell = tf.contrib.rnn.MultiRNNCell(drop_lstm_cells)
multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)

w = tf.get_variable('w', shape=[num_nodes[-1], 1], initializer=tf.contrib.layers.xavier_initializer())
b = tf.get_variable('b', initializer=tf.random_uniform([1], -0.1, 0.1))

# 计算LSTM输出并将其输入回归层， 得到最终预测结果
c, h = [], []
initial_state = []
for li in range(n_layers):
    c.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
    h.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
    initial_state.append(tf.contrib.rnn.LSTMStateTuple(c[li], h[li]))

all_inputs = tf.concat([tf.expand_dims(t, 0) for t in train_inputs], axis=0)
all_lstm_outputs, state = tf.nn.dynamic_rnn(
    drop_multi_cell, all_inputs, initial_state=tuple(initial_state),
    time_major=True, dtype=tf.float32
)
all_lstm_outputs = tf.reshape(all_lstm_outputs, [batch_size * num_unrollings, num_nodes[-1]])
all_outputs = tf.nn.xw_plus_b(all_lstm_outputs, w, b)
split_outputs = tf.split(all_outputs, num_unrollings, axis=0)

# 损失计算和优化器
print('Defining training Loss')
loss = 0.0
with tf.control_dependencies([tf.assign(c[li], state[li][0]) for li in range(n_layers)] +
                             [tf.assign(h[li], state[li][1]) for li in range(n_layers)]):
    for ui in range(num_unrollings):
        loss += tf.reduce_mean(0.5 * (split_outputs[ui] - train_outputs[ui]) ** 2)

print('Learning rate decay operations')
global_step = tf.Variable(0, trainable=False)
inc_gstep = tf.assign(global_step, global_step + 1)
tf_learning_rate = tf.placeholder(shape=None, dtype=tf.float32)
tf_min_learning_rate = tf.placeholder(shape=None, dtype=tf.float32)

learning_rate = tf.maximum(
    tf.train.exponential_decay(tf_learning_rate, global_step, decay_steps=1, decay_rate=0.5, staircase=True),
    tf_min_learning_rate)

print('TF Optimization operations')
optimizer = tf.train.AdamOptimizer(learning_rate)
gradients, v = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
optimizer = optimizer.apply_gradients(zip(gradients, v))
print('\tAll done')

print('Defining prediction related TF functions')

sample_inputs = tf.placeholder(tf.float32, shape=[1, D])

sample_c, sample_h, initial_sample_state = [], [], []
for li in range(n_layers):
    sample_c.append(tf.Variable(tf.zeros([1, num_nodes[li]]), trainable=False))
    sample_h.append(tf.Variable(tf.zeros([1, num_nodes[li]]), trainable=False))
    initial_sample_state.append(tf.contrib.rnn.LSTMStateTuple(sample_c[li], sample_h[li]))

reset_sample_states = tf.group(*[tf.assign(sample_c[li], tf.zeros([1, num_nodes[li]])) for li in range(n_layers)],
                               *[tf.assign(sample_h[li], tf.zeros([1, num_nodes[li]])) for li in range(n_layers)])

sample_outputs, sample_state = tf.nn.dynamic_rnn(multi_cell, tf.expand_dims(sample_inputs, 0),
                                                 initial_state=tuple(initial_sample_state),
                                                 time_major=True,
                                                 dtype=tf.float32)

with tf.control_dependencies([tf.assign(sample_c[li], sample_state[li][0]) for li in range(n_layers)] +
                             [tf.assign(sample_h[li], sample_state[li][1]) for li in range(n_layers)]):
    sample_prediction = tf.nn.xw_plus_b(tf.reshape(sample_outputs, [1, -1]), w, b)

print('\tAll done')
