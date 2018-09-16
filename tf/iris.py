''#！/usr/bin/env python
# _*_ coding:utf-8 _*_

'''

@author: yerik

@contact: xiangzz159@qq.com

@time: 2018/9/16 20:49

@desc:

'''

import numpy as np
import sklearn as skl
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data  # 数据，训练测试使用
iris_Y = iris.target    # 标签，监督学习验证结果
print(iris.data.shape)
print(iris_X[0])

# 拆分，训练数据和测试数据。按照 0.7 0.3 比例。
X_train,X_test,y_train,y_test = train_test_split(iris_X,iris_Y,test_size=0.3)
# print(y_train) #打乱测试数据。

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
print(knn.predict(X_test)-y_test) #计算预测和目标的差
print("score:", knn.score(X_train,y_train)) #评分