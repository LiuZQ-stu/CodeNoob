# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 20:14:12 2020

@author: 华为
"""

import numpy as np
import csv
from matplotlib import pyplot as plt

#读取数据
data = []
for i in range(18):
    data.append([])
    
with open("data/train.csv",'r',encoding='big5') as mydata:
    lines = csv.reader(mydata, delimiter=',')
    n = 0
    n_row = 0
    for line in lines:
        if n_row != 0:
            for i in range(3,27):
                if line[i] != "NR":
                    data[(n_row-1)%18].append(float(line[i]))
                else:
                    data[(n_row-1)%18].append(0)
        n_row = n_row + 1
#data = np.array(data)
#data = np.reshape(data,(n-1,24))
#print(data)

#准备测试数据
test_data = []
for i in range(18):
    test_data.append([])
    
with open("data/test.csv",'r') as mytest:
    lines = csv.reader(mytest, delimiter=',')
    n_row = 0
    for line in lines:
        for i in range(2,11):
            if line[i] == "NR":
                test_data[n_row%18].append(0)
            else:
                test_data[n_row%18].append(float(line[i]))
        n_row = n_row + 1
test_X = test_data[9]
test_X = np.array(test_X)
test_X = test_X.reshape((240,9))
#print(test_X)
            
#定义adagrad
def ada_update(Y, X, w, eta, iteration, lambdaL2):
    cost_list = []
    sum_grad2 = np.zeros(len(X[0]))
    for i in range(iteration):
        Y1 = np.dot(X, w)
        loss = Y1 - Y
        cost_list.append(np.sum(loss **2)/len(X))
        grad = np.dot(X.T, loss)/len(X) + lambdaL2 * w
        sum_grad2 = sum_grad2 + grad ** 2
        w = w - eta / np.sqrt(sum_grad2) * grad
    return w, cost_list

#测试模型程序
def test(test_Y, test_X, w):
    return np.dot(test_X, w)
    
    

#准备训练数据
x = []
y = []

for i in range(240):
    for j in range(14):
        x.append([])
        for k in range(9):
            x[i*14+j].append(data[9][24*i+j+k])
        y.append(data[9][24*i+j+k+1])
x = np.array(x)
y = np.array(y)
#print("x =", x)
#print("y =", y)
#print(data[8])


#训练数据
#eex = np.concatenate((np.ones((x.shape[0],1)), x), axis=1)
w = np.zeros(len(x[0]))
w_ada, cost_list = ada_update(y, x, w, eta=0.01, iteration=20000, lambdaL2=0)
#print(cost_list[-1])
#print(w_spd)
#print(cost_list)


#可视化
cost_x = range(len(cost_list))
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title('Adagrad Update')
plt.plot(cost_x, cost_list)
plt.show()






