# -*- coding: utf-8 -*-
# @Time    : 2016/11/21 14:38
# @Author  : plr
# @Site    : 
# @File    : logisticRegressionGif.py
# @Software: PyCharm

import numpy as np
from math import exp
import matplotlib.pyplot as plt
from matplotlib import animation
import copy

filepath = 'Ch05/'

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open(filepath + 'testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    """
    梯度上升法
    :param inX:
    :return:
    """
    y = []
    for i in inX:
        a = 1.0/(1+exp(-i))
        y.append(a)
    return np.mat(y).transpose()

def sigmoid_one(inX):
    """
    随机梯度下降法
    :param inX:
    :return:
    """
    return 1.0 / (1+exp(-inX))

def gradAscent(dataMatIn, classLabels, history_weight):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights += alpha * dataMatrix.transpose() * error
        history_weight.append(copy.copy(weights))
    return weights

def stocGradAscent0(dataMatrix, classLabels, history_weight):
    dataMatrix = np.array(dataMatrix)
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid_one(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + dataMatrix[i] * alpha * error
        history_weight.append(copy.copy(weights))
    return weights

def stocGradAscent1(dataMatrix, classLabels, history_weight, numIter=150):
    dataMatrix = np.array(dataMatrix)
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4.0/(1.0+j+i) + 0.0001
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid_one(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
            history_weight.append(copy.copy(weights))
    return weights

history_weight = []
dataMat, labelMat = loadDataSet()
stocGradAscent1(dataMat, labelMat, history_weight)
fig = plt.figure()
currentAxis = plt.gca()
ax = fig.add_subplot(111)
line, = ax.plot([], [], 'b', lw=2)

def draw_line(weights):
    x = np.arange(-5.0, 5.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    line.set_data(x, y)
    return line,

def init():
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    plt.xlabel('X1'); plt.ylabel('X2')
    return draw_line(np.zeros((n,1)))
    
def animate(i):
    return draw_line(history_weight[i])

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(history_weight), interval=10,
                               repeat=False, blit=True)
plt.show()
anim.save('C:/Users/plr/Desktop/gradAscent.gif', fps=2, writer='imagemagick')
