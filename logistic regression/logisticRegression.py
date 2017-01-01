# -*- coding: utf-8 -*-
# @Time    : 2016/11/21 9:44
# @Author  : plr
# @Site    : 
# @File    : logisticRegression.py
# @Software: PyCharm

import numpy as np
from math import exp

filepath = 'Ch05/'

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open(filepath + 'testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    if weights is not None:
        x = np.arange(-3.0, 3.0, 0.1)
        y = (-weights[0] - weights[1]*x) / weights[2]
        ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

# plotBestFit(None)

def sigmoid(inX):
    y = []
    for i in inX:
        a = 1.0/(1+exp(-i))
        y.append(a)
    return np.mat(y).transpose()

def sigmoid_one(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights) # 更新过程不太懂，h是输出y
        error = (labelMat - h) # error是训练数据集的y值与通过估计权值得出的y值之间的误差向量
        weights += alpha * dataMatrix.transpose() * error # 对权值进行更新

    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    dataMatrix = np.array(dataMatrix)
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for i in range(numIter):
        dataIndex = range(m)
        for j in dataIndex:
            alpha = 4.0/(1.0+i+j) + 0.0001
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid_one(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

dataArr, labelMat = loadDataSet()
# weights_0 = gradAscent(dataArr, labelMat)
weights_1 = stocGradAscent1(dataArr, labelMat)
plotBestFit(weights_1)