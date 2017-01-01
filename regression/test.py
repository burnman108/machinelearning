# -*- coding: utf-8 -*-
# @Time    : 2016/11/24 20:27
# @Author  : plr
# @Site    : 
# @File    : test.py
# @Software: PyCharm

from regression import *
import matplotlib.pyplot as plt


def stand():
    xArr, yArr = loadDataSet('Ch08/ex0.txt')
    ws = standRegres(xArr, yArr)

    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    # yHat = xMat * ws

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1], yMat.T[:,0])

    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:,1], yHat)

    plt.show()

def lwlr():
    xArr, yArr = loadDataSet('Ch08/ex0.txt')
    yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    xMat = np.mat(xArr)
    strInd = xMat[:,1].argsort(0)
    xSort = xMat[strInd][:,0,:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1], yHat[strInd])
    ax.scatter(xMat[:,1], np.mat(yArr).T[:,0], s=2, c='r')
    plt.show()

def ridge():
    abX, abY = loadDataSet('Ch08/abalone.txt')
    ridgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()

def stage():
    abX, abY = loadDataSet('Ch08/abalone.txt')
    # abX, abY = loadDataSet('Ch08/ex0.txt')

    stageWise(abX, abY, 0.01, 200)

if __name__ == '__main__':
    ridge()
