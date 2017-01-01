# -*- coding: utf-8 -*-
# @Time    : 2016/11/24 19:48
# @Author  : plr
# @Site    : 
# @File    : regression.py
# @Software: PyCharm

import numpy as np


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr, yArr):
    """
    w = (xTx) ** -1 xTy
    :param xArr:
    :param yArr:
    :return:
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print 'This matrix is singular, cannot do inverse.'
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

def lwlr(testPoint, xArr, yArr, k=1.0):
    """
    依据某个训练实例的特征值，来计算权重，从而根据这个权重计算出系数，这组系数只能用来计算这个训练实例的y值
    :param testPoint:
    :param xArr:
    :param yArr:
    :param k:
    :return:
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * weights * xMat
    if np.linalg.det(xTx) == 0.0:
        print 'This matrix is singular, cannot do inverse.'
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    """
    通过局部加权线性回归来估计所有的y值
    :param testArr:
    :param xArr:
    :param yArr:
    :param k:
    :return:
    """
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def rssError(yArr,yHatArr):
    """
    误差平方和
    :param yArr:
    :param yHatArr:
    :return:
    """
    print ((yArr-yHatArr)**2).sum()
    return ((yArr-yHatArr)**2).sum()

def ridgeRegres(xMat, yMat, lam=0.2):
    """
    岭回归
    :param xMat:
    :param yMat:
    :param lam:
    :return:
    """
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print 'This matrix is singular, cannot do inverse.'
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    """
    将数据标准化，针对不同的lambda值得出不同的系数，观察他们之间的关系
    :param xArr:
    :param yArr:
    :return:
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i-10))
        wMat[i, :] = ws.T
    return wMat

def stageWise(xArr, yArr, eps=0.01, numIt=100):
    """
    分别让权重向量中的j元素正向和反向的前进0.01，计算出估计的yTest，将其与训练数据集已知y进行误差分析，如果小于阈值就讲权重更新
    前向逐步回归
    :param xArr:
    :param yArr:
    :param eps:
    :param numIt:
    :return: 得到权重更新矩阵
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsMax = ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError = np.inf;
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat