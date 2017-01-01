# -*- coding: utf-8 -*-
# @Time    : 2016/12/12 17:40
# @Author  : plr
# @Site    : 
# @File    : test.py
# @Software: PyCharm

from svmMLiA import *
import matplotlib.pyplot as plt

def simpleSMO():
    """
    简化版的smo算法
    分离超平面是由少数的支持向量来决定的
    :return:
    """
    filepath = 'Ch06/'
    dataArr, labelArr = loadDataSet(filepath + 'testSet.txt')
    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    print('b is ', b)
    print(alphas[alphas>0])
    for i in range(100):
        if alphas[i] > 0.0:
            print(dataArr[i], labelArr[i])
    return b, alphas, dataArr, labelArr

def entireSMO():
    """
    完整版的smo算法
    :return:
    """
    filepath = 'Ch06/'
    dataArr, labelArr = loadDataSet(filepath + 'testSet.txt')
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    print('b is ', b)
    print(alphas[alphas > 0])
    for i in range(100):
        if alphas[i] > 0.0:
            print(dataArr[i], labelArr[i])
    return b, alphas, dataArr, labelArr

def calcWs(alphas, dataArr, classLabels):
    """
    通过得出的最优alphas值来计算分离超平面的权重参数w
    w = sigma(alphas_i * label_i * X_i.T)
    :param alphas:
    :param dataArr:
    :param classLabels:
    :return:
    """
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i], X[i, :].T)
    return w

def plotSVM(dataArr, labelArr, w, b, svList):
    """
    首先在图中画出每个点，并根据标签类分为圆点方点
    将支持向量标记出来
    画出超分离平面来
    :param dataArr:
    :param labelArr:
    :param w:
    :param b:
    :param svList:
    :return:
    """
    xcord0 = []
    ycord0 = []
    xcord1 = []
    ycord1 = []
    for i in range(len(labelArr)):
        xPt = dataArr[i][0]
        yPt = dataArr[i][1]
        label = labelArr[i]
        if (label == -1):
            xcord0.append(xPt)
            ycord0.append(yPt)
        else:
            xcord1.append(xPt)
            ycord1.append(yPt)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord0, ycord0, marker='s', s=90)
    ax.scatter(xcord1, ycord1, marker='o', s=50, c='red')
    plt.title('Support Vectors Circled')
    for sv in svList:
        circle = plt.Circle((dataArr[sv][0], dataArr[sv][1]), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3,
                            alpha=0.5)
        ax.add_patch(circle)

    w0 = w[0][0]
    w1 = w[1][0]
    b = float(b)
    x = np.arange(-2.0, 12.0, 0.1)
    y = (-w0*x - b) / w1
    ax.plot(x, y)
    ax.axis([-2, 12, -8, 6])
    # ax.axis([-12, 12, -12, 12])
    plt.show()




if __name__ == '__main__':
    # b, alphas, dataArr, labelArr = simpleSMO()
    b, alphas, dataArr, labelArr = entireSMO()
    w = calcWs(alphas, dataArr, labelArr)
    print('w is ', w)
    svList = []
    for i in range(len(alphas)):
        if (abs(alphas[i]) > 0.0000001):
            print('support Vectors is', dataArr[i], labelArr[i])
            svList.append(i)

    plotSVM(dataArr, labelArr, w, b, svList)