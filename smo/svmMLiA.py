# -*- coding: utf-8 -*-
# @Time    : 2016/12/12 11:20
# @Author  : plr
# @Site    : 
# @File    : svmMLiA.py
# @Software: PyCharm

import random
import numpy as np

def loadDataSet(fileName):
    """
    加载数据集
    :param fileName:
    :return: 特征数据列表， 分类标签列表
    """
    dataMat = []  # 特征数据
    labelMat = []  # 分类标签
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i, m):
    """
    只要函数值不等于输入值i，函数就会进行随机选择
    第二个alpha随机取了与i不相同的数
    :param i: 第一个alpha的下标
    :param m: 所有alpha的数目
    :return:
    """
    j = i
    while(j == i):
        # 如果j等于i的话，就让j在样本集中随机取数，如果j不等于i，就返回结果
        j = int(random.uniform(0, m))
    return j

def cliAlpha(aj, H, L):
    """
    用于调整大于H或小于L的alpha的值
    :param aj:
    :param H:
    :param L:
    :return:
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

# SMO函数的伪码大致如下：
#
#   创建一个alpha向量并将其初始化为0向量
#   当迭代次数小于最大迭代次数时（外循环）
#      对数据集中的每个数据向量（内循环）：
#         如果该数据向量可以被优化：
#            随机选择另外一个数据向量
#            同时优化这两个向量
#            如果两个向量都不能被优化，退出内循环
#      如果所有向量都没被优化，增加迭代数目，继续下一次循环

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """

    :param dataMatIn: 数据集
    :param classLabels: 类别标签
    :param C: 常数C
    :param toler: 容错率
    :param maxIter: 取消前最大的循环次数
    :return:
    """
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0  # iter变量储存的是在没有任何alpha改变的情况下遍历数据的次数，当该变量达到输入值maxIter时，函数结束并退出
    while(iter < maxIter):
        alphaPairsChanged = 0  # 用于记录alpha是否已经进行优化
        for i in range(m):
            #   可求出的有:
            #       ————Ei Ej alphaIold alphaJold
            #   w = alpha * y * x;  f(x_i) = w^T * x_i + b
            fXi = float(np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b  # 预测的类别
            Ei = fXi - float(labelMat[i])  # 基于实例的预测结果和真实结果的比对，计算出误差Ei
            # 如果误差很大，那么可以对该数据实例所对应的alpha值进行优化，不管是正间隔还是负间隔都会被测试，保证alpha不能等于0或C
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)  # 随机选择第二个alpha值
                fXj = float(np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()  # alpha_I_old
                alphaJold = alphas[j].copy()  # alpha_J_old
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[i] + alphas[j] - C)
                    H = min(C, alphas[i] + alphas[j])
                if L == H:
                    # L == H的话重新选择alphaI 和 alphaJ
                    print('L==H')
                    continue
                eta = 2.0 * dataMatrix[i, :]*dataMatrix[j, :].T - dataMatrix[i, :]*dataMatrix[i, :].T - \
                      dataMatrix[j, :]*dataMatrix[j, :].T  # alpha[j]的最优修改量，与书本上符号相反了，即eta=-ETA(书本上）
                if eta >= 0:
                    print('eta>=0')
                    continue
                alphas[j] -= labelMat[j]*(Ei - Ej) / eta # 沿着约束方向未经剪辑时的解，未考虑L和H
                alphas[j] = cliAlpha(alphas[j], H, L)  # 经过剪辑后的alphaJ的解,考虑L和H
                if (np.abs(alphas[j] - alphaJold) < 0.00001):
                    print('j not moving enough')
                    continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])  # 书本上通过alphaJ求得alphaI
                b1 = b - Ei - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - \
                    labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - \
                    labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print('iter: %d  i: %d, pairs changed %d' % (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1  # 当没有任何alpha发生改变时会将整个集合的一次遍历过程计成一次迭代
        else:
            iter = 0
        print('iteration number: %d' % iter)
    return b, alphas

#=================完整版Platt SMO的支持函数====================

class optStruct:
    """
    这里使用对象的目的并不是面向对象编程，而只是作为一个数据结构来使用对象
    """
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))  # eCache的第一列给出的是eCache是否有效的标志位，而第二列给出的是实际的E值

def calcEk(oS, k):
    """
    计算预测的类别
    基于实例的预测结果和真实结果的比对，计算出误差Ek
    :param oS: 传入的对象
    :param k: 第k个实例
    :return:
    """
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X*oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    """
    这里的目标是选择合适的第二个alpha值以保证在每次优化中采用最大步长。该函数的误差值与第一个alpha值Ei和下标i有关。
    从所有不为0的alpha中选择一个与当前i样本误差距离最大的那个
    :param i:
    :param oS:
    :param Ei:
    :return:
    """
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0])[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):
    """
    在eCache中更新第k个样本与其预测误差
    :param oS:
    :param k:
    :return:
    """
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def innerL(i, oS):
    """
    这里的代码几乎与smoSimple()函数一模一样，但是有两点不同：
    1 代码使用了自己的数据结构，该结构在参数oS中传递
    2 使用程序中的seletJ()函数而不是selectJrand()来选择第二个alpha的值
    :param i:
    :param oS:
    :return:
    """
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)  #
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print('L == H')
            return 0
        eta = 2.0 * oS.X[i, :]*oS.X[j, :].T - oS.X[i, :]*oS.X[i, :].T - oS.X[j, :]*oS.X[j, :].T
        if eta >= 0:
            print('eta >= 0')
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej) / eta
        oS.alphas[j] = cliAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)  # 在alpha改变时更新Ecache误差缓存
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print(' j not moving enough')
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.X[i, :]*oS.X[i, :].T - \
            oS.labelMat[j]*(oS.alphas[j] - alphaJold)*oS.X[i, :]*oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter):
    """
    完整版Platt SMO的外循环代码，
    对于外层循环，这里的实现并没有在训练样本中选取违反KKT条件最严重的样本点，而是优先顺序遍历间隔边界上的支持向量点，若无法优化模型则遍历
    整个数据集。
    :param dataMatIn:
    :param classLabels:
    :param C:
    :param toler:
    :param maxIter:
    :param kTup:
    :return:
    """
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)  # 构建一个数据结构来容纳所有的数据
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or entireSet):
        # 当迭代次数超过指定的最大值，或者遍历整个集合都未对任意alpha对进行修改时，就退出循环
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            print('fullSet, iter: %d   i: %d, pairs changed %d' % (iter, i, alphaPairsChanged))
            iter += 1  # 这里的一次迭代定义为一次循环过程，而不管循环具体做了什么事
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
            print('non-bound, iter: %d   i: %d, pairs changed %d' % (iter, i, alphaPairsChanged))
            iter += 1  # 这里的一次迭代定义为一次循环过程，而不管循环具体做了什么事
        if entireSet:
            entireSet = False   # 翻转entireSet
        elif (alphaPairsChanged == 0):
            entireSet = True
        print('iteration number: %d' % iter)
    return oS.b, oS.alphas