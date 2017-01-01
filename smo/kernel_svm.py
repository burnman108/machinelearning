# -*- coding: utf-8 -*-
# @Time    : 2016/12/14 16:26
# @Author  : plr
# @Site    : 
# @File    : kernel_svm.py
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

def kernelTrans(X, A, kTup):
    """
    :param X:
    :param A:
    :param kTup: kTup是一个包含核函数信息的元组,元组的第一个参数是描述所用核函数类型的一个字符串，其他两个参数则都是核函数可能需要的可选参数
    :return:
    """
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        # 线性核函数情况下，内积计算在“所有数据集”和“数据集中的一行”这两个输入之间展开
        K = X * A.T
    elif kTup[0] == 'rbf':
        # 径向核函数情况下，在for循环中对于矩阵的每个元素计算高斯函数的值，for循环结束后，将计算过程应用到整个向量上
        # 这里的kTup[1]是用户定义的用于确定到达率（reach）或者说函数值跌落到0的速度参数
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow*deltaRow.T
        K = np.exp(K / (-1*kTup[1]**2))
    else:
        raise NameError('Houston we Have a Problem -- That Kernel is not recognized')
    return K

class optStruct:
    """
    这里使用对象的目的并不是面向对象编程，而只是作为一个数据结构来使用对象
    """
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))  # eCache的第一列给出的是eCache是否有效的标志位，而第二列给出的是实际的E值
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)

def calcEk(oS, k):
    """
    计算预测的类别
    基于实例的预测结果和真实结果的比对，计算出误差Ek
    :param oS: 传入的对象
    :param k: 第k个实例
    :return:
    """
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k]) + oS.b
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
        eta = 2.0 * oS.K[i,j] - oS.K[i, i] - oS.K[j, j]
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
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.K[i, i] - \
            oS.labelMat[j]*(oS.alphas[j] - alphaJold)*oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
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
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)  # 构建一个数据结构来容纳所有的数据
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