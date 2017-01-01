# -*- coding: utf-8 -*-
# @Time    : 2016/11/30 10:29
# @Author  : plr
# @Site    : 
# @File    : regTrees.py
# @Software: PyCharm

import numpy as np

def loadDataSet(fileName):
    """
    载入数据集
    :param fileName:
    :return:
    """
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        # curLine = line.strip().split(',')
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    """
    根据选出的特征和分割点，将数据集分割成两个部分
    :param dataSet:
    :param feature:
    :param value:
    :return:
    """
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1

def regLeaf(dataSet):
    """
    :param dataSet:
    :return: 目标值集合的均值
    """
    return np.mean(dataSet[:, -1])

def regErr(dataSet):
    """
    :param dataSet:
    :return: 误差：dataSet的平方误差
    """
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    选择最佳的切分特征和切割点
    :param dataSet:
    :param leafType:
    :param errType:
    :param ops:
    :return:
    """
    tolS = ops[0]; tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)  # 当前数据子集的平方误差
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue

def creatTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    构建树
    :param dataSet:
    :param leafType:
    :param errType:
    :param ops:
    :return:
    """
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    rSet, lSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = creatTree(lSet, leafType, errType, ops)
    retTree['right'] = creatTree(rSet, leafType, errType, ops)
    return retTree

# 后剪枝

def isTree(obj):
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['right'] + tree['left']) / 2.0

def prune(tree, testData):
    """
    使用后剪枝方法需要将数据集分成测试集和训练集。
    首先指定参数，使得构建出的树足够大、足够复杂，便于剪枝。
    接下来从上而下找到叶节点，用测试集来判断将这些叶节点合并是否能降低误差，如果是的话就合并。
    :param tree:
    :param testData:
    :return:
    """
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(np.power(lSet[:, -1] - tree['left'], 2)) + sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(np.power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree

# 模型树

# def func(x, p):
#     a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4, a5, b5, c5, a6, b6, c6 = p
#     return (a1*x[:,4]**2 + b1*x[:,4] + c1) * (a2*x[:,5]**2 + b2*x[:,5] + c2) * \
#            (a3*x[:,2]**2 + b3*x[:,2] + c3) * (a4*x[:,3]**2 + b4*x[:,3] + c4) * \
#            (a5*x[:,0]**2 + b5*x[:,0] + c5) * (a6*x[:,1]**2 + b6*x[:,1] + c6)
#
# def func_error(p, y, x):
#     return np.sum((y-func(x, p))**2)

def linearSolve(dataSet):
    # dataSet_a = np.array(dataSet)[:, :-1]  # 除去y值的特征值矩阵
    # dataSet_a_2 = dataSet_a ** 2  # 每个特征值做平方
    # dataSet_con = np.concatenate([dataSet_a, dataSet_a_2], axis=1)  # 将一次项矩阵与二次项矩阵合并
    # dataSetMat = np.mat(dataSet_con)
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m, n)))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse, try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y
    # from scipy import optimize
    # dataSet_x = np.array(dataSet)[:, :-1]
    # dataSet_y = np.array(dataSet)[:, -1]
    # result = optimize.basinhopping(func_error, (-0.0003, 0.005, -0.1552, -1.17e-05, -3.17e-05, 0.0195,
    #                                             -0.0043, 0.2688, 1.935, 0.0011, -0.902, 5.23, -9.71e-07,
    #                                             -0.0277, -0.788, 3.57e-05, -0.033, 10.174), niter=10,
    #                                minimizer_kwargs={"method": "L-BFGS-B", "args": (dataSet_y, dataSet_x)})
    # return result.x

def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(np.power(Y - yHat, 2))
    # ws = linearSolve(dataSet)
    # dataSet_x = np.array(dataSet)[:, :-1]
    # dataSet_y = np.array(dataSet)[:, -1]
    # return func_error(ws, dataSet_y, dataSet_x)

# 用树回归进行预测

# def regTreeEval(model):
#     return float(model)
# def modelTreeEval(model, inDat):
#     n = np.shape(inDat)[1]
#     X = np.mat(np.ones((1, n + 1)))
#     X[:, 1:n + 1] = inDat
#     return float(X * model)
# def treeForeCast(tree, inData, modelEval=regTreeEval):
#     if not isTree(tree):
#         return modelEval(tree, inData)
#     if inData[:, tree['spInd']] < tree['spVal']:
#         if isTree(tree['left']):
#             return treeForeCast(tree['left'], inData, modelEval)
#         else:
#             return modelEval(tree['left'], inData)
#     else:
#         if isTree(tree['right']):
#             return treeForeCast(tree['right'], inData, modelEval)
#         else:
#             return modelEval(tree['right'], inData)
# def createForeCast(tree, testData, modelEval=regTreeEval):
#     m = len(testData)
#     yHat = np.mat(np.zeros((m, 1)))
#     for i in range(m):
#         yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
#     return yHat
def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n+1)))
    X[:, 1:n+1] = inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[:, tree['spInd']] < tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)

def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat
#
# def createForeCast(tree, testData, modelEval=regTreeEval):
#     m = len(testData)
#     yHat = np.mat(np.zeros((m, 1)))
#     for i in range(m):
#         testData_a = np.array(testData[i])
#         testData_a_2 = testData_a ** 2
#         testData_con = np.concatenate([testData_a, testData_a_2], axis=1)
#         yHat[i, 0] = treeForeCast(tree, np.mat(testData_con), modelEval)
#     return yHat