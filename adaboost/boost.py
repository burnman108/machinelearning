# -*- coding: utf-8 -*-
# @Time    : 2016/12/19 16:29
# @Author  : plr
# @Site    : 
# @File    : boost.py
# @Software: PyCharm

import numpy as np

# 将最小错误率minError设为正无穷
# 对数据集中的每一个特征（第一层循环）：
#     对每个步长（第二层循环）：
#         对每个不等号（第三层循环）：
#             建立一颗单层决策树并利用加权数据集对它进行测试
#             如果错误率低于minError，则将当前单层决策树设为最佳单层决策树
# 返回最佳单层决策树

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    通过阈值比较对数据进行分类
    :param dataMatrix:
    :param dimen: 数据集中的某一元素（列）
    :param threshVal: 阈值
    :param threshIneq: 切换<=和>号
    :return:
    """
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':  # 小于等于
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:  # 大于
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    """
    一个弱分类方法
    :param dataArr:
    :param classLabels:
    :param D: 数值权重向量
    :return:
    """
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0 # 用于在特征的所有可能值上进行遍历，二分类下的分类点
    bestStump = {} # 储存给定权重向量D时所得到的最佳单层决策树的相关信息
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf # 初始化成正无穷大，之后用于寻找可能的最小错误率
    for i in range(n):
        # 第一层for循环在数据集的所有特征上遍历
        rangeMin = dataMatrix[:, i].min()  # 考虑到数值的特征，通过计算最大值和最小值来了解应该需要多大的步长
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps  # 步长计算
        for j in range(-1, int(numSteps)+1):
            # 第二层for循环在所有的阈值上遍历
            for inequal in ['lt', 'gt']:
                # 第三层for循环在大于和小于之间切换不等式
                threshVal = (rangeMin + float(j) * stepSize)  # 阈值的计算（有很多选择，要选出能使分类误差率最小的那一个）
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)  # 通过分类器得出的预测结果
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat] = 0  # 将预测结果中与实际分类相符的标记为0，因为计算分类误差率时只用到错误分类的权重
                weightedError = D.T * errArr  # 分类器在训练数据集上的分类误差率

                # print('split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f' %
                #       (i, threshVal, inequal, weightedError))

                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i  # 阈值所在的特征
                    bestStump['thresh'] = threshVal  # 阈值
                    bestStump['ineq'] = inequal  # 不等号的方向
    return bestStump, minError, bestClasEst