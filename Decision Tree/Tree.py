# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:13:02 2016

@author: plr
"""

from math import log
import operator


def createDataSet():
    dataSet = [['youth', 'no', 'no', 'generally', 'refuse'],
               ['youth', 'no', 'no', 'good', 'refuse'],
               ['youth', 'yes', 'no', 'good', 'agree'],
               ['youth', 'yes', 'yes', 'generally', 'agree'],
               ['youth', 'no', 'no', 'generally', 'refuse'],
               ['middle-aged', 'no', 'no', 'generally', 'refuse'],
               ['middle-aged', 'no', 'no', 'good', 'refuse'],
               ['middle-aged', 'yes', 'yes', 'good', 'agree'],
               ['middle-aged', 'no', 'yes', 'excellent', 'agree'],
               ['middle-aged', 'no', 'yes', 'excellent', 'agree'],
               ['the old', 'no', 'yes', 'excellent', 'agree'],
               ['the old', 'no', 'yes', 'good', 'agree'],
               ['the old', 'yes', 'no', 'good', 'agree'],
               ['the old', 'yes', 'no', 'excellent', 'agree'],
               ['the old', 'no', 'no', 'generally', 'refuse']
               ]
    labels = ['ages', 'have job', 'have house', 'credit conditions']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    """
    选定某一维度进行拆分数据，比如年龄，将这个维度分为三部分，并去掉这一维度，做成这样的
    [[u'no', u'no', u'generally', u'refuse'],
     [u'no', u'no', u'good', u'refuse'],
     [u'yes', u'no', u'good', u'agree'],
     [u'yes', u'yes', u'generally', u'agree'],
     [u'no', u'no', u'generally', u'refuse']]
     以上yes “youth”部分，“middle-aged”“the old”部分类似
    """
    # global reducedFeatvec
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatvec = featVec[:axis]
            reducedFeatvec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatvec)
    return retDataSet


def calcShannonEnt(dataSet):
    """
    计算给定的数据集的香农熵
    """
    numEntries = len(dataSet)  # 实例的总数
    labelCounts = {}  # {类别：类别数目}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # 计算dataSet的经验熵 H(dataSet)
    return shannonEnt


def calcConditionalEntropy(dataSet, i, uniqueVals):
    """
    uniqueVals: [u'youth', u'middle-aged', u'the old']
    """
    ce = 0.0
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet, i, value)
        prob = len(subDataSet) / float(len(dataSet))  # 极大似然估计概率
        ce += prob * calcShannonEnt(subDataSet)  # ∑p*H(Y|X=xi) 条件熵的计算
    return ce


def calcInformationGain(dataSet, i, baseEntropy):
    """
    计算信息熵增益
    这里传入的baseEntropy，需要注意！！！！！！！！！！！！！！！！！！！
    """
    featList = [example[i] for example in dataSet]  # 第i维特征列表
    uniqueVals = set(featList)  # 转换成集合
    newEntropy = calcConditionalEntropy(dataSet, i, uniqueVals)
    infoGain = baseEntropy - newEntropy  # 信息增益，就是熵的减少，也就是不确定性的减少
    return infoGain


# def calcInformationGainRate(dataSet, baseEntropy, i):
#    return calcInformationGain(dataSet, i, )   

def chooseBestFeatureToSplitByID3(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        infoGain = calcInformationGain(dataSet, i, baseEntropy)
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    """
    >> a = {'b':1, 'c':2, 'd':2, 'e':3}
    >> sorted(a.iteritems(), key=operator.itemgetter(1), reverse=True)
     - [('e', 3), ('c', 2), ('d', 2), ('b', 1)]
    >>
    如果遍历完所有属性后类标签还是不统一，选择数量最多的作为该分支的叶节点
    """
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels, chooseBestFeatureToSplitFunc=chooseBestFeatureToSplitByID3):
    """
    检测数据集中的每个子项是否属于同一分类：
        if so return 类标签；
        else
            寻找划分数据集的最好特征
            划分数据集
            创建分支节点
                for 每个划分的子集
                    调用函数createTree并增加返回结果到分支节点中
            return 分支节点
    """
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): # 递归结束条件1：每个分支下的所有实例都具有相同的分类
        return classList[0]
    if len(dataSet[0]) == 1: # 递归结束条件2：程序遍历完所有划分数据集的属性
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplitFunc(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree
