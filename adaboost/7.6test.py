# -*- coding: utf-8 -*-
# @Time    : 2016/12/20 15:25
# @Author  : plr
# @Site    : 
# @File    : 7.6test.py
# @Software: PyCharm

from adaboost import *

"""
（1）收集数据：提供的文本文件
（2）准备数据：确保类别标签是+1和-1而非1和0
（3）分析数据：手工检查数据
（4）训练算法：在数据上，利用adaBoostTrainDS()函数训练处一系列分类器
（5）测试算法：我们拥有两个数据集。在不采用随机抽样的方法下，我们就会对AdaBoost和Logistic回归的结果进行完全对等的比较
（6）使用算法：观察该例子上的错误率。不过，也可以构建一个web网站，让驯马师输入马的症状然后预测马是否会死去
"""

def loadDataSet(fileName):
    """
    导入马疝病的数据集，并分为数据集合，类别集合
    :param fileName:
    :return:
    """
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

if __name__ == '__main__':
    datArr, labelArr = loadDataSet('Ch07/horseColicTraining2.txt')
    classifierArr = adaBoostTrainDS(datArr, labelArr, 50)  # 这个线性组合分类器最终的分类误差率为0.23
    # 应用到测试数据集上来判断该方法的误差
    testArr, testLabelArr = loadDataSet('Ch07/horseColicTest2.txt')
    prediction10 = adaClassify(testArr, classifierArr)
    errArr = np.mat(np.ones((67,1)))
    print(errArr[prediction10 != np.mat(testLabelArr).T].sum())