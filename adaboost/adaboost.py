# -*- coding: utf-8 -*-
# @Time    : 2016/12/19 17:07
# @Author  : plr
# @Site    :
# @File    : adaboost.py
# @Software: PyCharm



from boost import *

def loadSimpData():
    """

    构建一个单层决策树，它仅基于单个特征来做决策，由于这棵树只有一次分裂过程，因此它实际上就是一个树桩
    :return:
    """
    datMat = np.matrix([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    D = np.mat(np.ones((5, 1)) / 5)
    result = buildStump(datMat, classLabels, D)
    print(result)

# 对每次迭代：
#   利用bulidStump()函数找到最佳的单层决策树
#   将最佳单层决策树加入到单层决策树组
#   计算alpha
#   计算新的权重向量D
#   更新累计类别估计值
#   如果错误率等于0.0，则退出循环

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    函数名称尾部的DS代表的就是单层决策树
    :param dataArr: 数据集
    :param classLabels: 类别标签
    :param numIt: 迭代次数
    :return: 多个弱分类器组成的数组[{'dim': ?, 'thresh': ?, 'ineq': ?, 'alpha': ?},.....]
    """
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m,1))/m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)

        # print('D:', D.T)

        alpha = float(0.5*np.log((1.0-error)/max(error, 1e-16)))  # alpha是分类器的系数
        bestStump['alpha'] = alpha  # 将alpha加入到字典当中
        weakClassArr.append(bestStump)  # 将每轮迭代产生的bestStump加入到weakClassArr中

        # print('classEst:', classEst.T)

        expon = np.multiply(-1*alpha*np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()  # 这三步是来计算新的训练数据的权值分布
        aggClassEst += alpha * classEst  # 构建基本分类器的线性组合，这个列向量记录着每个数据点的类别估计累计值

        # print('aggClassEst:', aggClassEst.T)

        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m,1)))  # 分类器预测错误会返回1，预测正确返回0
        errorRate = aggErrors.sum() / m  # 分类器误差率

        print('total error:', errorRate, '\n')

        if errorRate == 0.0:
            # 如果迭代中的错误率为0，则会退出迭代过程
            break
    return weakClassArr

"""
现在需要做的就只是将弱分类器的训练过程从程序中抽出来，然后应用到某个具体的实例上去。每个弱分类器的结果以其对应的
alpha值作为权重。所有这些弱分类器的结果加权求和就得到了最后的结果。
"""
def adaClassify(datToClass, classifierArr):
    """
    利用训练出的多个弱分类器进行分类的函数
    :param datToClass: 一个或者多个待分类样例
    :param classifierArr: 多个弱分类器组成的数组
    :return:
    """
    dataMatrix = np.mat(datToClass)  # 将待分类样例转化为numpy矩阵
    m = np.shape(dataMatrix)[0]  # 将样例个数赋值到m
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        # print(aggClassEst, '\n')
    return np.sign(aggClassEst)

if __name__ == '__main__':
    datMat = np.matrix([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    classifierArr = adaBoostTrainDS(datMat, classLabels, 30)
    print(adaClassify([[0,0], [5,5]], classifierArr))

