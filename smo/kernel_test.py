# -*- coding: utf-8 -*-
# @Time    : 2016/12/14 17:13
# @Author  : plr
# @Site    : 
# @File    : kernel_test.py
# @Software: PyCharm

from kernel_svm import *

def testRbf(k1=1.3):
    filepath = 'Ch06/'
    dataArr, labelArr = loadDataSet(filepath + 'testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print('There are %d Support Vectors' % np.shape(sVs)[0])
    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print('the training error rate is: %f' % (float(errorCount)/m))

    dataArr_test, labelArr_test = loadDataSet(filepath + 'testSetRBF2.txt')
    errorCount_test = 0
    datMat_test = np.mat(dataArr_test)
    labelMat_test = np.mat(labelArr_test)
    m, n = np.shape(dataArr_test)
    for i in range(m):
        kernelEval_test = kernelTrans(sVs, datMat_test[i, :], ('rbf', k1))
        predict_test = kernelEval_test.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict_test) != np.sign(labelArr_test[i]):
            errorCount_test += 1
    print('the test error rate is: %f' % (float(errorCount_test)/m))

if __name__ =='__main__':
    testRbf()