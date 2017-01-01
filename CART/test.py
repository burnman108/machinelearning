# -*- coding: utf-8 -*-
# @Time    : 2016/11/30 11:52
# @Author  : plr
# @Site    : 
# @File    : test.py
# @Software: PyCharm

from regTrees import *
import matplotlib.pyplot as plt

# filePath = 'C:/Users/plr/Desktop/'
filePath = 'Ch09/'

def ex00_regtree():
    myDat = loadDataSet(filePath + 'ex0.txt')
    myMat = np.mat(myDat)
    regtree = creatTree(myMat)
    print(regtree)

def ex00_img():
    myDat = loadDataSet(filePath + 'bikeSpeedVsIq_test.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(np.array(myDat)[:, 0], np.array(myDat)[:, 1])
    plt.show()

def ex2_reg_prune():
    myDat = loadDataSet(filePath + 'ex2.txt')
    myMat = np.mat(myDat)
    myTree = creatTree(myMat)
    myDatTest = loadDataSet(filePath + 'ex2test.txt')
    myMatTest = np.mat(myDatTest)
    pruneTree = prune(myTree, myMatTest)
    print(pruneTree)

def linearTree():
    myDat = loadDataSet(filePath + 'exp2.txt')
    myMat = np.mat(myDat)
    myTree = creatTree(myMat, leafType=modelLeaf, errType=modelErr, ops=(1, 10))
    print(myTree)

def bikespeedvsiq_tree(leafType, errType):
    # trainMat = np.mat(loadDataSet(filePath + 'data.csv'))
    trainMat = np.mat(loadDataSet(filePath + 'bikeSpeedVsIq_train.txt'))
    myTree = creatTree(trainMat, leafType, errType, ops=(1,20))
    print(myTree)
    return myTree

def bikespeedvsiq_corrcoef(mytree, modelEval):
    # testMat = np.mat(loadDataSet(filePath + 'datatest.csv'))
    testMat = np.mat(loadDataSet(filePath + 'bikeSpeedVsIq_test.txt'))
    yHat = createForeCast(mytree, testMat[:, 0], modelEval)
    r2 = np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0,1]
    return r2

if __name__ == '__main__':
    reg_tree = bikespeedvsiq_tree(regLeaf, regErr)
    reg_r2 = bikespeedvsiq_corrcoef(reg_tree, regTreeEval)
    print('reg: the square of r is  ' + str(reg_r2))

    model_tree = bikespeedvsiq_tree(modelLeaf, modelErr)
    model_r2 = bikespeedvsiq_corrcoef(model_tree, modelTreeEval)
    print('model: the square of r is  ' + str(model_r2))
    # linearTree()

