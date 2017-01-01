# -*- coding: utf-8 -*-
# @Time    : 2016/11/23 17:57
# @Author  : plr
# @Site    : 
# @File    : bpnn.py
# @Software: PyCharm

import math
import numpy as np
import random

random.seed(1)

def rand(a, b):
    return (b - a) * random.random() + a

def makeMatrix(I, J):
    return np.zeros((I, J))

def randomizeMatrix(matrix, a, b):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = random.uniform(a, b)

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def dsigmoid(y):
    """
    求sigmoid函数的导数的方法
    :param y:
    :return:
    """
    return y * (1 - y)

class NN:
    def __init__(self, ni, nh, no):
        self.ni = ni + 1  # 输入单元数量，加入了偏置节点
        self.nh = nh + 1  # 隐藏单元数量，加入了偏置节点
        self.no = no      # 输出单元数量

        self.ai = [1.0] * self.ni  # 激活值，输出值
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        self.wi = makeMatrix(self.ni - 1, self.nh - 1)  # 权重矩阵——输入层到隐藏层
        self.wo = makeMatrix(self.nh - 1, self.no)  # 权重矩阵——隐藏层到输出层

        randomizeMatrix(self.wi, -0.2, 0.2)  # 将权重矩阵随机化，（初始随机）
        randomizeMatrix(self.wo, -2.0, 2.0)

        self.ci = makeMatrix(self.ni, self.nh)  # 权重矩阵的上次梯度（用来？？）
        self.co = makeMatrix(self.nh, self.no)

    def runNN(self, inputs):
        """
        这里是“前向传播”，通过输入值得出输出值，进行分类
        训练的时候计算误差也需要用到预测的输出值来计算误差
        :param inputs:
        :return:
        """
        if len(inputs) != self.ni - 1:  # 如果输入的特征向量的长度与对象的初始输入单元数量不一致的话，返回错误
            print 'incorrect number of inputs'
        for i in range(self.ni - 1):  # 赋值
            self.ai[i] = inputs[i]
        for j in range(self.nh - 1):
            sum = 0.0
            for i in range(self.ni - 1):
                sum += (self.ai[i] * self.wi[i][j])  # 通过权值矩阵 X 对应输入值，然后求和
            sum += 1.0                               # 这里是加上了固定的偏置值1.0
            self.ah[j] = sigmoid(sum)                # 通过sigmoid函数来得出隐藏层的输出值，最后一个元素为偏置节点，1.0
        for k in range(self.no):  # 因为输出单元无所谓的偏置节点，所以不用减1
            sum = 0.0
            for j in range(self.nh - 1):
                sum += (self.ah[j] * self.wo[j][k])
            sum += 1.0
            self.ao[k] = sigmoid(sum)
        return self.ao

    def backPropagate(self, targets, N, M):
        """
        后向神经传输
        输出层的输出误差（或称损失函数吧），其实就是所有实例对应的误差的平方和的一半，训练的目标就是最小化该误差。
        怎么最小化呢？看损失函数对参数的导数
        :param targets: 实例的类别
        :param N:
        :param M:
        :return:
        """

        # 计算输出层 deltas
        # dE/dw[j][k] = (t[k] - ao[k]) * s'( SUM( w[j][k]*ah[j] ) ) * ah[j]
        output_deltas = [0.0] * self.no
        for k in range(self.no):  # 在这个例子中这里输出单元数量其实就一个值，循环一次，k=0
            error = targets[k] - self.ao[k]
            output_deltas[k] = error * dsigmoid(self.ao[k])

        for j in range(self.nh - 1):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] += N * change + M * self.co[j][k]
                self.co[j][k] = change

        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh - 1):
            error = 0.0
            for k in range(self.no):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = error * dsigmoid(self.ah[j])

        for i in range(self.ni - 1):
            for j in range(self.nh - 1):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] += N * change + M * self.ci[i][j]
                self.ci[i][j] = change

        error = 0.0
        for k in range(len(targets)):
            error = 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def test(self, patterns):
        for p in patterns:
            inputs = p[0]
            print 'Inputs:', p[0], '-->', self.runNN(inputs), '\tTarget', p[1]

    def train(self, patterns, max_iterations=1000, N=0.5, M=0.1):
        """

        :param patterns: 实例
        :param max_iterations: 最大迭代次数
        :param N: 本次学习率
        :param M: 上次学习率
        :return:
        """
        for i in range(max_iterations):  # 最大迭代次数为1000次
            for p in patterns:
                inputs = p[0]  # 将输入训练集分为输入向量和目标向量
                targets = p[1]
                self.runNN(inputs)
                error = self.backPropagate(targets, N, M)
            if i % 50 == 0:
                print 'Combined error', error
        self.test(patterns)

    def weights(self):
        print 'Input weights:'
        for i in range(self.ni):
            print self.wi[i]
        print
        print 'Output weights:'
        for j in range(self.nh):
            print self.wo[j]
        print ''

def main():
    pat = [
        [[0, 0], [1]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
    ]
    myNN = NN(2, 2, 1)
    myNN.train(pat)

if __name__ == '__main__':
    main()