# -*- coding: utf-8 -*-
# @Time    : 2016/11/22 11:26
# @Author  : plr
# @Site    :
# @File    : maxent.py
# @Software: PyCharm

from collections import defaultdict
import math
import sys

class MaxEnt:
    def __init__(self):
        self._samples = []  # 样本集，元素是[y,x1,x2,...,xn]的元组
        self._Y = set([])  # 标签集合，相当于去重之后的y
        self._numXY = defaultdict(int)  # Key是(xi, yi)对， Value是count(xi, yi)
        self._N = 0  # 样本数量
        self._n = 0  # 特征对(xi, yi)总数量
        self._xyID = {}  # 对(x, y)对做的顺序编号(ID)，Key是(xi, yi)对，Value是ID
        self._C = 0  # 样本最大的特征数量，用于求参数时的迭代，见IIS原理说明
        self._ep_ = []  # 样本分布的特征期望值
        self._ep = []  # 模型分布的特征期望值
        self._w = []  # 对应n个特征的权值
        self._lastw = []  # 上一轮迭代的权值
        self._EPS = 0.01  # 判断是否收敛的阈值

    def load_data(self, filename):
        for line in open(filename, 'r'):
            sample = line.strip().split('\t')
            if len(sample) < 2:
                continue  # 至少标签+一个特征
            y = sample[0]
            X = sample[1:]
            self._samples.append(sample)
            self._Y.add(y)
            for x in set(X):
                self._numXY[(x, y)] = self._numXY.get((x, y), 0) + 1

    def _initparams(self):
        self._N = len(self._samples)  # 训练样本集长度，样本数量
        self._n = len(self._numXY)  # (xi, yi)的总数
        self._C = max([len(sample) - 1 for sample in self._samples])  # 训练样本集中特征数目的最大值
        self._w = [0.0] * self._n  # [0.0, 0.0,......,0.0]
        self._lastw = self._w[:]
        self._sample_ep()  # 特征经验值

    def _sample_ep(self):
        self._ep_ = [0.0] * self._n
        for i, xy in enumerate(self._numXY):
            self._ep_[i] = self._numXY[xy] * 1.0 / self._N  # 每个(xi, yi)的频率
            self._xyID[xy] = i  # 对(x, y)对做的顺序编号(ID)，Key是(xi, yi)对，Value是ID

    def _zx(self, X):
        ZX = 0.0
        for y in self._Y:
            sum1 = 0.0
            for x in X:
                if (x, y) in self._numXY:
                    sum1 += self._w[self._xyID[(x, y)]]
            ZX += math.exp(sum1)  # 归一化因子
        return ZX

    def _pyx(self, X):
        ZX = self._zx(X)
        results = []
        for y in self._Y:
            sum1 = 0.0
            for x in X:
                if (x, y) in self._numXY:
                    sum1 += self._w[self._xyID[(x, y)]]
            pyx = 1.0 * math.exp(sum1) / ZX
            results.append((y, pyx))
        return results

    def _model_ep(self):
        self._ep = [0.0] * self._n
        for sample in self._samples:
            X = sample[1:]
            pyx = self._pyx(X)
            for y, p in pyx:
                for x in X:
                    if (x, y) in self._numXY:
                        self._ep[self._xyID[(x, y)]] += p * 1.0 / self._N

    def _convergence(self):
        for w, lw in zip(self._w, self._lastw):
            if math.fabs(w - lw) >= self._EPS:
                return False
        return True

    def train(self, maxiter=1000):
        self._initparams()
        for i in range(0, maxiter):
            print "Iter: %d..." % i
            self._lastw = self._w[:]  # 保存上一轮权值
            self._model_ep()
            for i, w in enumerate(self._w):
                self._w[i] += 1.0 / self._C * math.log(self._ep_[i] / self._ep[i])
            print self._w
            if self._convergence():
                break

    def predict(self, input):
        X = input.strip().split("\t")
        prob = self._pyx(X)
        return prob

if __name__ == '__main__':
    maxent = MaxEnt()
    maxent.load_data('data.txt')
    maxent.train()
    # print maxent.predict("sunny\thot\thigh\tFALSE")
    # print maxent.predict("overcast\thot\thigh\tFALSE")
    # print maxent.predict("sunny\tcool\thigh\tTRUE")
    # sys.exit(0)
    print maxent._ep_
    print maxent._numXY