# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:33:09 2016

@author: plr
"""

from math import log, exp


class LaplaceEstimate(object):
    """
    这里的“一种标签”只是针对本例子中的声明来说的
    """
    def __init__(self):
        self.d = {} # 一种标签下[词-词频]的map  例如 neg
        self.total = 0.0 # 一种标签下全部词的词频， 例如 neg
        self.none = 1 # 当一个词不存在的时候，它的词频（等于0+1）
        
    def exists(self, key):
        return key in self.d
        
    def get(self, key):
        if not self.exists(key):
            return False, self.none
        return True, self.d[key]

    def getprob(self, key):
        """
        估计先验概率
        """
        return float(self.get(key)[1]) / self.total
        
    def getsum(self):
        """
        返回一种标签下的词频总数
        """
        return self.total
    
    def add(self, key, value):
        self.total += value
        if not self.exists(key):
            self.d[key] = 1
            self.total += 1
        self.d[key] += value
        
        
class Bayes(object):
    
    def __init__(self):
        self.d = {} # [标签， 概率] map
        self.total = 0 # 全部词频
    
    def train(self, data):
        for d in data: # d是[[词链表]，标签]
            c = d[1] # c是分类标签
            if c not in self.d:
                self.d[c] = LaplaceEstimate() # 为每种标签建立类LaplaceEstimate()
            for word in d[0]:
                self.d[c].add(word, 1) # 统计一种标签下的词频
        self.total = sum(map(lambda x: self.d[x].getsum(), self.d.keys())) # 将所有标签下的词频总数求和
    
    def classify(self, x):
        """
        1、x确定后不同类别的后验概率的对数，后验概率的和不是1，但是可以根据他们之
        间的大小来判断测试数据的类别
        2、neg类的值为log(A)，pos类的值为log(B)，A和B分别是x确定下不同类别的后验概率，
        此例中B>A，所以x是pos类，第2部分我们计算x属于pos的概率是多少，并通过这个概
        率大小来判断x的类别，当然不是B了，
        计算方法为  now = exp(log(A) - log(A)) + exp(log(B) - log(A))
                   now = (A + B) / A
        所以        now = 1 / now = A / (A + B)   此时为x属于neg类的概率大小
        同理可得x属于pos类的概率大小，比较两个值的大小，最终得出x属于的类别
        """
        # --------------1---------------
        tmp = {}
        for c in self.d:
            tmp[c] = log(self.d[c].getsum()) - log(self.total)
            for word in x:
                tmp[c] += log(self.d[c].getprob(word))
        #--------------2----------------        
        ret, prob = 0, 0
        for c in self.d:
            now = 0
            try:
                for otherc in self.d:
                    now += exp(tmp[otherc] - tmp[c])
                now = 1 / now
            except OverflowError:
                now = 0
            if now > prob:
                ret, prob = c, now
        return (ret, prob)
                    
class Sentiment(object):
    
    def __init__(self):
        self.classifier = Bayes()
        
    def segment(self, sent):
        words = sent.split(' ')
        return words
        
    def train(self, neg_docs, pos_docs):
        data = []
        for sent in neg_docs:
            data.append([self.segment(sent), u'neg'])
        for sent in pos_docs:
            data.append([self.segment(sent), u'pos'])
        self.classifier.train(data)
        
    def classify(self, sent):
        return self.classifier.classify(self.segment(sent))
        
s = Sentiment()
s.train(['糟糕'.decode('utf-8').encode('gbk'), '好 差劲'.decode('utf-8').encode('gbk')], 
         ['优秀'.decode('utf-8').encode('gbk'), '很 好'.decode('utf-8').encode('gbk')])

print s.classify('好 优秀'.decode('utf-8').encode('gbk'))
        
