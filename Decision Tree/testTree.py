# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 17:39:05 2016

@author: plr
"""


from Tree import *


myDat, labels = createDataSet()
myTree = createTree(myDat, labels)
print myTree