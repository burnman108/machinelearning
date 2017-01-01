# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 9:15 2016

@auth: plr
@content: logistic regression img
"""
import numpy as np
from math import exp
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 100)
y = []
for num in x:
    item = 1/(1+exp(-num/0.1))
    y.append(item)

plt.figure()
plt.plot(x, y, 'r-')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.grid(True)
plt.show()
