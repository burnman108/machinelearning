# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:33:42 2016

@author: plr
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

training_set = np.array([[[3, 3], 1],
                         [[4, 3], 1],
                         [[1, 1], -1]
                         ])
a = np.zeros(len(training_set), np.float)
b = 0.0
Gram = None
y = np.array(training_set[:, 1])
x = np.empty((len(training_set), 2), np.float)
for i in range(len(training_set)):
    x[i] = training_set[i][0]
history = []

def cal_gram():
    """
    这里计算Gram矩阵， G=[xi·xj]
    """
    g = np.empty((len(training_set), len(training_set)), np.int)
    for i in range(len(training_set)):
        for j in range(len(training_set)):
            g[i][j] = np.dot(training_set[i][0], training_set[j][0])
    return g
    
def cal(i):
    """
    这里计算需用来判断是否为误分类点的值
    """
    global a, b, x, y, Gram
    Gram = cal_gram()
    res = np.dot(a * y, Gram[i])
    res = (res + b) * y[i]
    return res
    
def update(i):
    """
    这里history是w和b的合集
    """
    global a, b
    a[i] += 1
    b += 1 * y[i]
    history.append([np.dot(a * y, x), b])
    
def check():
    global a, b, x, y
    flag = False
    for i in range(len(training_set)):
        if cal(i) <= 0:
            flag = True
            update(i)
    if not flag:
        w = np.dot(a * y, x)
        print 'RESULT: w: ' + str(w) + ' b: ' + str(b)
    return flag
    
if __name__ == "__main__":
    for i in range(1000):
        if not check(): 
            break
        
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
    line, = ax.plot([], [], 'g', lw=2)
    label = ax.text([], [], '')
    
    def init():
        line.set_data([], [])
        x, y, x_, y_ = [], [], [], []
        for p in training_set:
            if p[1] > 0:
                x.append(p[0][0])
                y.append(p[0][1])
            else:
                x_.append(p[0][0])
                y_.append(p[0][1])
        
        plt.plot(x, y, 'bo', x_, y_, 'rx')
        plt.axis([-6, 6, -6, 6])
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Perceptron Algorithm')
        return line, label
        
    def animate(i):
        global history, ax, line, label
        
        w = history[i][0]
        b = history[i][1]
        if w[1] == 0: return line, label
        x1 = -7.0
        y1 = -(b + w[0] * x1) / w[1]
        x2 = 7.0
        y2 = -(b + w[0] * x2) / w[1]
        line.set_data([x1, x2], [y1, y2])
        x1 = 0.0
        y1 = -(b + w[0] * x1) / w[1]
        label.set_text(str(history[i][0]) + ' ' + str(b))
        label.set_position([x1, y1])
        return line, label
    
    print history
        
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=range(len(history)),
                                   interval=1000, repeat=True, blit=True)
    plt.show()
    anim.save('C:\Users\plr\Desktop\perceptron.gif', fps=2, writer='imagemagick')
    
    
    
    
    
    
    
    
    