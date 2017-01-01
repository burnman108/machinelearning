# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 20:01:55 2016

@author: plr
"""

import copy
import matplotlib.pyplot as plt
from matplotlib import animation

training_set = [[(3, 3), 1], [(4, 3), 1], [(1, 1), -1]]
w = [0, 0]
b = 0
history = []

def cal(item):
    """
    判断实例点是否为误分类点所需要的求值
    """
    res = 0
    for i in range(len(item[0])):
        res += item[0][i] * w[i]
    res += b
    res *= item[1]
    return res
    
def update(item):
    """
    传入误分类点，并对参数w,b进行更新
    """
    global w, b, history # global---将变量定义为全局变量。可以通过定义为全局变量，实现在函数内部改变变量值
    w[0] += 1 * item[1] * item[0][0]
    w[1] += 1 * item[1] * item[0][1]
    b += 1 * item[1]
    print w, b
    history.append([copy.copy(w), b])
    
def check():
    """
    如果返回的flag是True，那么说明有误分类点，并且出现了更新，这个时候要重新遍历整个训练数据集合；
    当没有误分类点，flag是False，将此时的w和b两个参数打印出来
    """
    flag = False
    for item in training_set:
        if cal(item) <= 0:
            flag = True
            update(item)
    if not flag:
        print "RESULT: w: " + str(w) + " b: " + str(b)
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
        line.set_data([], []) # ??????
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
        return line, label # ?????
    
        
    def animate(i):
        global history, ax, line, label
        
        w = history[i][0]
        b = history[i][1]
        if w[1] == 0: return line, label
        x1 = -7
        y1 = -(b + w[0] * x1) / w[1]
        x2 = 7
        y2 = -(b + w[0] * x2) / w[1]
        line.set_data([x1, x2], [y1, y2])
        x1 = 0
        y1 = -(b + w[0] * x1) / w[1]
        label.set_text(history[i])
        label.set_position([x1, y1])
        return line, label
        
    print history
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=range(len(history)),
                                   interval=1000, blit=True)
    plt.show()
    anim.save('C:\Users\plr\Desktop\perceptron.gif', fps=2, writer='imagemagick')
    plt.close()
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    