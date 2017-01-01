# machinelearning
其中所有机器学习算法的学习与代码实现都是依据某大神[码农场][0]中所讲述的内容，本人只是实验搬运注释，书籍参考《机器学习实战》和李航老师的《统计学习方法》，在此谢过各位前辈对知识的无私分享！
[0]:http://www.hankcs.com/
## perceptron 感知机模型
感知机是二分类模型，输入实例的特征向量，输出实例的±类别。
### 运行环境
> python 2.7
### 结构
> + train2_1.py: 感知机学习算法的原始形式
+ train2_2.py: 感知机学习算法的对偶形式，对偶是指将w和b表示为测试数据i的线性组合形式，通过求解系数得到w和b

### 需要模块
> matplotlib, numpy

## k_Nearest Neighbor K近邻模型
给定一个训练数据集，对新的输入实例，在训练数据集中找到跟它最近的k个实例，根据这k个实例的类判断它自己的类（一般采用多数表决的方法）。
### 运行环境
> python 2.7
### 结构
> + kd_tree.py: 算法核心在于怎么快速搜索k个近邻出来
+ search_kdtree.py: 搜索跟二叉树一样来，是一个递归的过程。先找到目标点的插入位置，然后往上走，逐步用自己到目标点的距离画个超球体，用超球体圈住的点来更新最近邻（或k最近邻）。

## Naive Bayes classifier 朴素贝叶斯模型
朴素贝叶斯法是基于贝叶斯定理与特征条件独立假设的分类方法。训练的时候，学习输入输出的联合概率分布；分类的时候，利用贝叶斯定理计算后验概率最大的输出。
### 运行环境
> python 2.7
### 结构
> + Bayes.py: 一个基于贝叶斯文本分类器实现的简单情感极性分析器。

## Decision Tree 决策树模型
分类决策树模型是一种描述对实例进行分类的树形结构。决策树由结点和有向边组成。结点有两种类型：内部节点和叶节点，内部节点表示一个特征或属性，叶节点表示一个类。
### 运行环境
> python 2.7
### 结构
> + Tree.py: 依据《统计学习方法》中的例题，建立决策树，信息熵增益
+ testTree.py

## logistic regression 逻辑斯谛回归模型
根据现有数据对分类边界线建立回归公式，以此进行分类
### 运行环境
> python 2.7
### 结构
> + maxent.py: 最大熵模型
+ logisticRegression.py: 逻辑斯谛回归模型
+ logisticRegressionGif.py: 边界线形成的动态图
+ logistic_img.py: Sigmoid函数图像

### 需要模块
> numpy,matplotlib

## neural_network 神经网络传播模型
神经网络就是多个神经元的级联，上一级神经元的输出是下一级神经元的输入，而且信号在两级的两个神经元之间传播的时候需要乘上这两个神经元对应的权值。
### 运行环境
> python 3.5
### 结构
> + bpnn.py: 前向与后向神经网络传播模型
+ code_recognizer.py: IBM利用Neil Schemenauer的这一模块（旧版）做了一个识别代码语言的例子，我只是稍稍理解了一下

### 需要模块
> numpy

## SVM 支持向量机
支持向量机（support vector machines，SVM)是一种二类分类模型。它的基本模型是定义在特征空间上的间隔最大的线性分类器，间隔最大使它有别于感知机；支持向量机还包括核技巧，这使它成为实质上的非线性分类器。支持向量机的学习策略就是间隔最大化，可形式化为一个求解凸二次规划（convex quadratic programming）的问题，也等价于正则化的合页损失函数的最小化问题。支持向量机的学习算法是求解凸二次规划的最优化算法。
### 运行环境
> python 3.5
### 结构
> + kernel_svm.py: 
+ kernel_test.py: 
+ svmMLiA.py:
+ test.py:

### 需要模块
> numpy,matplotlib

## CART 树回归模型
使用二元切分来处理连续变量
### 运行环境
> python 3.5
### 结构
> + regTrees.py: 
+ test.py: 

### 需要模块
> numpy

## adaboost 提升方法
提升方法的思路是综合多个分类器，得到更准确的分类结果。  
### 运行环境
> python 3.5
### 结构
> + 7.6test.py: 
+ adaboost.py: 
+ boost.py: 

### 需要模块
> numpy

## regression 岭回归与逐步向前回归
  
### 运行环境
> python 3.5
### 结构
> + regression.py: 

### 需要模块
> numpy