---
title: 深度学习面试问题总结
date: 2020-06-08 21:14:22
categories: Deep Learning
---

### 常见面试问题



Disney+Hotstar 面试复盘





### 字节面试复盘

MLP代替点积的效果一定好？（推翻NCF的最新论文）

如何证明NN能学到特征交叉？

特征交叉的方式有哪些？

如何设计实验证明BN对ICS有效？

不同场景embedding怎么保证嵌入空间一致性？

（跨场景的vid embedding的使用方式）

召回离线评估

如何设计多样性评价指标？

召回阶段和排序阶段的样本构造差异？

随机负采样和曝光未点击负采样哪种方式效果更好？

TF模型的上线方式（不用tf-serving）

Xgb解决分类和回归问题的差异？多分类下节点的分裂方式？

线上线下不一致问题的坑有哪些？如何解决？



合并k个有序链表

字符串s匹配pattern串

python实现lr，kmeans

找到数组中第一个未出现的正整数 LC41



### 百度面试复盘

推荐系统的bias有哪些？

sampling bias

position bias

类似于example age， 使用**「这是一种非常通用的来消除机器学习系统固有偏差的方法。另外训练和在线区别对待也非常值得借鉴，一些单侧特征，训练的时候把bias影响特征加进去，在线的时候则置0」**，比如美团分享的消除广告position bias（位置给广告点击带来的偏差）也是类似的做法



曝光bias 马太效应

bias 和时长/分发量消偏平滑间的差异



Parameter Server的实现原理

分布式一致性 数据并行/模型并行

解释共轭分布

TDM召回

EE召回的abtest结果分析

​	baseline过低，召回占比提升对应的vv和完成率

​	冷启动类实验设计的问题，消费相同内容池，消费侧指标持平



python实现bandit算法

FM 优化后复杂度O(KN) VS FM的训练复杂度？

​	训练复杂度也为O(KN)

手写一个栈，以O(1)时间维护栈中最大值



内容：

模型中台，视频内容理解，强化学习，ps架构



阿里面试复盘

阿里妈妈

x属于(0,1)均匀分布，y属于(0,1)均匀分布，max(x,y)的期望

搜索推荐

wide&deep系列讲解

multi-head attention

self-attention

Hard-attention

Soft-attention

淘宝直播

一面

最大子数组和 注意下标



二面

softmax 溢出截断， 减最大值

如何融合内容侧特征

rerank listwise 模型

算法题 ip地址匹配

数学题 100个人比99次 赢家

双栏feed流和沉浸式 特征的区别



三面

1. multi-task model如何同时处理分类和回归的target

2. user embedding和其他sparse feature大小

相差较大，如何处理



快手面试复盘

用两个栈实现对队列

整数按位反转

按数组分段反转链表

python深拷贝 浅拷贝

java 重载 重写



腾讯面试复盘

wxg一面

x服从均匀分布(0,1)，y服从均匀分布(0,1)，求max(x, y)的期望

二叉树的最大路径和

wxg二面

一个从负数到0到正数的数组，求数组中不同的绝对值的个数

建堆的过程和时间复杂度





cdg一面

赛马问题

广告和推荐的区别

cdg二面

递归，括号展开





京东面试复盘

贝叶斯概率，真阳性的概率

abtest 的结果置信度

线性筛素数



微软面试复盘

第一场

序列化二叉树并恢复

第二场

二维矩阵和





##### 过拟合/欠拟合

过拟合的表现：

看loss

train loss 不断下降，test loss不断下降，说明网络仍在学习;

train loss 不断下降，test loss趋于不变，说明网络过拟合;

train loss 趋于不变，test loss不断下降，说明数据集100%有问题;

train loss 趋于不变，test loss趋于不变，说明学习遇到瓶颈，需要减小学习率或批量数目;

train loss 不断上升，test loss不断上升，说明网络结构设计不当，训练超参数设置不当，数据集经过清洗等问题

过拟合的解决方法：

降低模型复杂度，例如神经网络：减少网络层、神经元个数决策树；降低树的深度、剪枝

权值约束，增加正则化项，L1稀疏，L2权重衰减

Batch Normalization

landscape平滑，x归一化到0附近，更容易被激活

early stop 避免权重一直更新

dropout（dropout会导致输出结果随机，因此在测试时，根据概率计算的平均结果我们需要将激活函数乘以dropping概率，通常为0.5 进行伸缩作为最终结果，或在训练时的dropout mask步骤直接除以dropping概率）

使用ReLU激活函数替代Sigmoid，ReLU具备稀疏激活性，负半区梯度变为0

数据增强

集成学习

##### 梯度爆炸的解决方法

（w>1不断累积）

梯度裁剪 clip gradient

模型结构 resnet、lstm遗忘门

BN 使x的期望在0附近

参数的初始化

针对ReLU激活函数的神经元，其权重初始化通常使用随机数并使用$sqrt(2.0/n)$来平衡方差[^weightInit]，而bias通常直接初始化为零



##### 梯度消失

（w<1不断累积）

激活函数使用ReLU替代Sigmoid，ReLU的梯度x>0始终为1，但x<0时梯度为0神经元死亡，一定程度上可以调小学习率解决



##### ReLU取代Sigmoid的优点

避免梯度弥散

`ReLU` 的求导不涉及浮点运算，加速计算

负半区的输出为 0，稀疏激活，减少过拟合



##### Maxout

$f(x)=max(w_1^Tx+b_1, w_2^Tx + b_2)$
ReLU和Leaky ReLU都是这一函数的特例，例如ReLU对应w1,b1=0。

##### 正则化范数

L0 非0个数

L1 

距离的度量

无穷范数 x或y的最大值

##### BN

##### 优化算法

- 一阶方法
梯度下降

- 二阶方法
  Hessian 矩阵，计算Hessian矩阵可以反映坡度的陡缓
  牛顿法

  用Hessian矩阵替代学习率->自适应
  但计算量太大->近似算法

- 共轭牛顿法

- 伪牛顿法

##### 随机梯度下降（SGD）的“随机”性体现

SGD使用整个数据集的子集（mini-batch SGD）而不是完整的数据集迭代估计优化的最佳方向，因为整个数据集可能非常大，因而是随机的梯度下降并不能保证每一步都是最优方向。除SGD算法外，现在已有更多改进方案可用于计算权重的变化值进行权重优化，我们将在“优化方法”一节中进一步介绍。

##### SGD改进

动量：跳出局部最小值和鞍点；解决poor conditioning（当损失函数在一个方向上改变很快而在另一方向改变很慢，使用普通SGD会出现在变化敏感方向上的锯齿跳动，这种情况在高维下很常见。动量项将先前权重更新的一小部分添加到当前权重更新中。如果两次更新在同一方向则会加快收敛，而更新在不同方向上时则会平滑方差，从而能够尽快结束这种情况下的曲折前进Zigzagging）



自适应学习方法

Adagrad: 记录所有梯度的平方和，使得能够在较缓的维度上除以一个较小值进行加速而在较陡的维度上除以一个较大值从而减速。但由于梯度的平方和越来越大，步幅会越来越小，可能会停在鞍点处无法出来，因而Adagrad只适用于卷积层的学习。

RMSprop: RMSprop在Adagrad基础上进行小幅改动，对梯度的平方和进行衰减，衰减率（decay rate）通常设为0.9或0.99。实现了指数移动平均，类似于lstm的遗忘门。

Adam综合上述两种方法和动量



##### XGB

并行化的实现：特征值预排序

参数调优：

- 正则项 gamma调叶子结点个数，lambda调叶子结点取值的L2模平方
- early_stopping
- shrinkage，学习率控制拟合速度，单步生成树的权重
- 列采样，同随机森林



XGB VS GBDT

一阶 -> 二阶泰勒展开

为什么使用二阶泰勒展开

使用二阶泰勒展开是为了xgboost能够自定义loss function，只要这个损失函数可以求二阶导



特征预排序

稀疏感知：将缺失值归为一个分支

直方图



LGB VS XGB

leaf-wise VS level-wise level-wise方便并行计算每一层的分裂节点，提高了训练速度，但同时也因为每一level中增益较小的节点分裂增加了很多不必要的分裂；leaf-wise每次分裂增益最大的叶子节点，但容易过拟合，需要控制好depth

直方图+GOSS （Gradient-based One-Side Sampling）单边梯度抽样算法

对梯度较小的样本随机抽样，保留梯度较大的样本

直方图加速

叶节点的直方图可以通过父节点的直方图与兄弟节点的直方图相减的方式构建

https://cloud.tencent.com/developer/article/1534903



CAT VS XGB

target statistic



##### ROC AUC

横坐标假阳性率，纵坐标真阳性率

统计正样本P、负样本N个数，横坐标划分1/N，纵坐标划分1/P，然后从原点出发正样本向上，负样本向右



##### 卷积

尺寸计算：输出维度公式 (n + 2p - f) / s + 1

采用same padding填充行数为 f - n % s

参数数量：filter_size * filter_size * out_channel + out_channel(每个out_channel对应一个偏置量) 





















