---
title: 知识图谱在推荐系统的应用
date: 2020-05-28 17:09:09
categories: RecSys
---

知识图谱引入推荐系统的两类方式：

#### 基于特征的方法

利用知识图谱得到实体、关系的embedding

学习方式：

- 分别学习

DKN（Deep Knowledge-aware Network）

首先通过实体链接将新闻标题文本中实体链接到知识图谱，获得实体的实体特征和上下文实体特征（所有一跳邻居节点的实体上下文特征的均值）。

将新闻标题词向量、实体向量、实体上下文向量作为多通道使用CNN融合得到candidate的embedding；用户历史兴趣通过attention机制结合candidate embedding学习不同的权重作为user embedding；Candidate embedding和user embedding经过concat通过mlp预测点击率。



- 交替学习（多任务学习）

MKR（Multi-task ）

推荐网络使用user和item特征预测点击率；知识图谱网络使用三元组的头节点和关系表示作为输入预测尾节点；两者通过交叉特征共享单元链接，然后分别固定一侧网络参数交替训练。



- 联合学习（end2end）

CKE（collaborative knowledge base embedding）

知识图谱实体表示和图像表示、文本表示三类目标函数与协同过滤结合得到联合损失函数训练

Ripple Network

以用户历史记录为中心在图谱上扩散，并在扩散过程中衰减得到item embedding和user embedding，使用联合损失函数训练

KGAT

GCN建模，邻居节点使用注意力机制融合，使用联合损失函数训练



#### 基于结构的方法

利用知识图谱使用bfs、dfs得到多跳关联实体



