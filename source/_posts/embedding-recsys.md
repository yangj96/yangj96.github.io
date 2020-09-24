---
title: Embedding在推荐系统中的应用
date: 2020-06-05 12:04:26
categories: RecSys
---



#### Embedding在推荐系统中的应用

![image-20200525165327642](/Users/jingy/Library/Application Support/typora-user-images/image-20200525165327642.png)

https://mp.weixin.qq.com/s/8Mx8CznNBlJ6adlwXbcHXQ

embedding相当于one hot的平滑，one hot相当于embedding的max-pooling

embedding通常取神经网络倒数第二层的参数权重

embedding向量单独训练还是端到端训练？

单独训练的embedding训练样本大，参数学习充分；

端到端训练的embedding参数多，收敛速度慢，数据量少较难充分训练





embedding 静态表征

word2vec, fasttext, glove

embedding 动态表征

elmo 双向LSTM抽取特征

gpt 单向语言模型，transformer抽取特征，输入输出attention，不受长度限制易并行

bert 双向语言模型，transformer抽取特征，其他同上



i2i召回

tag2vec, 取文章的tag的fasttext生成的embeding等权重相加，faiss取相似，按相似度截断再利用热度，ctr等加权排序

item2vec，取文章id向量，取文章作者向量

loc2vec，地名对应向量

title2vec，lstm训练标题向量

doc2vec，bert计算文章文本向量

entity2vec，tranE生成实体向量



u2i召回

user2vec 用户tag向量和文章tag向量（多个tag的向量进行加权和，归一化）

对所有用户向量进行minibatch-kmeans聚为400簇（5k users per），簇内计算相似用户，写入天级redis，相似用户topn文章候选集去重计算相似度得分，根据相似度，热度，新鲜度，质量分，ctr加权形成倒排，写入天级redis



DSSM

crossTag，用户tag按类别统计，每个类别取k个tag，m组tag分别和用户tag向量计算相似度



分群召回

1. 簇召回：所有用户的tag向量或用户行为lstm向量用聚类算法（如minibatch-kmeans）聚成若干个簇（比如500个，根据肘点法确定），然后簇内做实时CF

   - 增量聚类，一段时间内保持聚类中心不变，新增数据点选择现有最近距离中心，业务低峰时期全量更新聚类中心
   - 动态规则聚类，选择用户画像兴趣点组合作为兴趣标签，保留用户数超过阈值的兴趣标签作为聚类中心

   RFM模型用户分群

2. 多用户融合作为群画像

![image-20200525230021916](/Users/jingy/Library/Application Support/typora-user-images/image-20200525230021916.png)



![image-20200525230100683](/Users/jingy/Library/Application Support/typora-user-images/image-20200525230100683.png)



![image-20200525230358968](/Users/jingy/Library/Application Support/typora-user-images/image-20200525230358968.png)



Embedding的问题和优化



![image-20200525230436683](/Users/jingy/Library/Application Support/typora-user-images/image-20200525230436683.png)



![image-20200525230514339](/Users/jingy/Library/Application Support/typora-user-images/image-20200525230514339.png)



总结

![image-20200525230525617](/Users/jingy/Library/Application Support/typora-user-images/image-20200525230525617.png)



