---
title: 多任务学习在推荐系统中的应用
date: 2020-06-01 11:56:48
categories: RecSys
---

##### Hard参数共享

所有任务共享底层参数，shared-bottom

![image-20200720162726147](/Users/bytedance/Library/Application Support/typora-user-images/image-20200720162726147.png)

![image-20200720161902483](/Users/bytedance/Library/Application Support/typora-user-images/image-20200720161902483.png)

优点：减少过拟合风险

缺点：效果可能受到任务差异和数据分布带来的影响

##### Soft参数共享

不同任务的参数间增加约束控制任务的相似性，例如增加L2正则，trace norm

优点：在任务差异会影响公共参数的情况下对最终效果有提升

缺点：模型增加了参数量所以需要更大的数据量来训练模型，而且模型更复杂并不利于在真实生产环境中实际部署使用

##### MOE(**Mixture-of-Experts**)模型

所有任务共享底层若干专家子网络，不同专家网络学习不同角度知识，上层利用门控结构进行加权融合。

![image-20200720161846247](/Users/bytedance/Library/Application Support/typora-user-images/image-20200720161846247.png)

![image-20200720161947020](/Users/bytedance/Library/Application Support/typora-user-images/image-20200720161947020.png)

n个expert network，g是组合experts结果的gating network。g产生n个experts上的概率分布，最终的输出是**所有experts的带权加和**。

MoE可作为一个基本组成单元，堆叠在大网络中

##### MMOE模型

**shared-bottom网络中的函数f替换成MoE层**，不同上层任务使用不同门控进行专家子网络的融合

 ![image-20200720163230408](/Users/bytedance/Library/Application Support/typora-user-images/image-20200720163230408.png)

优点：不同任务的gating networks可以学习到不同的组合experts的模式，可以捕捉到任务的相关性和区别。



分层萃取共享，渐进式分离



