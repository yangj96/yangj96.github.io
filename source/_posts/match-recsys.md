---
title: 推荐系统召回算法总结
date: 2020-05-28 22:23:24
categories: RecSys
---

#### 通用召回（非个性化）

全局热门召回

热门即以全局后验统计概率估计用户点击的概率



热点事件bert召回

实时爬取新闻话题并对其进行bert向量预存，使用DBSCAN进行密度聚类，每个聚类中心作为事件向量根据热度时间等计算热度值；在item的bert向量中检索相似向量召回



#### CB召回(content-based)

##### 属性标签召回

tag、cate倒排

倒排优化：

问题：1. 点击率倒排，头部标题党；完成率倒排，头部短视频 2.后验指标置信度 3.倒排的优化目标与线上目标不一致，例如离线统计窗口的后验值和线上排序的状态并不契合，新的内容很难进入倒排的头部等

解决方案：

1.资源的后验分布与资源本身的类型有关，因此将视频根据时长和类别组合分桶，在桶内做z--score归一化，将完成率的分布收缩到准正态分布

2.置信阈值的判定

1）用分发趋势预估阈值的值->直接预估完成率，facebook热度预估模型

2）优化方差，在视频分发曲线上确定阈值点A，并在阈值点后取B点计算两点差值，累计所有人的差值和，求出总和最小的点作为阈值点

或CTR平滑，解决曝光量小导致的ctr置信度低的问题

1）手工平滑

将曝光分段，每段取24小时内的数据min-max归一化

2）指数平滑

$I_j = I_j, j =1$

$I_j = \beta I_j + (1 - \beta)I_{j-1}, j > 1$

$I_N = \beta I_N + (1 - \beta) I_{N-1} = \beta I_N + \beta(1-\beta)I_{N-1} + ... + \beta(1-\beta)^j I_{N-j} + ... + \beta(1-\beta)^{N-1}I_1$

对曝光<1000的数据不召回

3）Wilson平滑

一般效果最好

4）贝叶斯平滑

$ctr = (C_i + \alpha) / (I_i + \alpha + \beta) $

$\alpha$和$\beta$采用矩估计

总体而言，随着曝光次数的增加，对应的点击率是下降的，因此点击率需要考虑分发带来的衰减；不同类目下点击的行为也应有不同的权重，小众类目的点击比大众类目更有意义，tfidf；时间衰减可以修正处在置信阈值边界的兴趣点，即着时间推移没有新增兴趣表达的误判行为会逐渐衰减到阈值以下；对某一类目连续发生大量展现未点击的行为需要对其权重作降权。



3.倒排的优化 - 从统计到函数的趋势

使用视频信息构建模型，以倒排的目标（实际的完成率/点击率作为优化目标，以及时拟合线上的状况同时增加一定的泛化能力

聚焦置信点击率的后验值能够最大化提升推荐效果



触发tag、cate的优化：

用户不同的互动行为差异化权重



多term召回/多维度：

考虑内容的多维度属性，例如将内容的多个tag综合推荐而不是针对每个tag单独做倒排，即将用户的tag标签和内容的tag标签构造为sentence进行训练。



序列建模同时又会引入不同行为的序列权重的差异->更改word2vec的label生成逻辑和样本构造

word2vec优化
增量skip-gram 负采样



##### 基于内容的MF召回

基于内容的MF召回是通过评分矩阵经过矩阵分解和最小化目标函数学习得到用户和item的向量（对比：传统基于行为过滤的协同过滤方法是基于共现矩阵）



#### CF召回（协同过滤）

##### Memory-based CF

Item-based CF

划分窗口利用用户的点击历史，以窗口内内容id出现的条件概率作为后验预估

问题： 热点视频和大多数视频的共现概率都很高

解决：基于共现统计的cf在分母上乘视频的分发量抵消降权；item2vec本身集成word2vec高频负采样，这和数据的整体分布相关，如果参数设置不合理，分布和参数空间不match会折损训练效果；还有完成率建模带来的时长的bias，通常需要时长调权，归一化等



User-based CF

用户的相似度度量

jaccard对大数据量而言计算困难->simHash

用户list2list建模，利用孪生网络（适合稀疏分类问题）以用户是否为同一个用户作为label提供监督，在线召回使用bagging，不同用户间的差异化权重可以利用attention机制等学习。



##### Model-based CF

基于MF、SVD、pLSA、LDA的CF



###### DSSM/DeepCF/MatchNet

DSSM双塔，用户点击做正样本，全局随机采样负样本

如果使用展现未点击作负样本，召回结果会偏向热门，为什么？



DSSM优化

- user和item特征交叉

uid的embedding直接连到瓶颈层把doc向量作增维，强化uid记忆性。为什么不连vid特征？

- 增加attention、FM等结构增强模型表达能力，增加side-info

在user侧增加attention，学习用户session内不同类型行为的权重

在item侧增加side info以强化长尾item特征缓解高热和attention，学习不同side info属性信息的主次

融合FM特征

- pair-wise双塔提升模型预测能力
- multi-view DSSM 结合多业务信息丰富用户表达
- 多塔多目标，结合多业务指标



CF的问题：

Icf时效性，新视频不容易出，即i2i 如果使用共现概率会存在新item的冷启动问题，但如果i2i基于内容本身的相似度，则可以很容易召回新的item。同理ucf会存在新用户和新item的冷启动问题。

解决方案：

icf中引入泛化特征side info，利用side info的共现信息。

lookalike 用户点击的一系列视频，其中某一个新视频可以通过邻近点击视频向量加权平均近似求得



###### EGES

在行为向量基础上增加side info构成语义行为向量，以解决用户行为数量少的item的冷启动

用concat、avg pooling、self-attention和transformer encoder的方法融合side info



**Base Graph Embedding（BGE）**
 user 的行为序列构建网络结构，并将网络定义为有向有权图。其中：根据行为的时间间隔，将一个 user 的行为序列分割为多个session进行deepwalk
**Graph Embedding with Side Information（GES）**
该方案增加 item 的额外信息（例如category, brand, price等）丰富 item 表征力度。根据 EGES的算法框架可知：
（1）item 和 side information（例如category, brand, price等） 的 Embedding 是通过 word2vec 算法一起训练得到的。如果分开训练，得到的item_embedding和category_embedding（brand_embedding， price_embedding）不在一个向量空间中，做运算无意义。
即：通过 DeepWalk 方案得到 item 的游走序列，同时得到对应的category（brand, price）序列。然后将所有序列数据放到word2vec模型中进行训练。
（2）针对每个 item，将得到：item_embedding，category_embedding，brand_embedding，price_embedding 等 embedding 信息。将这些 embedding 信息求均值来表示该 item
**Enhanced Graph Embedding with Side Information（EGES）**
组合表示 item_embedding 时，对 item 和 side information（例如category, brand, price等）的embedding施加不同的权重，用改进的word2vec算法（Weighted Skip-Gram）确定模型的参数。
Sparse Features代表 item 和 side information 的ID信息；
Dense Embeddings 表示 item 和 side information 的 embedding 信息；
alpha分别代表 item 和 side information 的 embedding 权重；
Sampled Softmax Classifier中的N代表采样的负样本（见论文中的Algorithm 2 Weighted Skip-Gram描述的第8行），P代表正样本（某个item周边上下n个item均为正样本，在模型中表示时不区分远近）；



###### Youtube DNN

**训练集**
对每个用户提取等数量的训练样本，防止高度活跃用户对于loss的过度影响。
**测试集**
YouTube为什么不采用经典的随机留一法（random holdout），而是一定要把用户最近的一次观看行为作为测试集，避免引入future information，产生与事实不符的数据穿越。
video embedding的时候，要直接把大量长尾的video直接用0向量代替
**输入处理：**
样本通过spark处理为tfrecord，传至ceph，输入采用Dataset多线程读取

**特征处理：**

多值离散特征，例如历史播放列表、搜索词表：获取每个词的embedding平均求和得到固定长度隐向量

单值离散特征，例如地理位置，获取固定长度embedding向量

单值连续特征：年龄、性别：年龄构造平方、开方项

**模型结构：**

将上述三类特征concat，传入三层全连接层，使用ReLU函数，输入固定维度的user向量。

**损失函数：**

损失函数为softmax多分类交叉熵。将一次训练样本的播放vid作为正样本，多分类问题的类目个数为vid的数量，最后一层输出的user向量和vid向量的内积加上偏置表示user对该vid的感兴趣程度$z_i = u ·I_i + b_i$，然后通过softmax函数将所有感兴趣程度转化为概率：

$p_i = \frac{e^{z_i}}{\Sigma_j e^{z_j} }$，利用交叉熵得到损失函数：$L=- \Sigma_i y_i·log(p_i)$，其中$y_i$是一个vid数量维度的one-hot向量，只有被播放的视频对应一维是1，其余为0，可以采用负采样降低0的维度数量。

**输出：**

DNN 最后一层为user的embedding表示，softmax的权重为item的embedding表示

**线上serving：**

使用faiss进行topk search

Faiss支持l2欧式距离和IP内积的方式，但基于内积的方式需要user和item向量均是归一化向量，但youtube DNN模型训练时不进行向量的l2归一化，因此需要使用在原向量中增加一维，将内积问题转换为欧式距离问题来解决

MIP(maximum inner product)2NN(nearest neighbor)https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf

**样本不平衡问题**

使用Focal Loss

clip gradient 可以缓解梯度爆炸 / 防止出现网络训练中出现NaN

Triplet loss通常配合OHEM(online hard example mining)

权值共享： 多个特征的同一概念embedding的look_up table共享

基于牛顿冷却定律的时间衰减

**负采样**

1. tf.sample_softmax_loss调用log_uniform_candidate_sampler负采样，数值越小被采样的概率越大，因此对于vid，按照其在所有用户播放历史中出现的次数降序排列，标号越小出现次数越大，0保留用于替换确实值。但这样也会导致头部效应。可以采用word2vec论文或源码中的高频采样公式构造样本：

   $ran = (\sqrt{\frac{x}{0.001}} + 1)*(0.001/x)$，使高频词被保留的概率更小。

2. NEG（negative）采样代替softmax，tf.nn.learned_unigram_candidate_sampler。噪声对比估计(NCE,Noise-Contrastive Estimation)，它将点击样本的softmax变为多个二分类logistic问题。由于softmax具有sum-to-one的性质，最大化点击样本的概率必然能导致最小化非点击样本的概率，而二分类logistic则没有这个性质，所以需要人为指定负样本，NCE就是一种**根据热度随机选取负样本**的技术，通过优化NCE损失，可以快速地训练得到模型的所有参数。在实践中，我们采用了TensorFlow提供的函数tf.nn.nce_loss去做候选采样并计算NCE损失。

**user embedding和item embedding如何保证在同一个空间**
和dssm类似，都是通过内积限制两个embedding在相同空间，在CF中可以通过矩阵分解得到user和vedio的向量表示，这里最后的softmax相当于广义矩阵分解，模型最后一层隐层就是user embedding，通过u*v得到vedio的概率，v就是vedio embedding，只不过这里用来作为softmax层的权重；最后一层输出的user向量和vid向量的内积加上偏置表示user对该vid的感兴趣程度，然后通过softmax函数将所有感兴趣程度转化为概率

**和排序模型的差异：**
引入另一套DNN作为ranking model的目的就是引入更多描述视频、用户以及二者之间关系的特征，例如language embedding: 用户语言的embedding和当前视频语言的embedding
time since last watch: 自上次观看同channel视频的时间

previous impressions: 该视频已经被曝光给该用户的次数

example age，倾向于新视频，sample log距离当前时间，预测时置为0保证预测时处于训练的最后一刻
此外，ranking model不采用经典的logistic regression当作输出层，而是采用了weighted logistic regression



#### 召回的评估问题

相关性高能否代表召回效果有提升？

相关性提升到80以上后应该更关注效果

召回占比的提升

排序时这一路召回q值的提升还要关注其分布，是否均值变小方差变大，以及这路召回的uniq占比



MRR指标











