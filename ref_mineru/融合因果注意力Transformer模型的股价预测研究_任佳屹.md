# 融合因果注意力Transformer模型的股价预测研究

任佳屹，王爱银

新疆财经大学 统计与数据科学学院，乌鲁木齐 830012摘 要：股票价格预测是金融研究和量化投资共同关注的重点话题，近年来利用深度学习技术揭示股票市场的行情规律成为研究热点。现有股票价格预测深度学习模型多数仅研究时间点数据，这种结构上的缺陷导致其不能反映出特征因子的累积作用对股价的影响。针对此，通过重新设计模型处理时间序列数据，提出一种基于Transformer的股票价格预测模型Stockformer。它通过因果自注意力机制挖掘股票价格与特征因子之间的时序依赖关系，采用趋势增强模块为模型提供序列的趋势特征，同时利用编码器的特定输入为预测提供输入特征的直接先验信息。实验结果表明，Stockformer的预测精度显著优于已有深度学习模型，且相较经典Transformer预测模型的平均绝对误差和均方根误差分别降低了 $2 3 . 2 \%$ 和 $2 5 . 7 \%$ ，预测值与真实值更为拟合；通过消融实验分别评估了Stockformer的因果注意力机制、时序特征提取手段以及特定的模型输入的效果及必要性，验证了所提模型的优越性及普适性。

关键词：股票价格预测；时间序列；深度学习；Transformer；注意力机制文献标志码：A 中图分类号：TP181 doi：10.3778/j.issn.1002-8331.2212-0127

# Causal Attention Transformer Model for Stock Price Prediction

REN Jiayi, WANG Aiyin

School of Statistics and Data Science, Xinjiang University of Finance and Economics, Urumqi 830012, China

Abstract：Stock price prediction is a key topic of common concern for financial research and quantitative investment, and the use of deep learning techniques to reveal the market patterns of stock markets has become a hot research topic in recent years. Most of the existing deep learning models for stock price prediction only study point- in- time data, and this structural shortcoming causes them to fail to reflect the impact of the cumulative effect of feature factors on stock prices. To address this, a Transformer- based stock price forecasting model Stockformer is proposed by redesigning the model to handle time series data. it mines the time-series dependence between stock prices and feature factors through a causal self-attentiveness mechanism, employs a trend enhancement module to provide the model with the trend features of the series, and uses encoder-specific inputs to provide the prediction direct a priori information of the input features. The experimental results show that the prediction accuracy of Stockformer is significantly better than that of existing deep learning models, and the average absolute error and root mean square error are reduced by $2 3 . 2 \%$ and $2 5 . 7 \%$ , respectively, compared with the classical Transformer prediction model, and the predicted values are more suitable to the real values; and the ablation experiments are conducted to evaluate the causal attention mechanism of Stockformer, the effects of the time-series feature extraction means and specific model inputs are evaluated by ablation experiments respectively, and the superiority and generalizability of the proposed model are verified.

Key words：stock price prediction; time series; deep learning; Transformer; attention mechanism

随着国民投资理财意识的不断提高，股票市场作为市场经济的重要组成部分，成为国民经济发展的“风向标”。市场行为涵盖一切信息，股价的涨跌行情一定程度上受到政治、经济、社会以及心理等多方面因素影响，具有高度的随机性及波动性，投资股票成为一项高风险、高回报的经济行为。投资者希望通过对股市未来趋势或价格的预测，多头排列时加仓盈利，空头排列时减仓止损，以防范风险并最大程度获取收益。如果能够对股价变化作出较为及时、准确的判断，也就能够提前作出相应的应对措施，因此，对股票价格变化趋势的预测得到学术界及商业界的广泛关注。股价数据时间序列的高维度、非平稳、非线性以及非参数等特点对其预测方法提出了更高的要求，同时带来了巨大的挑战。

近年来，计算机科学研究日益精进，针对股票价格的预测方法取得大量研究成果，由早期的基于统计指标的方法[1]、时间序列分析[2]等计量经济学方法，再到如今的机器学习[3]、深度学习[4]等人工智能技术，这些方法通过大量历史数据进行建模及训练，极大提高了预测精度，具有重要的理论及实践意义，尤其是随着深度学习应用于诸多不同领域的研究并取得显著成就，利用深度学习方法进行股票预测成为研究热点。

早期的计量经济学方法无法处理股票价格这种复杂的非线性时间序列，而支持向量回归机（support vactorregression，SVR）、神经网络以及决策树等机器学习方法能够直接挖掘数据集中最具价值的信息并具有强大的学习能力，处理高维、非线性数据时具有显著优势。文献[5]采用人工生态系统优化算法优化SVR参数并提高了 SVR 的预测精度；文献[6]横向对比了美国 NASDAQ市场中不同前馈神经网络的股价预测效果；文献[7]基于情感分析量化投资者的情感意向并将其作为数据特征，采用生成对抗网络预测股价走势。

传统神经网络受其拓扑结构影响，泛化能力较弱，迭代过程中容易陷入局部最优。随着大数据时代的高速发展，深度学习技术以其更为强大的学习能力及特征提取能力取得了突破性进展，主要包括卷积神经网络（convolutional neural network，CNN）、循 环 神 经 网 络（recurrent neural network，RNN）以及长短期记忆网络（long short term memory network，LSTM）等。文献[8]针对股票市场通过重新设计了CNN的网络结构以更为适用于不同层级下分别提取日内、日间特征，但CNN通过不断堆叠卷积层以提取信息的这种计算机制可能导致计算量大幅增加，网络收敛性不强。文献[9]基于RNN和矩阵分解分别得到股票的动态和静态特征并利用多层感知机模型预测股票排名。虽然RNN能够有效处理时序数据，但其训练过程较为复杂，难以寻找到某个合适的初始值令其收敛，并且RNN的训练过程中历史数据集对预测输出的影响可能随着序列长度的增加而削弱，产生梯度消失问题。LSTM作为一种特殊的RNN，通过引入门机制重新设计RNN的网络结构，以有效解决长序列训练过程中的梯度消失及爆炸问题。文献[10]通过构建 CNN 和 LSTM 的混合模型分别预测了股票指数的波动方向及涨跌数值，验证了 CNN-LSTM预测模型的普适性；文献[11]利用自适应学习的粒子群算法对LSTM的关键参数进行寻优，缓解了股票数据特征和网络拓扑结构不匹配的问题；文献[12]提出一种融合经验模态分解和嵌入时间注意力网络的股价预测方法，进一步优化了预测准确率。

Transformer模型采用编码-解码框架，利用自注意力机制以实现对时间序列更为高效的特征提取，捕捉长序列中各元素之间的依赖关系，且能够实现并行训练，具有强大的提取特征能力、融合多模态能力以及可解释能力。由于时序任务中采用Transformer的方法取得了极大成功，其逐渐被广泛应用于图形图像识别、自然语言处理以及经济证券预测等诸多领域[13-14]。文献[15]提出一种分层多尺度高斯改进的Transformer以捕捉股票时间序列的长期及短期依赖性；文献[16]基于门控循环单元结构引入Attention机制以侧重于重要时间点的股票特性信息，准确反映了股价的变动规律；文献[17]采用Transformer作为用户数据的自编码器深入建模了用户行为。

以上研究采用不同方法，优化提升了股票价格的预测性能，但现有基于Transformer的股票价格预测研究并未结合股价特点进行模型的适配性设计，泛化能力及预测准确率方面具有较大的提升空间。具体地，经典Transformer的提出是为了针对长时间序列进行建模，而股票价格的波动特性决定了该预测问题所涉及的输入输出序列长度较短。此外，经典Transformer中自注意力机制所采用的点乘计算法只能学习到序列中各时间点之间的依赖关系，对序列的局部特征并不敏感，因此本文提出设计一种新的学习能力更强、具备因果推断能力的自注意力机制以提高模型预测能力。考虑 Trans-former自身的优异性能及可拓展性，为进一步降低股票价格的预测误差，本文基于Transformer框架提出一种更为适合股票价格预测的深度学习预测模型Stockformer，其通过多头因果自注意力机制挖掘股票价格与特征因子之间的时序依赖关系，采用趋势增强模块为模型提供序列的趋势特征，同时利用编码器的特定输入为预测提供输入特征的直接先验信息。实验选取沪深300指数及3只典型个股作为数据集，对比分析了Stockformer与其他4种基线模型的预测结果及预测效率，并通过3组消融实验评估了因果注意力机制、时序特征提取手段以及特定的模型输入功能模块的效果及必要性，验证了所提 Stockformer 模型相较经典 Transformer 的优越性以及应用于股价预测任务的有效性，并具有一定的实际应用价值。

# 1 预测方案设计

# 1.1 问题定义

股票价格预测属于时间序列预测问题，通过给定一组由 $t _ { 1 }$ 到 $t _ { 2 }$ 时刻（时间序列长度为 $L$ ）的时序观测值$X _ { d \times L } = \left[ x _ { t _ { 1 } } , x _ { t _ { 2 } } , \cdots , x _ { t _ { L } } \right]$ ，时间序列预测的任务即为找到一个函数 $f$ 满足以下映射，可描述为：

$$
f ( x _ { t _ { 1 } } , x _ { t _ { 2 } } , \cdots , x _ { t _ { L } } ) { = } [ \hat { y } _ { t _ { L + 1 } } , \hat { y } _ { t _ { L + 2 } } , \cdots , \hat { y } _ { t _ { L + T } } ]
$$

同时能够最大程度减小真实值 $[ y _ { t _ { L + 1 } } , y _ { t _ { L + 2 } } , \cdots , y _ { t _ { L + T } } ]$ 和预测值 $[ \hat { y } _ { t _ { L + 1 } } , \hat { y } _ { t _ { L + 2 } } , \cdots , \hat { y } _ { t _ { L + T } } ]$ 之间的差异。其中， $L$ 和 $T$ 分别为观测长度和预测长度。从数据格式的角度看，股票价格预测模型的输入为序列长度为 $L$ 的股价历史数据以及其他 $N$ 维特征因子数据，即 $\boldsymbol { X } \in \mathbf { R } ^ { ( N + 1 ) \times L }$ ；模型的输出为序列长度为 $T$ 的价格预测值，即 $\hat { Y } \in \mathbf { R } ^ { 1 \times T }$ 。

# 1.2 训练框架

针对股票价格的预测任务，选择和设计训练框架主要需要考虑以下三个特征：

（1）股票收益率不满足独立分布假设，违背了机器学习的基本假设。由于同一截面的股票收益率并非独立，其共同受到当日市场行情的影响而呈现较高的相关性，因此对模型的训练和验证将会产生影响。

（2）需要充分考虑样本数量和模型复杂度之间的对应关系，降低过拟合对模型泛化能力的负面影响。由于深度学习模型的参数数量较为灵活，拟合能力强，因此容易过度拟合到股票数据中的噪声部分，导致样本内的预测效果与样本外产生巨大偏差。

（3）确保不出现“数据前窥”问题。量化金融研究中，杜绝数据前窥现象是一种基本原则，其确保了量化模型应用的可行性及正确性。对于任何时间节点，模型只能以当前时间节点及之前已经发生和可获得的数据作为输入，任何输入都不能采用到未来数据。

考虑上述三种问题，构建本文基础训练框架，简述如下：

（1）按照时间顺序划分训练集、验证集和测试集，要求三个数据集时间维度不重叠。训练过程中，只有训练集的数据不是完全已知的，验证集的数据只能用于验证，以确保了引入验证环节、预防过拟合的情况下，杜绝了数据前窥。

（2）通过深度学习中验证和早停（early stopping）方式以规避过拟合的风险，同时避免不必要的训练，减少训练时间。

# 2 Stockformer 股票价格预测模型

Stockformer 以 Transformer 模型为基础架构设计，数值嵌入及时间编码模块主要用以提取原始输入数据的时序关系特征，编码器和解码器的多头因果注意力模块是Stockformer的核心，编码器用以提取输入时间序列中的语义信息，解码器利用语义信息还原价格曲线。作为主要创新点，Stockformer分别为编码器和解码器设计了因果自注意力机制及趋势强化两个模块，前者主要挖掘股票价格与特征因子之间的时序依赖关系，后者为模型提供序列的趋势特征，最终通过线性模块回归得到预测输出。基于 Transformer 框架的 Stockformer 股票价格预测模型结构如图1所示。

![](images/550dba5d18cd0dc96e679d6cf1c2513c5e18abc49f332e4d880b83ab11f86b4c.jpg)  
图 1 Stockformer 模型结构  
Fig.1 Stockformer model structure

# 2.1 编码器及解码器

Stockformer 的编码器由数量为 $N$ 且相同的层组成 ，各 层 由 多 头 因 果 注 意 力（multi-head causal atten-tion）及趋势强化（trend enhance）两个主要模块组成。原始输入序列 $X _ { \mathrm { i n } }$ 经过数值嵌入（value embedding）与时间编码（time encoding）的结果相加，构成编码器的输入 $X _ { \mathrm { e n c } }$ 。其中，数值嵌入操作是为了方便后续的残差连接及模型堆叠，将数据由 $d _ { \mathrm { i n } }$ 维的输入空间投影到 $d _ { \mathrm { m o d e l } }$ 维的模型空间以对齐模型的数据维度；时间编码则是为了提供模型嵌入序列的时间戳信息。

Stockformer 的解码器由数量为 $M$ 且相同的层组成，其新增了一组多头注意力（multi-head attention）模块，根据编码器的输出 $K$ 和 $V$ 为解码器的中间结果 $Q$ 赋予了不同的权重，以计算 $K$ 和 $Q$ 之间的相关程度。

单步生成式推理最早应用于Informe[18]中，通过重新设计其解码器的输入模式，可以充分利用历史数据集并学习经验。Stockformer解码器的输入主要包括初始输入（start token） $X _ { \mathrm { s t a r t } }$ 及输出基准（output base） $X _ { \mathrm { b a s e } }$ 两部分。其中， $X _ { \mathrm { s t a r t } }$ 由长度为 $L _ { \mathrm { t o k e n } }$ 的历史参考序列$X _ { \mathrm { t o k e n } }$ 和长度为 $T$ 的零填充序列 0 组成； $X _ { \mathrm { b a s e } }$ 由 $X _ { \mathrm { t o k e n } }$ 和长度为 $T$ 的尾值填充序列 $X _ { \mathrm { l a s t } }$ 组成，分别可描述为：

$$
X _ { \mathrm { s t a r t } } = C o n c a t ( X _ { \mathrm { t o k e n } } , 0 )
$$

$$
X _ { \mathrm { b a s e } } = C o n c a t ( X _ { \mathrm { t o k e n } } , X _ { \mathrm { l a s t } } )
$$

式中，Concat()表示元素的拼接。将解码器的输出和经过数值嵌入的 $X _ { \mathrm { b a s e } }$ 结果相加，再经过一个线性层即可得到最终的预测结果。Stockformer解码器的输入模式

可以为预测模型提供更为直接的已知历史信息，简单、高效地为模型提供输出值的先验信息。

# 2.2 多头因果注意力

考虑股价历史走势在未来重演的可能性，Stock-former对于经典Transformer进行优化改进，提出构建多头因果注意力模块，利用因果卷积（causal convolution）提取时间序列中的局部特征，再通过多头因果自注意力（multi-head causal self-attention）捕捉其中的时序依赖关系，进而由前馈神经网络（feed forward network）得到模块的输出，生成与历史分布更为拟合的预测数据，提升股票价格趋势预测效果。多头因果注意力模块如图2所示。

![](images/222d318b53989b561be2e46b28870f94ca479a3de90787d61b50d19d8800fc02.jpg)  
图2 多头因果注意力模块  
Fig.2 Multi-head causal attention module

# 2.2.1 因果卷积

经典 Transformer 模型自注意力（self-attention）机制所采用的点乘计算法只能学习到序列中各时间点之间的依赖关系，但对于序列的局部特征并不敏感，此时自注意力可能由于价格或特征因子序列中的部分点噪声或波动而学习到错误的时序依赖关系，导致预测结果产生偏差。为充分利用序列的局部特征，根据卷积神经网络具有局部相关性的特点，提出采用因果卷积为自注意力模块提取序列中的局部特征。

因果卷积相较常规卷积的优势在于可以有效避免历史信息泄露问题，即 $t$ 时刻的输出只由 $t$ 时刻及之前更早时刻的元素卷积得到，大小为 $k$ 的卷积核 $i$ 在 $t$ 时刻的输出 $y _ { t } ^ { i }$ 可描述为：

$$
y _ { t } ^ { i } = A c t i v a t i o n ( W ^ { ( i ) } [ x _ { t - k + 1 } , x _ { t - k + 2 } , \cdots , x _ { t } ] )
$$

式中： $\boldsymbol { W } ^ { ( i ) }$ 表示卷积操作，Activation表示激活函数。 $k$ 的大小为序列局部性的感受范围，当 $k = 1$ 时，因果卷积将退化为逐点的前馈神经网络，不再具有捕捉序列局部特征的能力。

# 2.2.2 多头因果自注意力

Transformer 采用的注意力（full attention）机制能够挖掘并保留序列各元素之间全部的相关关系信息，通过学习输入序列中的各输入元素对目标元素的重要程度，并根据不同的重要程度划分等级，对不同的特征赋予不同的权重。为提高预测精度，Stockformer提出通过引入遮罩以限制这种相关关系只能向历史方向计算，避免时间维度的信息泄漏，将模型的最终决策聚焦于对预测目标具有正向帮助的特征维度中。

Attention 机制的本质是依据各个数据的重要性系数进行相乘后再相加求和，全注意力机制的计算采用请求-主键-数值（query-key-value，QKV）的模式，通过将序列第 $i$ 个元素产生的 $q _ { i }$ 与序列第 $j$ 个元素产生的 $k _ { j }$ 和$\mathcal { V } _ { j }$ 相乘，进而得到位置 $i$ 与位置 $k$ 的相关性 $a _ { i j } = q _ { i } k _ { j } ^ { \mathrm { T } } v _ { j }$ ，$i$ 点的输出 $a _ { i }$ 为它与其他位置的相关性之和，称为该点的注意力分数，可描述为：

$$
a _ { i } = \sum _ { j = 1 } ^ { L _ { M } } a _ { i j }
$$

因果自注意力机制是自注意力机制的进一步优化，以更紧凑的矩阵形式可描述为：

$$
A t t e n t i o n ( Q , K , V ) { = } s o f t m a x \left( m a s k { \left( \frac { Q K ^ { \mathrm { T } } } { \sqrt { D _ { k } } } \right) } \right) V
$$

式中， $\boldsymbol { Q } \in \mathbf { R } ^ { L _ { N } \times D _ { k } }$ 、 $\boldsymbol { K } \in \mathbf { R } ^ { L _ { M } \times D _ { k } }$ 以及 $V \in \mathbf { R } ^ { L _ { M } \times D _ { v } }$ 分别表示请求矩阵、主键矩阵以及数值矩阵； $K ^ { \mathrm { T } }$ 为 $K$ 的转置矩阵。 $L _ { N }$ 和 $L _ { M }$ 分别表示请求序列和主键序列（或数值序列）的长度； $D _ { k }$ 和 $D _ { v }$ 分别表示请求（或主键）和数值向量的维度； $\boldsymbol { Q }$ 和 $K ^ { \mathrm { T } }$ 的点乘得到的权重分数刻画的是向量之间的相关关系，再除以 $\sqrt { D _ { k } }$ 以缓解 Softmax 函数中的梯度消失问题。Softmax激活函数通过将这些权重分数进行归一化处理并将它们的和置为1，得到时间及空间尺度上某点的重要程度，再将Softmax的函数值与数值矩阵 $V$ 按位相乘，最终得到注意力矩阵。 mask(⋅)表示遮罩操作，其可将矩阵的上三角元素（不包括对角线）置为 $- \infty$ ，其余元素不变，以确保某点的注意力分数只能通过该点之前的元素计算得到，有效防止了历史信息泄露，同时增强/削弱了相关关系强/弱的空间特征，有利于模型更好地聚焦于重要信息，因此称为因果注意力。当 $Q \ 、 K$ 和 $V$ 均来自同一序列时，此时的注意力又称为自注意力，其中， $Q \ 、 K$ 和 $V$ 是神经网络需要学习的参数。

实际操作时还将采用多个不同的 $Q \ 、 K$ 和 $V$ 将输入投影到 $h$ 个不同的子空间，通过多次拆分并计算注意力权重分数以捕捉序列中不同类别的相关关系，增强网络的学习能力，因此称为多头注意力机制。各子空间中的输出由式（6）计算，经过多次并行计算，将所有子空间的输出拼接后再投影回模型的 $d _ { \mathrm { m o d e l } }$ 维空间。由于不同子空间的注意力分布不同，多个独立的头部能够关联序列中的不同信息，因此多头注意力赋予了 Stockformer更为强大的特征提取能力，可描述为：

$$
{ \cal M } u { \cal l } t i { \cal H } e a d ( Q , K , V ) =
$$

$$
C o n c a t ( h e a d _ { 1 } , h e a d _ { 2 } , \cdots , h e a d _ { h } ) _ { } \mathrm { { W } } ^ { \mathrm { { O } } }
$$

$$
h e a d _ { i } = A t t e n t i o n ( Q W _ { i } ^ { o } , K W _ { i } ^ { o } , V W _ { i } ^ { O } )
$$

式中， $W _ { i } ^ { Q } \in \mathbf { R } ^ { d _ { \mathrm { m o d e l } } \times d _ { k } }$ ， $\boldsymbol { W } _ { i } ^ { K } \in \mathbf { R } ^ { d _ { \mathrm { m o d e l } } \times d _ { k } }$ ， $W _ { i } ^ { V } \in \mathbf { R } ^ { d _ { \mathrm { m o d e l } } \times d _ { v } }$ ，W Oi ∈ Rhdv × dmodel 。

# 2.2.3 前馈神经网络及层标准化

Stockformer 采用两个内核尺寸为 1 的卷积层组成

前馈神经网络，可描述为：

$$
F ( X ) { = } C o n v ^ { \prime } { \big ( } G E L U { \big ( } C o n v ( X ) { \big ) } { \big ) }
$$

式中，Conv(⋅):Xd 和 $C o n v ^ { \prime } ( \cdot ) { : } X _ { d _ { f f } } { \longrightarrow } X _ { d _ { \mathrm { m o d e l } } }$ 分别表示两个卷积层，其内部连接的维度为 $d _ { f f }$ 。 $G E L U ( )$ 表示高斯误差线性单元激活函数。

同时，Stockformer采用残差连接及层标准化操作，以缓解深度堆叠过程中网络的梯度消失问题：

$$
H ^ { \prime } { = } L N \big ( M C S A ( X ) + X \big )
$$

$$
H = L N \big ( F ( H ^ { \prime } ) + H ^ { \prime } \big )
$$

式中， $M C S A ( )$ 表示多头因果自注意力模块； $L N ( \cdot )$ 表示层标准化操作。

# 2.3 趋势强化模块

多头排列时的股价大概率将持续上涨，空头排列时的股价大概率将持续下跌，考虑股票价格沿趋势方向演变且波动性较大，其涉及的时间序列较短，因此需特别关注短期趋势对价格预测结果的影响。假设某网络能够轻松获得特征因子随时间变化的某种趋势信息，则其预测精度就越高。为降低网络挖掘序列中趋势项信息的复杂度，Stockformer 提出嵌入趋势强化（tread enhance，TE）模块，向网络中提供更多序列的趋势项特征。具体地，利用滑动平均法提取出时间序列中的趋势项，随后采用残差连接的方式将提取到的趋势特征嵌入到网络模型中，等效为增强序列的趋势项，因此称为趋势增强。对长度为 $L$ ，维度为 $d$ 的输入 $X \in \mathbf { R } ^ { L \times d }$ ，趋势增强的结果可描述为：

$$
T E ( X ) { = } A v g P o o l { \big ( } P a d d i n g ( X ) { \big ) } + X
$$

式中， $A v g P o o l ( )$ 表示平均化，等价于对序列进行滑动平均；Padding() 表示对输入进行填充，以确保输入输出序列长度一致。

# 2.4 时间编码

序列中某点的自注意力计算结果是通过该点的 $q _ { i }$ 与其余点的 $k _ { j }$ 和 $\boldsymbol { v } _ { j }$ 相乘后再相加得到，如式（6）所示，但其中并未考虑序列中其余各点的位置信息，由于其位置信息对于捕捉序列中各元素之间的依赖关系具有重要意义，因此需要将位置信息编码后嵌入到输入序列中。对于某个时间序列，序列中元素的位置可利用其时间戳代替，因此将此模块称为时间编码（time encoding），并提出将其分解为相对时间编码（relative time encoding，RTE）及绝对时间编码（absolute time encoding，ATE）两个模块分别讨论。

# 2.4.1 相对时间编码

对于某条长为 $L$ 的时间序列 $\{ \boldsymbol { x } _ { t _ { 1 } } , \boldsymbol { x } _ { t _ { 2 } } , \cdots , \boldsymbol { x } _ { t _ { L } } \}$ 进行相对时间编码就是对序列中各时间点的相对位置信息进行编码。通过对 Vaswani等的研究成果作进一步优化[19]，对输入序列的相对时间编码可描述为：

$$
R T E ( t _ { i } , 2 j ) { = } \mathrm { s i n } \Bigg ( \frac { t _ { i } - t _ { 1 } } { 1 0 0 0 0 ^ { 2 j / d _ { \mathrm { m o d e l } } } } \Bigg )
$$

$$
R T E ( t _ { i } , 2 j + 1 ) { = } \mathrm { c o s } { \left( \frac { t _ { i } - t _ { 1 } } { 1 0 0 0 0 ^ { 2 j / d _ { \mathrm { m o d e l } } } } \right) }
$$

式中， $R T E ( )$ 表示相对时间编码； $t _ { i }$ 表示序列中第 $i$ 个位置的时间戳； $t _ { 1 }$ 表示输入序列的第一个时间点； $j$ 表示维度； $i \in \{ 1 , 2 , \cdots , L \}$ ， $j \in \{ 1 , 2 , \cdots , d _ { \mathrm { m o d e l } } / 2 \} .$ 。

# 2.4.2 绝对时间编码

由于时间戳自身携带诸多具有价值的信息，无法单方面依靠序列中各点的相对位置对输入序列的时间戳进行编码，因此需要通过计算输入序列中所有点的时间戳与某固定时间戳的时间差，将绝对时间编码转换为相对时间编码。这种时间编码的计算方式相较现有的其他方式更为简洁清晰，且效果等价。绝对时间编码可描述为：

$$
A T E ( t _ { i } , 2 j ) { = } \mathrm { s i n } \Bigg ( \frac { t _ { i } - t _ { c } } { 1 0 0 0 0 ^ { 2 j / d _ { \mathrm { m o d e l } } } } \Bigg )
$$

$$
A T E ( t _ { i } , 2 j + 1 ) { = } \mathrm { c o s } \Bigg ( \frac { t _ { i } - t _ { c } } { 1 0 0 0 0 ^ { 2 j / d _ { \mathrm { m o d e l } } } } \Bigg )
$$

式中， $t _ { c }$ 为固定时间戳。

通过相对时间编码及绝对时间编码得到输入序列长度为 $L$ 、维度为 $d _ { \mathrm { m o d e l } }$ 的时间编码，可描述为：

$$
T i m e E n c ( t _ { i } , j ) = R T E ( t _ { i } , j ) + A T E ( t _ { i } , j )
$$

式中， $t _ { i }$ 表示序列中第 $i$ 个位置的时间戳， $j$ 表示维度，$i \in \{ 1 , 2 , \cdots , L \} , j \in \{ 1 , 2 , \cdots , d _ { \mathrm { m o d e l } } \} \circ$

# 3 实验结果及分析

# 3.1 实验描述

# 3.1.1 特征因子及数据来源

数据和特征决定了机器学习的上限，特征工程是股票价格预测实验的重要环节，其对预测结果具有显著影响。随着时间的推移，股票市场积累了大量反映其价格走势波动的历史数据。特征因子的数量和预测效果并不成正比例关系，即特征数量越多，其预测效果不一定就越好[20]，因此需要正确选取最具价值的输入特征指标进行训练，去除部分对于预测效果影响较小的特征因子，以提高模型预测性能。因此，实验基础数据除去选取历史股票价格信息外，还选取了两种技术指标即异同移动平均线（MACD）以及随机指标（KDJ）作为特征因子。其中，MACD指标基于快、慢均线位置关系以反映股票市场的变化趋势，KDJ指标主要反映市场的变化趋势强弱。

考虑中国上市公司数量庞大，为提升实验分析的客观性及普适性，分别对股票指数和典型个股价格进行预测。股指数据选取沪深300指数，数据的时间范围由2005年7月5日至2022年8月18日。个股数据选自金融、医药、通信3个不同热门行业的3只典型股票，分别为建设银行、中国医药以及中国联通，数据的时间范围由2010年5月10日至2022年8月18日。将以上数据分别以时间为序并按照7∶1∶2的方式分别划分为训练集、验证集以及测试集，分别在各数据集中生成正样本作为训练样本集、验证样本集以及测试样本集。上述数据来源均为Wind数据库。

# 3.1.2 模型参数设置

实验模型采用 Tesla V100 的 GPU 搭建并计算，Stockformer 由 $N = 2$ 的编码器以及 $M = 1$ 的解码器组成。模型训练采用学习率为0.000 1的Adam优化器[21]，损失函数类型为均方误差（mean squared error，MSE），批大小设置为32。训练集迭代20轮次，每轮次训练结束时在验证集上评估当前模型的平均绝对误差（meanabsolute error，MAE），如果其小于目前最优MAE，则保存该模型并更新最优MAE；如果连续5轮次训练的模型均大于最优MAE，则判定该模型已经达到学习上限并停止训练，这种策略称为提前停止（early stopping），可以有效防止模型过度拟合并提高训练效率。最终将以验证集上表现最佳的模型作为训练得到的完备模型，并在测试集上评估其表现。

为进一步验证Stockformer模型的预测效果，选取5种现有股票价格预测深度学习模型进行对比实验，分别为 SVR[22] 、CNN、LSTM[23] 、Transformer[24] 以 及 DST2V-Transformer[25] 。

# 3.1.3 评价指标

为量化评估模型的预测精度，采用平均绝对误差（MAE）以衡量预测值与真实值之间的差异，并采用均方根误差（root mean square error，RMSE）以提高评价指标对于特大或特小误差的灵敏度。其中，MAE和RMSE 数值越小，则表明预测值越接近真实值，预测结果越准确。此外，采用涨跌准确率 $K$ 以表征预测模型对趋势的预测情况。评价指标分别可描述为：

$$
\ M A E ( y , { \hat { y } } ) = { \frac { \displaystyle \sum _ { i = 1 } ^ { n } \left| y _ { i } - { \hat { y } } _ { i } \right| } { n } }
$$

$$
R M S E = \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \bigl ( y _ { i } - \hat { y } _ { i } \bigr ) ^ { 2 } }
$$

$$
K = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } I \bigl ( T r e n d _ { i , p } = T r e n d _ { i , a } \bigr )
$$

式中， $n$ 为总样本量， $y _ { i }$ 和 $\hat { y } _ { i }$ 分别表示真实值和预测值。 $T r e n d _ { i , p }$ 和 $T r e n d _ { i , a }$ 分别表示第 $i$ 个时间点的预测涨跌和实际涨跌， $I ( \cdot )$ 值域为 $\{ 0 , 1 \}$ ， $T r e n d _ { i , p } = T r e n d _ { i , a }$ 时 $I ( \cdot ) = 1$ ， $T r e n d _ { i , p } \ne T r e n d _ { i , a }$ 时 $I ( \cdot ) { = } 0$ 。

# 3.2 预测结果分析

# 3.2.1 股指预测实验

为验证 Stockformer 预测效果，采用 Stockformer 和以上5种基线模型对沪深300指数进行横向对比预测。同时，为消除单次测试的偶然性，测试结果由多次测试求平均值得到。不同模型预测结果如表1所示。

表1 不同模型对股指价格预测结果  
Table 1 Prediction results of stock index prices by different models   

<table><tr><td>模型</td><td>MAE</td><td>RMSE</td><td>K</td></tr><tr><td>SVR</td><td>3.826</td><td>4.855</td><td>0.5371</td></tr><tr><td>CNN</td><td>2.829</td><td>3.233</td><td>0.604 8</td></tr><tr><td>LSTM</td><td>1.903</td><td>1.745</td><td>0.7533</td></tr><tr><td>Transformer</td><td>1.227</td><td>1.634</td><td>0.786 8</td></tr><tr><td>DST2V-Transformer</td><td>1.137</td><td>1.601</td><td>0.769 3</td></tr><tr><td>Stockformer</td><td>0.942</td><td>1.214</td><td>0.8037</td></tr></table>

由表1可知，SVR作为一种经典机器学习模型，其MAE和RMSE最大且趋势指标K最小，这是由于SVR对于股价这类具有强波动性的时间序列预测问题的拟合程度较低，预测性能表现最差。同样，各项评价指标均表明 CNN 的预测结果并不理想。这是由于 CNN 并不具备时间序列的建模能力，无法充分利用历史股价和技术因子数据，不可避免地产生历史信息泄露问题。此外，LSTM作为CNN的改进神经网络模型，预测性能较CNN有所提升，虽然避免了模型训练过程中的梯度消失和爆炸问题，但基于自注意力机制的 Trans-former 的预测性能更为突出，这是由于 Transformer 能够快速提取时序信息并实现并行计算，有效减小了预测误差并提升了计算速度。DST2V-Transformer是一种基于Transformer的改进模型，其通过引入移动平均方法对数据进行平滑处理，以识别时间序列的趋势成分，并在数据中引入时间信息以捕获时间序列的周期和非周期分量，提高Transformer的长期预测性能。实验结果表明 DST2V-Transformer 的 MAE 和 RMSE 指标相较经典Transformer均得到了提升，但指标 $K$ 有所下降，因此DST2V-Transformer存在一定程度的过拟合现象。综上，Stockformer的各项评价指标表现均为最优，这是由于Stockformer具有更为强大的特征提取能力及网络学习能力，因此其MAE和RMSE最小，且较Transformer分别降低了 $2 3 . 2 \%$ 和 $2 5 . 7 \%$ ， $K$ 指标提高了 $2 . 3 \%$ 且更接近于1，对于趋势的预测更为准确且预测误差更小，预测效果显著优于其他5种基线模型。

如图3为5种深度学习模型对某段时间沪深300指数的预测结果。由图3可知，SVR和CNN的预测结果呈现乱序、波动大等负面情况；LSTM虽然具有微弱优势，但其预测结果过于依赖模型的超参数设置，不同的神经元个数、迭代次数以及学习率等参数均可能改变LSTM的运算复杂度且影响最终的预测准确度，因此导致其曲线拟合程度并不高，仍有较大提升空间。基于注意力机制的 Transformer 和 DST2V-Transformer 的预测值和真实值较为接近，且对于未来趋势的预测较为准确，这是由于两者具有相似的单元结构及特性，但Transformer 的预测值波动较大，DST2V-Transformer 在数据连续波动的拐点处时预测效果不好。综上，Stock-former能够准确捕捉趋势变化幅度较大的拐点，预测性能相较Transformer进一步得到提升，两条曲线近乎重合，预测值和真实值的误差最小且回归拟合效果最优，直观验证了Stockformer的拟合能力及优越性。

![](images/178ac342fd567fc074ba6427ac0440107a697c8829c9049dc23992baa54b7736.jpg)  
图3 不同模型对沪深300指数预测曲线  
Fig.3 Prediction curve of CSI 300 index by different models

# 3.2.2 个股预测实验

实际环境下影响股指和个股变化的因素不同，为进一 步 验 证 模 型 的 鲁 棒 性 及 普 适 性 ，选 取 建 设 银 行（60050）、中国医药（600056）以及中国联通（601939）这3只典型股票为样本，采用Stockformer和各基线模型进行个股价格预测实验，预测结果如表2所示。

由表2可知，不同个股样本下，Stockformer的各项指标性能均显著优于其他模型，其鲁棒性及普适性得到验证。经过多次实验，注意到个股预测的涨跌准确率 $K$ 指标方面，各模型预测结果相较股指样本下滑较为明显，表明个股相较股指价格波动的随机性更强，趋势建模更为困难。因此，股指及个股预测实验结果表明，Stockformer以其优异、鲁棒的预测性能可以提前为投资者的投资行为提供重要参考，有效降低投资风险，同时提高股民收益。

# 表2 不同模型对典型个股价格预测结果

Table 2 Prediction results of typical individual stock prices by different models   

<table><tr><td>个股</td><td>模型</td><td>MAE</td><td>RMSE</td><td>K</td></tr><tr><td rowspan="6">建设银行 (60050)</td><td>SVR</td><td>0.269</td><td>0.342</td><td>0.412</td></tr><tr><td>CNN</td><td>0.217</td><td>0.253</td><td>0.437</td></tr><tr><td>LSTM</td><td>0.185</td><td>0.201</td><td>0.522</td></tr><tr><td>Transformer</td><td>0.095</td><td>0.117</td><td>0.663</td></tr><tr><td>DST2V-Transformer</td><td>0.098</td><td>0.124</td><td>0.667</td></tr><tr><td>Stockformer</td><td>0.084</td><td>0.093</td><td>0.704</td></tr><tr><td rowspan="6">中国医药 (600056)</td><td>SVR</td><td>0.250</td><td>0.289</td><td>0.359</td></tr><tr><td>CNN</td><td>0.197</td><td>0.228</td><td>0.381</td></tr><tr><td>LSTM</td><td>0.154</td><td>0.177</td><td>0.549</td></tr><tr><td>Transformer</td><td>0.138</td><td>0.156</td><td>0.602</td></tr><tr><td>DST2V-Transformer</td><td>0.134</td><td>0.155</td><td>0.610</td></tr><tr><td>Stockformer</td><td>0.131</td><td>0.152</td><td>0.624</td></tr><tr><td rowspan="6">中国联通 (601939)</td><td>SVR</td><td>0.245</td><td>0.173</td><td>0.382</td></tr><tr><td>CNN</td><td>0.237</td><td>0.166</td><td>0.409</td></tr><tr><td>LSTM</td><td>0.203</td><td>0.161</td><td>0.511</td></tr><tr><td>Transformer</td><td>0.100</td><td>0.115</td><td>0.634</td></tr><tr><td>DST2V-Transformer</td><td>0.085</td><td>0.103</td><td>0.612</td></tr><tr><td>Stockformer</td><td>0.076</td><td>0.087</td><td>0.695</td></tr></table>

# 3.3 预测效率分析

为评估Stockformer预测效率，以3种不同预测长度下 Stockformer 和其他 Transformer 类的时序预测模型（Informer，Reformer，Autoformer）的运行效率进行灵敏度分析，并分别考虑迭代速率以及内存占用两个维度进行讨论，得到的4种模型预测效率如图4所示。其中，预测长度以10为步长递进，迭代速率的计算采用1 000次网络循环的平均值。

![](images/907823efa375625c45c8fd3dd3f668b2086f6cc1c5492c65225884280782259a.jpg)  
图4 不同模型效率对比  
Fig.5 Efficiency comparison of different models

由图4可知，同一预测长度下，Autoformer以占用大量内存为代价换取其快速迭代，实际运行效率并不高；Reformer 运行时的内存占用与 Stockformer 较为相似，但其迭代速率相对较慢，计算过程耗时较长。此外，虽然Stockformer相较Informer牺牲了运行效率，但其预测稳定性更优，由于股票价格预测任务所涉及的序列长度较短，其最终的实际运行效率并未受到实质性影响，预测结果仍然处于可接受的范围内。

# 3.4 消融实验

为验证所提Stockformer模型的预测效果以及各功能模块的有效性和必要性，通过移除模型中的不同模块得到3种变体模型，分别设计以下3组消融实验并根据评价指标的计算进行对比分析。

# 3.4.1 因果注意力消融实验

因果注意力消融实验通过控制模型的其他部分保持不变，仅将因果注意力模块替换为其他注意力模块 。 其 中 LogSparse attention 来 自 Transformer，LSHattention 来 自 Reformer[26] ，ProbSparse attention 来 自

Informer。Stockformer 的因果注意力机制与其他时序预测模型的注意力机制对比结果如表3所示。

表3 因果注意力消融实验结果  
Table 3 Results of causal attention ablation experiments   

<table><tr><td>注意力机制</td><td>MAE</td><td>RMSE</td><td>K</td></tr><tr><td>Vinilla attention</td><td>1.107</td><td>1.522</td><td>0.785 8</td></tr><tr><td>LogSparse attention</td><td>1.114</td><td>1.581</td><td>0.7746</td></tr><tr><td>LSH attention</td><td>1.172</td><td>1.613</td><td>0.765 1</td></tr><tr><td>ProbSparse attention</td><td>1.228</td><td>1.678</td><td>0.7522</td></tr><tr><td>Stockformer</td><td>0.942</td><td>1.214</td><td>0.8037</td></tr></table>

由表3可知，通过对比MAE和RMSE的计算结果，Stockformer的因果注意力机制性能均优于其他注意力机制，MAE和RMSE分别平均降低了 $1 8 . 3 \%$ 和 $2 3 . 9 \%$ ，验证了所提机制的有效性。这是由于参与对比的其他注意力机制的设计虽然一定程度减少了计算复杂度，但牺牲了对股票价格时序特征的提取能力，具有一定的局限性。此外，因果注意力相较其他注意力的趋势预测指标 $K$ 平均提高了 $4 . 3 \%$ ，进一步表明因果注意力机制的优越性，能够高效利用历史信息，对未来股价的趋势具有更强的预测能力。

# 3.4.2 时间序列特征消融实验

Stockformer采用因果卷积及趋势增强模块以处理序列数据，分别提高了模型对时间序列局部及趋势特征的学习能力。从提取时间序列特征的角度看，两者均能缓解注意力只能捕捉点与点之间依赖关系的局限性。Stockformer分别去除这两种模块后的实验结果如表4所示。其中 Stockformer∗表示去除了 Stockformer 中的因果卷积模块；Stockformer\*\*表示去除了 Stockformer中的趋势增强模块。

表4 时间序列特征消融实验结果  
Table 4 Results of time series characteristic ablation experiments   

<table><tr><td>模型</td><td>MAE</td><td>RMSE</td><td>K</td></tr><tr><td>Stockformer</td><td>0.942</td><td>1.214</td><td>0.8037</td></tr><tr><td>Stockformer*</td><td>1.197</td><td>1.748</td><td>0.7834</td></tr><tr><td>Stockformer**</td><td>1.083</td><td>1.573</td><td>0.7358</td></tr></table>

由表 4 可知，去除因果卷积模块后 MAE 和 RMSE分别提高了 $2 1 . 3 \%$ 和 $3 0 . 5 \%$ ，且 $K$ 降低了 $2 . 5 \%$ ；去除趋势增强模块后MAE和RMSE分别提高了 $1 3 . 1 \%$ 和 $2 2 . 8 \%$ ，且 $K$ 降低了 $8 . 4 \%$ ，表明预测模型拟合真实值的效果明显下降。因此，同时考虑因果卷积及趋势增强模块对股票价格的准确预测具有积极影响。其中，因果卷积可以有效避免历史信息的遗失泄漏问题；由于需要特别关注短期趋势对价格预测结果的影响，趋势强化能够获得更多特征因子随时间变化的某种趋势信息。因此，提取时间序列特征可以有效提高模型的预测效果及性能，且趋势增强模块对模型趋势预测能力具有显著影响，有利于投资者在某种趋势产生的前期通过判断及追随该趋势进行交易，最大程度获取自身收益。

# 3.4.3 特定模型输入消融实验

Stockformer提出设计一种新的解码器输入形式及时间编码方式，以验证所提策略的可行性及有效性，通过去除模型输入的部分功能环节得到5种变体Stock-former并进行消融实验，实验结果如表5所示。其中，Stockformer1 表示去除了输出基准 $X _ { \mathrm { b a s e } }$ 且无历史参考信息，即 $X _ { \mathrm { b a s e } } = 0$ 且 $L _ { \mathrm { t o k e n } } = 0$ ；Stockformer2 表示去除了输出基准 $X _ { \mathrm { b a s e } }$ ；Stockformer3 表示去除了历史参考信息，即 $L _ { \mathrm { t o k e n } } = 0$ ；Stockformer4 表示包括所有历史参考信息，即 $L _ { \mathrm { t o k e n } } = L = 1 4 4$ ；Stockformer5 表示去除了时间编码模块。

表5 特定模型输入消融实验结果  
Table 5 Results of model specific input ablation experiments   

<table><tr><td>模型</td><td>MAE</td><td>RMSE</td><td>K</td></tr><tr><td>Stockformer</td><td>0.942</td><td>1.214</td><td>0.8037</td></tr><tr><td>Stockformer1</td><td>1.012</td><td>1.484</td><td>0.7801</td></tr><tr><td>Stockformer2</td><td>0.976</td><td>1.282</td><td>0.794 7</td></tr><tr><td>Stockformer³</td><td>0.991</td><td>1.298</td><td>0.769 1</td></tr><tr><td>Stockformer</td><td>0.950</td><td>1.249</td><td>0.802 2</td></tr><tr><td>Stockformer5</td><td>0.953</td><td>1.253</td><td>0.7985</td></tr></table>

由表5可知，评价指标的计算结果表明，Stockformer的MAE和RMSE最小且 $K$ 最大，所提解码器输入方式及时间编码策略能够显著提高Stockformer预测性能。其中，采用单步生成式推理方式的解码器可以直接输出完整的目标序列，简单、高效地为模型提供输出值的先验信息。考虑序列中各点的位置信息对捕捉各元素之间依赖关系的重要性，将这些位置信息利用其时间戳代替并嵌入到输入序列中以进行时间编码，可以有效提升Stockformer的鲁棒性及泛化能力。此外，实际股票价格预测任务中过长的历史参考信息所带来的收益并不明显，反而可能引入额外的计算成本。

# 4 结论

为提升股票价格时间序列预测精度，针对现有深度学习模型结构不能有效反映特征因子对股价的累积作用这一问题，提出一种基于 Transformer 框架的 Stock-former预测模型，通过引入因果自注意力机制挖掘历史股价与特征因子之间的时序依赖关系，采用趋势增强模块为模型提供时序的趋势特征，同时利用编码器的特定输入为预测提供特征因子的直接先验信息。实验结果表明，Stockformer的预测性能及拟合程度显著优于现有SVR、CNN、LSTM、Transformer以及 DST2V-Transformer等经典深度学习模型，具有更优的特征提取能力、泛化能力以及运行效率。由于股票价格受政治、经济、社会以及心理等多方面因素影响难以预测，未来将进一步研究考虑融合财经新闻、股评舆情以及投资者情绪等情感分析理论的股价预测模型，优化Stockformer预测性能，为投资者提供更为准确且实际的参考。

# 参考文献：

[1] 王娜，施建淮.我国金融稳定指数的构建：基于主成分分析法[J]. 南方金融，2017（6）：46-55.WANG N，SHI J H.Construction of China’s financialstability index：based on principal component analysis[J].South China Finance，2017（6）：46-55.  
[2] 万校基，李海林.基于特征表示的金融多元时间序列数据分析[J]. 统计与决策，2015（23）：151-155.WAN X J，LI H L.Analysis of financial multivariate timeseries data based on feature representation[J].Statistics &Decision，2015（23）：151-155.  
[3] RASEKHSCHAFFE K C，ROBERT C J.Machine learningfor stock selection[J].Financial Analysts Journal，2019，75（3）：70-88.  
[4] 张倩玉，严冬梅，韩佳彤.结合深度学习和分解算法的股票价格预测研究[J].计算机工程与应用，2021，57（5）：56-64.ZHANG Q Y，YAN D M，HAN J T.Research on stockprice prediction based on deep learning and decompositionalgorithm[J].Computer Engineering and Applications，2021，57（5）：56-64.  
[5] 邓佳丽，赵凤群，王小侠.MTICA-AEO-SVR股票价格预测模型[J]. 计算机工程与应用，2022，58（8）：257-263.DENG J L，ZHAO F Q，WANG X X.MTICA-AEO-SVRstock price forecasting model[J].Computer Engineeringand Applications，2022，58（8）：257-263.  
[6] MOGHADDAM A H，MOGHADDAM M H，ESFAN-DYARI M.Stock market index prediction using artificialneural network[J].Journal of Economics，Finance andAdministrative Science，2016，21（41）：89-93.  
[7] 刘玉玲，赵国龙，邹自然，等 . 基于情感分析和 GAN 的股票价格预测方法[J]. 湖南大学学报（自然科学版），2022，49（10）：111-118.LIU Y L，ZHAO G L，ZOU Z R，et al.Stock price fore-casting method based on sentiment analysis and GAN[J].Journal of Hunan University（Natural Sciences），2022，49（10）：111-118.  
[8] HOSEINZADE E，HARATIZADEH S.CNNpred：CNN-based stock market prediction using a diverse set ofvariables[J].Expert Systems with Applications，2019，129：273-285.  
[9] QIU Y，YANG H Y，LU S，et al.A novel hybrid modelbased on recurrent neural networks for stock markettiming[J].Soft Computing，2020，24（20）：15273-15290.  
[10] 耿晶晶，刘玉敏，李洋，等.基于CNN-LSTM的股票指数预测模型[J]. 统计与决策，2021，37（5）：134-138.GENG J J，LIU Y M，LI Y，et al.Prediction model ofstock index based on CNN-LSTM[J].Statistics & Deci-sion，2021，37（5）：134-138.  
[11] 宋刚，张云峰，包芳勋，等.基于粒子群优化LSTM的股票预测模型[J]. 北京航空航天大学学报，2019，45（12）：2533-2542.SONG G，ZHANG Y F，BAO F X，et al.Stock forecastingmodel based on particle swarm optimization LSTM[J].Journal of Beijing University of Aeronautics and Astro-nautics，2019，45（12）：2533-2542.  
[12] 林昱，常晋源，黄雁勇.融合经验模态分解与深度时序模型的股价预测[J]. 系统工程理论与实践，2022，42（6）：1663-1677.LIN Y，CHANG J Y，HUANG Y Y.Stock price fore-casting based on empirical mode decomposition and deeptime series model[J].Systems Engineering- Theory &Practice，2022，42（6）：1663-1677.  
[13] ZHANG Q Y，QIN C，ZHANG F Y，et al.Transformer-based attention network for stock movement prediction[J].Expert Systems with Application，2022，202：117239.  
[14] WANG C J，CHEN Y Y，ZHANG S Q，et al.Stockmarket index prediction using deep Transformer model[J].Expert Systems with Application，2022，208：118128.  
[15] DING Q G，WU S F，SUN H，et al.Hierarchical multi-scalegaussian transformer for stock movement prediction[C]//Proceedings of International Joint Conference on Artifi-cial Intelligence，2022：4640-4646.  
[16] 谷丽琼，吴运杰，逄金辉.基于Attention机制的GRU股票预测模型[J]. 系统工程，2020，38（5）：134-140.GU L Q，WU Y J，PANG J H.GRU based on atten-tion mechanism stock forecast model[J].Systems Engi-neering，2020，38（5）：134-140.  
[17] 杨磊，姚汝婧.基于Transformer的信用卡违约预测模型研究[J]. 计算机仿真，2021，38（8）：440-444.YANG L，YAO R J.Research on credit card defaultprediction model based on transformer[J].ComputerSimulation，2021，38（8）：440-444.  
[18] ZHOU H Y，ZHANG S H，PENG J Q，et al.Informer：beyond efficient transformer for long sequence time-series forecasting[C]//Proceedings of the AAAI Confer-ence on Artificial Intelligence，2021：11106-11115.  
[19] VASWANI A，SHAZEER N，PARMAR N，et al.Attentionis all you need[C]//Advances in Neural InformationProcessing Systems，2017：5998-6008.  
[20] 徐浩然，许波，徐可文.机器学习在股票预测中的应用综述[J]. 计算机工程与应用，2020，56（12）：19-24.XU H R，XU B，XU K W.Application of machinelearning in stock prediction[J].Computer Engineeringand Applications，2020，56（12）：19-24.  
[21] KINGMA D P，BA J.Adam：a method for stochasticoptimization[C]//Proceedings of the 3rd InternationalConference for Learning Representations（ICLR），2015：1-15.  
[22] JIN Z，GUO K，SUN Y，et al.The industrial asymmetryof the stock price prediction with investor sentiment：based on the comparison of predictive effects withSVR[J].Journal of Forecasting，2020，39（7）：1166-1178.  
[23] NELSON D M Q，PEREIRA A C M，OLIVEIRA R A D.Stock market’s price movement prediction with LSTMneural networks[C]//Proceedings of 2017 InternationalJoint Conference on Neural Networks（IJCNN），2017：1419-1426.  
[24] LI S Y，JIN X Y，XUAN Y，et al.Enhancing the localityand breaking the memory bottleneck of transformer ontime series forecasting[C]//Advances in Neural Informa-tion Processing Systems，2019：5243-5253.  
[25] PREETI R B，SINGH R P.A dual-stage advanced deeplearning algorithm for long- term and long- sequenceprediction for multivariate financial time series[J].AppliedSoft Computing，2022，126：109317.  
[26] KITAEV N，KAISER Ł，LEVSKAYA A.Reformer：theefficient transformer[J].arXiv：2001.04451，2020.