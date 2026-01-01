# 融合增量学习与 Transformer模型的股价预测研究

陈东洋，毛 力+  
江南大学 人工智能与计算机学院 江苏省模式识别与计算智能工程实验室，江苏 无锡 214122  
$^ +$ 通信作者 E-mail: wxmaoli@163.com

摘 要：股票价格预测一直是金融研究和量化投资共同关注的重点话题。当前股价预测的深度学习模型多数基于批处理学习设置，这要求训练数据集是先验的，这些模型面对实时的数据流预测是不可扩展的，当数据分布动态变化时模型的预测效果将会下降。针对现有研究对非平稳股票价格数据预测精度不佳的问题，提出一种基于增量学习和持续注意力机制的在线股价预测模型（Increformer），通过持续自注意力机制挖掘特征变量之间的时序依赖关系，采用持续归一化机制处理数据非平稳问题，基于弹性权重巩固的增量训练策略获取数据流中的新知识，提高预测精度。在股票市场的股指与个股价格序列中选取五个公开数据集进行实验。实验结果表明，Increformer模型能够有效挖掘数据的时序信息以及特征维度的关联信息从而提高股票价格的预测性能。通过消融实验评估了Increformer模型的持续归一化机制、持续注意力机制以及增量训练策略的效果及必要性，验证了所提模型的准确性与普适性，Increformer模型能够有效捕捉股票价格序列的趋势与波动。

关键词：时间序列预测；Transformer模型；增量学习；持续注意力机制文献标志码：A 中图分类号：TP391

# Research on Stock Price Prediction Integrating Incremental Learning and Transformer Model

CHEN Dongyang, MAO Li+   
Jiangsu Engineering Laboratory of Pattern Recognition and Computational Intelligence, School of Artificial Intelli  
gence and Computer Science, Jiangnan University, Wuxi, Jiangsu 214122, China

Abstract: Stock price prediction has always been a focal topic in financial research and quantitative investment. Currently, most deep learning models for stock price prediction are based on batch learning settings, which require prior knowledge of the training dataset. These models are not scalable for real-time data stream prediction, and their performance decreases when the data distribution dynamically changes. To address the issue of poor prediction accuracy for non- stationary stock price data in existing research, this paper proposes an online stock price prediction model (Increformer) based on incremental learning and continuous attention mechanism. By leveraging continuous self-attention mechanism to capture the temporal dependencies among feature variables and employing continuous normalization mechanism to handle non-stationary data, the model enhances prediction accuracy through the incremental training strategy based on elastic weighting consolidation to acquire new knowledge from the data stream. Five public datasets are selected from the stock index and individual stock price sequences in the stock market for experiments. Experimental results demonstrate that Increformer effectively extracts temporal information and feature dimension correlation from the data, thus improving the prediction performance of stock prices. Additionally, ablation experiments are conducted to evaluate the effects and necessity of the continuous normalization mechanism, continuous attention mechanism, and incremental training strategy, thereby verifying the accuracy and generalizability of the proposed model. Increformer can effectively capture the trends and fluctuations of stock price series. Key words: time series prediction; Transformer; incremental learning; continuous attention

随着市场经济的快速发展，股票市场规模不断扩大，参与股票投资的人越来越多。预测股票价格的变化趋势有助于投资者作出正确决策。然而，股票价格作为一种金融时间序列数据受多种因素影响，具有高维度、非平稳和随机波动等特点。准确预测股票价格以提高投资者收益是一个长期且具挑战性的问题。

近年来，国内外学者针对股票价格预测方法取得了大量研究成果。传统的统计学方法，如自回归积分移动平均模型（autoregressive integrated movingaverage model，ARIMA）[1] 和广义自回归条件异方差模型（generalized autoregressive conditional heteroske-dasticity，GARCH）[2]，由于自身线性结构的约束，对于股票价格的预测效果不佳。基于传统机器学习的方法，如支持向量机[3]、随机森林[4]等，在预测过程需要对数据进行特征选择，容易产生过拟合、泛化能力较弱以及陷入局部最优等问题。随着大数据的发展，深度学习在时间序列领域提取特征的能力取得了突破进展[5]，对于股票价格预测，深度学习方法可以通过其复杂的网络结构学习到股票价格数据中的隐藏信息，结合时序信息与特征间的关联信息更准确地进行股价预测。长短期记忆网络（long short-term memory，LSTM）[6]通过引入门控机制重新设计循环神经网络的网络结构，有效解决了长序列训练过程中的梯度消失和梯度爆炸问题，从而更适用于时序预测领域。Lu等[7]提出一种利用卷积神经网络提取数据的特征，并结合双向长短期记忆网络（bi-directional longshort-term memory，BiLSTM）预测股票价格，为投资者的股票投资决策提供可靠的参考。Transformer[8]模型采用编码-解码框架，最初被应用于自然语言处理领域。由于其自注意力机制能够实现对时间序列特征的高效提取，Zhou 等[9] 提出的 Informer 模型对注意力机制做了改进，降低其时间复杂度并将Transformer模型引入了长序列预测领域，实现较好的长序列预测效果。Wu 等人[10] 也对 Transformer 模型进行了有效改进，降低了 Transformer的计算成本，并在长序列预测领域超越了 LSTM 系列的预测精度。Ding 等人[11]提出一种分层多尺度高斯改进的 Transformer以捕捉股票时间序列的长期及短期依赖性。Daiya等人[12]使用融合卷积和Transformer模型从财务指标和新闻数据中提取特征，提高了股价预测的准确性。

上述研究采用的不同模型和方法都是通过大量历史数据进行建模及训练，得到较高的预测精度。然而，这些研究都集中在批处理学习的设置上，这要求整个训练数据集都是先验的，意味着输入和输出之间的关系始终保持静态。然而，当数据流逐渐到达时，数据分布可能随着时间推移而改变，模型的预测精度将下降。从头开始训练模型可能非常耗时，而且由于隐私安全等问题，历史数据可能难以获取。因此，结合增量学习对模型实时更新就变得尤为重要。然而，灾难性遗忘的问题在此时容易出现，即在新任务的数据集上训练，会使得模型在旧数据集上的性能大幅度下降。对这个问题的解决可能会陷入“稳定性-可塑性”困境，即如何在保持旧数据的预测精度下学习新数据的知识，增量学习领域要克服这个问题主要的三种范式为正则化（regularization）、回放（replay）、参数隔离（parameter isolation）。代表性的算法如弹性权重巩固（elastic weight consolidation，EWC）[13]算法是一种正则化方法，还有基于回放机制的经验回放（experience replay，ER）[14] 与 DER $^ { + + }$ （darkexperience replay）[15] 算法，以及基于参数隔离的 Pack-net[16]算法。然而增量学习目前主要研究集中在类增量任务，在回归预测领域缺乏足够深入的研究。

Wang 等人[17] 提出基于极限学习机（extreme learn-ing machine，ELM）结合增量训练方法实现集成在线学习。在 ELMK（extreme learning machine with kernels）作为基模型的基础上，Yu等人[18]提出了DWE-IL（in-cremental learning algorithm via dynamically weightingensemble learning）算法，对模型参数进行了动态的权重更新，使得模型具有较好的泛化性能。然而随着时间的推移，集成模型中不同模型的差异可能导致误差增大。除了使用集成学习的思想外，还可以对深度学习模型的网络结构进行修改并适用于在线学习的场景。Wang 等人[19] 提出的 IncLSTM（incrementalensemble LSTM model）是一个增量式 LSTM 模型，通过对LSTM网络结构的优化实现增量式预测。Woo等人 [20] 基 于 时 间 卷 积 网 络（temporal convolutional net-work，TCN）作为主干进行改进，使用 online-TCN 模型保持对时序数据预测精度的同时进行增量更新。Huang 等人[21] 使用贝叶斯估计对 Transformer 模型进行改进优化，使得模型更适用于流式数据的计算。

结合以上背景，本文提出了一种增量式在线股价预测模型Increformer，用于处理非平稳的股票价格预测任务。通过多头持续注意力机制挖掘股票价格与特征因子之间的时序依赖关系，使用持续归一化机制对数据流进行平稳化操作，并使用改进的弹性权重巩固算法实现模型的增量训练。实验选取沪深300指数、中证500指数以及3支典型个股作为数据集，对比分析Increformer与其他增量学习和深度学习的基线模型预测结果。通过消融实验评估了模型中各模块的效果与必要性，验证了Increformer模型应用于股价预测的有效性，并具有一定的实际应用价值。

# 1 理论基础

# 1.1 在线时间序列预测

设 ${ \pmb \chi } = ( { \pmb x } _ { 1 } , { \pmb x } _ { 2 } , \cdots , { \pmb x } _ { L } ) \in { \bf R } ^ { L \times n }$ 是长度为 $L$ 的时间序列，每个观测值 $\pmb { x } _ { i }$ 的维度为 $n$ ，记为一个 token 。时间序列预测的目标为对于给定回顾窗口 ${ \boldsymbol { \chi } } _ { i , e } = ( { \pmb { x } } _ { i - e + 1 } ,$ ,$\pmb { x } _ { i - e + 2 } , \cdots , \pmb { x } _ { i } )$ ，预测未来 $H$ 步的时间序列，即预测窗口，最大程度减小预测值与真实值之间的差异，具体描述为：

$$
f _ { \omega } ( \pmb { \chi } _ { i , L } ) = ( \pmb { x } _ { i + 1 } , \pmb { x } _ { i + 2 } , \cdots , \pmb { x } _ { i + H } )
$$

其中， $\omega$ 为预测模型的参数。然而，考虑到数据以流的形式顺序到达更符合现实场景，特别是在金融时间序列具有非平稳、高度复杂和随机波动等特性的情况下，对于在线时间序列预测任务，模型学习知识的过程是在一系列回合中进行的。在每一轮中，模型收到一个回顾窗口，并对预测窗口进行预测，然后揭示真实值以改进模型对即将到来的回合的预测。即模型从连续输入的数据流中逐步学习新知识，模型的训练与预测交替进行。

# 1.2 EWC增量学习算法

EWC增量学习算法是对网络参数进行选择性正则化的方法。它使用 Fisher 信息矩阵来确定参数空间中对于执行己学习任务至关重要的方向，通过限制重要参数以便保留历史参数，找出对某个任务特别重要的部分权重，降低这些权值的学习率。因此，这些方向中的参数可以自由移动而不会忘记已学习任务的方向。

基于贝叶斯规则，假设输入的数据集为 $D$ ，从参数 $p ( \theta )$ 的先验概率和数据 $p ( D | \theta )$ 的概率获得条件概率 $p ( \theta | D )$ ，可以得到：

$$
\log _ { a } p ( \theta | D ) { = } \log _ { a } p ( D | \theta ) + \log _ { a } p ( \theta ) { - } \log _ { a } p ( D )
$$

如果数据已给定参数，那么可以将它的对数概率定义为负损失函数。假设数据划分为互不相干的两个部分，分别用 $D _ { s } \setminus D _ { B }$ 表示。对上述公式进行修改得到：

$$
\begin{array} { r } { \log _ { a } p ( \theta | D ) = \log _ { a } p ( D _ { \scriptscriptstyle B } | \theta ) + \log _ { a } p ( \theta | D _ { \scriptscriptstyle A } ) - \log _ { a } p ( D _ { \scriptscriptstyle B } ) } \end{array}
$$

式（3）左侧描述为给定整个数据集的参数的后验概率，而右侧仅取决于任务 $B$ 的损失函数，任务 $A$ 的所有信息必须被吸收到后验分布 $p ( \theta | D _ { A } )$ 中。然而，后验概率难以准确计算，可以通过 Fisher 信息矩阵的对角线给出的参数 ${ \boldsymbol { \theta } } _ { A } ^ { * }$ 和对角线的精度近似描述髙斯分布计算真实的后验概率。后验概率可用来表示每个子任务，给定后验概率的近似值，在 EWC中对损失函数 $L ( \theta )$ 进行最小化：

$$
L ( \theta ) { = } L _ { _ B } ( \theta ) + \sum _ { i } { \frac { \lambda } { 2 } } F _ { i } ( \theta _ { i } - \theta _ { _ { A , i } } ^ { * } ) ^ { 2 }
$$

其中， $L _ { \scriptscriptstyle B } ( \theta )$ 仅是任务 $B$ 的损失，每个参数 $i$ 的标签为$\lambda$ ，它表示旧任务相比于新任务的重要程度。

# 2 Increformer 时序预测模型框架

Increformer是一个结合深度学习与增量学习的时序预测框架。模型部分以Transformer模型为基础架构，使用输入编码模块提取时序特征获得输入向量作为编码器的输入，编码器与解码器使用更适用于增量场景的持续注意力机制，同时将持续归一化机制替代层归一化。编码器提取输入的时间序列特征，解码器利用获取的特征信息进行时序数据的预测，并将其输入至一个线性全连接层以获得最终的预测输出。此外，训练框架通过改进的时序弹性权重 巩 固 算 法（time series elastic weight consolidation，TS-EWC）对模型进行更新，使模型持续学习新知识并提升预测精度。Increformer 股票价格预测模型框架如图1所示。

# 2.1 输入编码

模型的输入编码采用类似 Informer[9]中的设置，包括三部分：数值编码（data embedding）、位置编码（position embedding）和 时 间 戳 编 码（timestamp em-bedding）。使用卷积核为3的一维卷积网络将输入序列 $X _ { \mathrm { i n } }$ 由 $d _ { \mathrm { i n } }$ 维的输入空间投影到 $d _ { \mathrm { m o d e l } }$ 维的模型空间以对齐模型的数据维度，得到数值编码向量。同时，为了使模型捕获输入序列的相对位置信息，加入对$X _ { \mathrm { i n } }$ 的位置编码来得到输入的顺序特征。除了输入的数值特征外， $X _ { \mathrm { i n } }$ 对应的时间戳也是输入特征的一部分，故引入对时间戳的编码，为输入引入了时间的全局特征。价格序列的时间戳包含年、月、日、星期等信息，依次对其进行编码后合并。将数值编码、位置编码和时间戳编码进行融合后，得到最终的输入编码向量：

![](images/ab3406f49d48f7e3d02d4388ac06e6ea29ae8d283a4e207dfaad051b707f3282.jpg)  
图 1 Increformer 模型框架  
Fig.1 Increformer model structure

$$
X _ { \mathrm { e m b } } = d a t a _ { \mathrm { e m b } } + p o s _ { \mathrm { e m b } } + t i m e _ { \mathrm { e m b } }
$$

# 2.2 编码器与解码器

Increformer的编码器由多个相同的层堆叠组成，每层包含 3 个子层，分别为多头持续注意力机制（continual multi-head attention，ConMHA）、前馈神经网络（feed- forward network，FFN）和持续归一化机制（continual normalization，CN）。原始输入序列 $X _ { \mathrm { i n } }$ 经过输入编码模块进行转换后得到编码器的输入序列 Xemb 。

Increformer的解码器与编码器类似，由多个相同的层组成，但子层中新增了一组带 Mask 的多头持续注意力模块，根据编码器的输出 $K$ 和V 为解码器的中间结果 Q 赋予不同的权重，以计算 $K$ 和 $Q$ 之间的相关程度。解码器中也是使用持续归一化 CN 对网络模型子层间的数据进行处理。解码器的输入向量为：

$$
X _ { \mathrm { d e } } = C o n c a t ( X _ { \mathrm { t o k e n } } , X _ { \mathrm { 0 } } )
$$

其中， $X _ { \mathrm { { t o k e n } } }$ 是长度为 $T _ { \mathrm { t o k e n } }$ 的历史序列初始输入， $X _ { \mathfrak { o } }$ 是 预 测 长 度 为 $H$ 的 序 列 占 位 符 ，采 用 零 填 充 。Concat() 表示元素的拼接， $X _ { \mathrm { d e } }$ 输入解码器后，经过一系列网络层的计算，最终通过线性全连接层来调整输出长度，得到最终的预测结果。

# 2.3 多头持续注意力

Transformer中的自注意力机制能够处理数据间的相似关系，挖掘并保留元素之间的重要性关联。自注意力机制采用查询-键值-数值（query-key-value，QKV）模式计算每个元素与其他元素之间的相似度提取时序依赖关系，计算方式描述为：

$$
A t t ( Q , K , V ) = D ^ { - 1 } A V
$$

$$
A = \exp ( Q K ^ { \mathrm { T } } / \sqrt { d } ) , D = \mathrm { d i a g } ( A \cdot \mathbb { I } _ { L } )
$$

其中， ${ \boldsymbol { Q } } \in \mathbf { R } ^ { L \times d _ { \mathrm { m o d e l } } }$ , $\pmb { K } \in \mathbf { R } ^ { L \times d _ { \mathrm { m o d e l } } }$ , $V \in \mathbf { R } ^ { L \times d _ { \mathrm { m o d e l } } }$ RL × dmodel ，d 是 模 型隐藏层的维度， $\mathbb { I } _ { L }$ 是元素全为 1 的列向量，这里 $\exp ( \cdot )$ 以元素的方式应用。模型的时间复杂度为 $\mathcal { O } ( n L ^ { 2 } )$ ，空间复杂度为 $\mathcal { O } ( L ^ { 2 } )$ 。 $L$ 与 $n$ 分别为时序数据的观测长度和观测维度。

考虑到在线时间序列预测的场景，Increformer通过步进的持续更新计算以适应数据流的顺序到达。对于每个时间步，通过丢弃最早的 token ，以先进先出的方式添加新 token 从而更新 $Q \ , K$ 和 $V$ ，在这种情况下注意力分数可以通过缓存的历史结果以及最新的查询、主键和数值 $( \pmb q _ { \mathrm { n e w } } , \pmb k _ { \mathrm { n e w } } , \pmb v _ { \mathrm { n e w } } \in { \bf R } ^ { 1 \times n } )$ 计算获得，通过矩阵的乘法计算可有效实现 $D ^ { - 1 }$ 的更新。缓存前一个时间步的 $L - 1$ 个值，设为 $d _ { \mathrm { m e m } } = A _ { \mathrm { p r e } } ^ { ( 2 : L ) } \mathbb { I } _ { L - 1 }$ ，更新

方式为：

$$
d ^ { ( 1 : L - 1 ) } = d _ { \mathrm { m e m } } ^ { ( 2 : L ) } - \mathrm { e x p } { \left( \frac { Q _ { \mathrm { m e m } } k _ { \mathrm { o l d } } ^ { \mathrm { T } } } { \sqrt { d _ { \mathrm { m o d e l } } } } \right) } + \mathrm { e x p } { \left( \frac { Q _ { \mathrm { m e m } } k _ { \mathrm { n e w } } ^ { \mathrm { T } } } { \sqrt { d _ { \mathrm { m o d e l } } } } \right) }
$$

$$
\pmb { d } ^ { ( L ) } = \exp \left( \pmb { q } _ { \mathrm { n e w } } \frac { C o n c a t ( \pmb { K } _ { \mathrm { m e m } } , \pmb { k } _ { \mathrm { n e w } } ) } { \sqrt { d _ { \mathrm { m o d e l } } } } \right) \mathbb { I } _ { L }
$$

其中， $Q _ { \mathrm { { m e m } } }$ 、 $K _ { \mathrm { { m e m } } }$ 分别为前 $L - 1$ 个缓存 token 的查询与键值， $\boldsymbol { k } _ { \mathrm { o l d } }$ 为前 $L$ 步长处的 token 键值。 $A V$ 可同样根据先验的 $A V _ { \mathrm { { m e m } } }$ 更新求得：

$$
A V ^ { ( 1 : L ^ { - 1 } ) } = A V _ { \mathrm { m e m } } ^ { ( 2 : L ) } - \exp \left( \frac { Q _ { \mathrm { m e m } } k _ { \mathrm { o l d } } ^ { \mathrm { T } } } { \sqrt { d _ { \mathrm { m o d e l } } } } \right) { \pmb v } _ { \mathrm { o l d } } +
$$

$$
\exp \left( \frac { Q _ { \mathrm { m e m } } { k _ { \mathrm { n e w } } ^ { \mathrm { T } } } } { \sqrt { d _ { \mathrm { m o d e l } } } } \right) \pmb { v } _ { \mathrm { n e w } }
$$

$$
\mathrm { e x p } \Bigg ( \pmb { q } _ { \mathrm { n e w } } \frac { C o n c a t ( { K } _ { \mathrm { m e m } } , \pmb { k } _ { \mathrm { n e w } } ) } { \sqrt { d _ { \mathrm { m o d e l } } } } \Bigg ) C o n c a t ( { V } _ { \mathrm { m e m } } , \pmb { v } _ { \mathrm { n e w } } )
$$

最终得到持续注意力输出描述为：

$$
C o n A t t ( { \pmb q } _ { \mathrm { n e w } } , { \pmb k } _ { \mathrm { n e w } } , { \pmb v } _ { \mathrm { n e w } } ) = { \pmb d } ^ { - 1 } \odot A V
$$

改进后的持续注意力在每一个时间步的时间复杂度与空间复杂度均为 $\mathcal { O } ( n L )$ ，更适用于流式计算的同时，也极大地减小了计算开销。

在实际操作过程中则采用多头注意力机制通过多个不同的 $Q \ , K$ 和 $V$ 将输入分别投影到 $h$ 个不同的子空间，增强网络模型捕捉序列不同维度依赖关系的能力，给定一组新的查询-键值-数值，则多头持续 注 意 力 机 制（continual multi- head attention，Con-MHA）可描述为：

$$
C o n M H A ( q , k , v ) = C o n c a t ( h e a d , h e a d _ { 2 } , \cdots , h e a d _ { h } ) { \mathbb { W } } _ { o }
$$

$$
h e a d _ { i } = C o n A t t ( q W _ { \varrho } ^ { i } , k W _ { \kappa } ^ { i } , \upsilon W _ { \nu } ^ { i } )
$$

其 中 ， ${ \pmb W } _ { \varrho } ^ { i } \in \mathbf { R } ^ { d _ { \mathrm { m o d e l } } \times d _ { k h } } , { \pmb W } _ { \kappa } ^ { i } \in \mathbf { R } ^ { d _ { \mathrm { m o d e l } } \times d _ { k h } } , { \pmb W } _ { \nu } ^ { i } \in \mathbf { R } ^ { d _ { \mathrm { m o d e l } } \times d _ { k h } } , { \pmb W } _ { o } \in$ Rd v × dmodel 。

# 2.4 持续归一化机制

Transformer 采 用 的 层 归 一 化（layer normaliza-tion，LN）适合于处理序列化变长数据，针对单个样本的不同特征做操作，只考虑样本维度的归一化。组归一化（group normalization，GN）[22] 则是将单个样本的不同特征分组进行归一化，LN 即分组为 1 时的 GN，批归一化（batch normalization，BN）[23] 则根据不同样本的同一特征进行归一化，有助于网络优化更快更好地收敛，标准归一化计算公式为：

$$
y = \gamma \times \frac { x - \mu } { \sqrt { \sigma ^ { 2 } + \epsilon } } + \beta
$$

式中， $\pmb { \mu }$ 和 $\pmb { \sigma } ^ { 2 }$ 是根据输入特征计算的均值和方差， $\epsilon$ 是为了避免被除零而添加的常数， $\gamma$ 和 $\beta$ 是仿射变换参数，其取值在训练过程中通过反向传播和优化算法进行学习获得，初始值设定为 $\gamma = 1$ ， $\beta = 0$ ，仿射变换后的数据范围由具体的数据集决定。而BN与GN中均值与方差计算方式如式（17）\~（19）所示：

$$
\begin{array} { r l } & { \mu _ { \mathrm { a s } } = \displaystyle \frac { 1 } { B H W } \displaystyle \sum _ { k = 1 } ^ { B } \sum _ { w = 1 } ^ { W } \lambda _ { w = s } \sigma _ { \mathrm { r s h } } ^ { 2 } = } \\ & { \qquad \quad \frac { 1 } { B H W } \displaystyle \sum _ { k = 1 , w = 1 , k = 1 } ^ { B } ( \boldsymbol { x } _ { \mathrm { s s u b s } } - \boldsymbol { \mu } _ { \mathrm { a s s } } - \boldsymbol { \mu } _ { \mathrm { a s s } } ) ^ { 2 } } \\ & { \qquad \quad \boldsymbol { x } _ { \mathrm { b a s h } } ^ { \prime }  \boldsymbol { x } _ { \mathrm { s u b s } } \mathrm { , ~ w h e r e ~ } k = \lfloor \frac { 1 } { B } \rfloor _ { G } ^ { C } } \\ & { \mu _ { \mathrm { 0 s } } ^ { ( \alpha ) } = \displaystyle \frac { 1 } { m } \sum _ { i = 1 } ^ { B } \sum _ { w = 1 } ^ { B } \sum _ { k = 1 } ^ { m } \boldsymbol { x } _ { \mathrm { s u b s } } ^ { \prime } \sigma _ { \mathrm { c s } } ^ { \prime } = } \\ & { \qquad \quad \frac { 1 } { m } \sum _ { i = 1 } ^ { B } \sum _ { w = 1 , k = 1 } ^ { B } ( \boldsymbol { x } _ { \mathrm { s u b s } } ^ { \prime } - \boldsymbol { \mu } _ { \mathrm { c s s } } ^ { ( \alpha ) } - \boldsymbol { \mu } _ { \mathrm { c s } } ^ { ( \alpha ) } ) ^ { 2 } } \end{array}
$$

对时序场景BN容易破坏时序维度的关联，结合GN和BN的优势提出适用于数据增量场景的持续归一化机制（continual normalization，CN），其工作原理是先使用GN对特征映射执行空间归一化，然后通过BN进一步对特征归一化。CN的计算方式可描述为：

$$
{ \pmb x } _ { \mathrm { o N } }  G N _ { _ { 1 , 0 } } ( { \pmb x } ) ; { \pmb x } _ { \mathrm { c N } }  \gamma B N _ { _ { 1 , 0 } } ( { \pmb x } _ { \mathrm { G N } } ) + \beta
$$

在增量学习场景，由于数据顺序到达的特性CN将式（17）中涉及的小批均值和方差替换为训练过程的全局值的估计值，可描述为：

$$
\pmb { \mu }  \pmb { \mu } + \eta ( \pmb { \mu } _ { b } - \pmb { \mu } ) , \pmb { \sigma }  \pmb { \sigma } + \eta ( \pmb { \sigma } _ { b } - \pmb { \sigma } )
$$

# 2.5 TS-EWC增量学习算法

在 Increformer 框架中模型的更新方法和触发条件起着至关重要的作用。针对股票价格数据具有非平稳特性的情况，当数据分布发生变化时，模型需要增量学习新数据的知识。对于在线时间序列预测任务，基于 EWC算法进行了改进优化，提出时序弹性权重巩固算法（TS-EWC）。每一次数据分布发生变化时，将新数据作为一个新的任务去进行学习，那么随着时间的推进，EWC会为数据流产生的每个历史任务都维护一个惩罚项，惩罚数量随着任务数量线性增长造成较大的计算开销，而每个新任务都是对上一个旧任务施加惩罚项得来的，因此在更新模型时只需要维护上一个惩罚项，并对前面的历史任务产生的Fisher矩阵进行加权求和，TS-EWC中损失函数具体描述为：

$$
L ( \theta ) = L _ { _ { D } } ( \theta ) + \sum _ { i } \frac { \lambda } { 2 } \sum _ { d < D } \lambda _ { d } F _ { _ { i , i } } ( \theta _ { i } - \theta _ { _ { D - 1 , i } } ^ { * } ) ^ { 2 }
$$

其中， $L _ { p } ( \theta )$ 为当前的任务损失，更新模型的触发条件则设为一个超参数，设定两个缓冲区，将均方误差作为阈值与数据进行比较，标记数据的新颖性和熟悉性。新颖性缓冲区检测数据流概率分布的变化，并触发对模型进行训练更新的操作，同时在更新后动态调整阈值；熟悉性缓冲区则包含模型熟悉的信息，可以测试模型更新后是否保留旧的知识，未来的数据流中若出现重复模式则可实现快速准确的预测。

# 3 实验结果及分析

# 3.1 数据集与实验设置

实验分别对指数与个股股票价格进行预测。从同花顺iFinD金融数据库获取国内股票市场数据，包括沪深 300 指数（000300.SH）和中证 500 指数（399905.SZ）在 2007 年 1 月 16 日至 2023 年 12 月 29 日期间的日交易数据。鉴于市场股票数量众多，为提升实验分析的普适性与客观性，本文从两个指数成分股中再选取金融、传媒和医药这3个不同热门行业的代表性股票，分别是平安银行、中文传媒和国药股份。选取这些股票在 2010 年 1 月 4 日至 2023 年 12 月 29 日期间的日交易数据。每条数据包含异同移动平均线（moving average convergence divergence，MACD）、涨跌幅（chg_ratio）、开盘价（open）、收盘价（close）、最高价（high）、最低价（low）等信息，部分数据及其统计特征如表1所示。

表 1 沪深 300指数部分数据  
Table 1 Partial data display of 000300.SH   

<table><tr><td>Date</td><td>High</td><td>Low</td><td>Open</td><td>Close</td></tr><tr><td>2007-01-16</td><td>2354.43</td><td>2297.24</td><td>2310.96</td><td>2353.87</td></tr><tr><td>2007-01-17</td><td>2393.22</td><td>2 266.34</td><td>2360.41</td><td>2308.93</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>2023-12-28</td><td>3423.40</td><td>3331.21</td><td>3335.56</td><td>3414.54</td></tr><tr><td>2023-12-29</td><td>3432.54</td><td>3409.58</td><td>3411.11</td><td>3431.11</td></tr><tr><td>mean</td><td>3473.68</td><td>3505.68</td><td>3441.08</td><td>3476.22</td></tr><tr><td>std</td><td>876.13</td><td>881.74</td><td>866.89</td><td>875.33</td></tr><tr><td>median</td><td>3437.49</td><td>3466.57</td><td>3413.22</td><td>3440.97</td></tr><tr><td>min</td><td>1614.62</td><td>1648.45</td><td>1606.73</td><td>1627.76</td></tr><tr><td>max</td><td>5922.07</td><td>5930.91</td><td>5815.61</td><td>5877.20</td></tr></table>

为了消除数据各个特征维度不同量纲的影响，对序列进行归一化处理，数据集可分为预热阶段、验证阶段和在线预测阶段，比例为2∶1∶7。在预热阶段和验证阶段对模型进行预训练，在线预测阶段则将数据集顺序输入模型，模拟不断增长的数据流。回顾窗口长度设置为 60，预测窗口设置预测步长为

24。在每轮预测之后，将真实值与预测值进行比较。实验模型是基于 Nvidia Tesla T4 GPU 进行搭建并计算的。Increformer 由 $N { = } 2$ 的编码器以及 $M { = } 1$ 的解码器组成。预热阶段采用学习率为0.000 1的Adam优化 器 ，损 失 函 数 选 择 均 方 误 差（mean squared error，MSE），在线预测阶段使用 TS-EWC 算法计算模型的损失函数，并进行模型的增量更新。

# 3.2 评价指标

为了更直观地体现Increformer模型的预测性能，采用平均绝对误差（mean absolute error，MAE）、均方误 差（MSE）、均 方 根 误 差（root mean square error，RMSE）和平均绝对百分比误差（mean absolute percent-age error，MAPE）这 4 种回归评价指标对预测结果进行综合评价。其计算公式可描述为：

$$
M A E = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \left| \hat { y } _ { n } - y _ { n } \right|
$$

$$
M S E = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } { ( \hat { y } _ { n } - y _ { n } ) } ^ { 2 }
$$

$$
R M S E = \sqrt { \frac { 1 } { N } { \sum _ { n = 1 } ^ { N } ( \hat { y } _ { n } - y _ { n } ) } ^ { 2 } }
$$

$$
M A P E = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \frac { \left| \hat { y } _ { n } - y _ { n } \right| } { y _ { n } }
$$

式中， $n$ 为总样本量， $\boldsymbol { \mathcal { Y } } _ { n }$ 为真实值， $\hat { y } _ { n }$ 为预测值，回归评价指标 MAE、MSE、RMSE和 MAPE的值越小，则预测值与真实值的差值越小，预测结果越准确。

# 3.3 实验结果分析

# 3.3.1 股指预测分析

为验证Increformer预测框架的性能，本文在深度学习和增量学习领域选取以下模型进行股票价格预测的对比实验：LSTM[6] 、Transformer[8] 、Informer[9] 、IncLSTM[19] 、DWE-IL[18] 、OnlineTCN[20] 、ER[14] 和 DER $^ { + + }$ [15] 。其中，LSTM、Transformer 和 Informer 采用离线批处理设置，在预热阶段训练模型。损失函数选择 MSE，优化器选择 Adam，批处理大小设置为 64。此外，LSTM 隐含层单元数量设置为 50，激活函数为 ReLU。Transformer 与 Informer 使用 LN 进行网络层的归一化操作，注意力机制分别为 LogSparse attention 和 Prob-Sparse attention。 IncLSTM、DWE- IL 和 OnlineTCN3 种模型均进行集成增量学习，随着在线预测阶段真实值的不断揭露，持续更新模型。IncLSTM 基模型选取 LSTM，使用 Tradaboost 方法对 LSTM 进行集成学 习 ，迭 代 次 数 设 为 50，Tradaboost number 设 为 2。DWE-IL的基模型为 ELMK，选取 RBF Kernel为核函数，回顾窗口长度设为 60。OnlineTCN 使用标准TCN主干，设置10个隐藏层，每个隐藏层都有2个残差卷积滤波器。ER、DER $^ { + + }$ 方法使用 Transformer 作为主干网络进行在线预测，并保持与Increformer相同的缓冲区大小以回顾旧样本。采用Increformer与上述模型对沪深 300 指数和中证 500 指数进行预测。为确保模型比较的有效性，同一数据集在使用模型进行训练前，使用相同的归一化操作进行预处理，由于Transformer类的网络模型使用的归一化机制中涉及仿射变换，参数是在网络层中学习的。只有相同的输入价格序列，其相关参数才相同。此外，以 Transformer 的仿射变换参数为基准，对 LSTM、ELM和TCN等其他网络模型的预测结果进行仿射变换后再进行比较。为避免实验结果被偶然因素影响，进行20次预测实验并计算评价指标的平均值。在线预测阶段各模型预测结果如表2所示。

# 表 2 沪深 300指数与中证 500指数预测结果

Table 2 Prediction results of 000300.SH and 399905.SZ   

<table><tr><td>指数</td><td>Model</td><td>MAE</td><td>MSE</td><td>RMSE</td><td>MAPE</td></tr><tr><td rowspan="9">沪深300</td><td>LSTM</td><td>0.426</td><td>0.981</td><td>0.678</td><td>1.789</td></tr><tr><td>Transformer</td><td>0.402</td><td>0.879</td><td>0.729</td><td>1.672</td></tr><tr><td>Informer</td><td>0.322</td><td>0.712</td><td>0.742</td><td>1.644</td></tr><tr><td>IncLSTM</td><td>0.362</td><td>0.763</td><td>0.678</td><td>1.498</td></tr><tr><td>DWE-IL</td><td>0.331</td><td>0.542</td><td>0.615</td><td>1.484</td></tr><tr><td>OnlineTCN</td><td>0.334</td><td>0.533</td><td>0.607</td><td>1.498</td></tr><tr><td>ER</td><td>0.259</td><td>0.325</td><td>0.442</td><td>1.178</td></tr><tr><td>DER++</td><td>0.229</td><td>0.295</td><td>0.427</td><td>0.899</td></tr><tr><td>Increformer</td><td>0.225</td><td>0.274</td><td>0.431</td><td>0.742</td></tr><tr><td rowspan="8">中证500</td><td>LSTM</td><td>0.415</td><td>0.622</td><td>0.718</td><td>1.160</td></tr><tr><td>Transformer</td><td>0.379</td><td>0.451</td><td>0.696</td><td>1.158</td></tr><tr><td>Informer</td><td>0.356</td><td>0.456</td><td>0.600</td><td>1.017</td></tr><tr><td>IncLSTM</td><td>0.332</td><td>0.339</td><td>0.519</td><td>0.967</td></tr><tr><td>DWE-IL</td><td>0.338</td><td>0.355</td><td>0.530</td><td>0.912</td></tr><tr><td>OnlineTCN</td><td>0.327</td><td>0.346</td><td>0.510</td><td>0.989</td></tr><tr><td>ER</td><td>0.251</td><td>0.235</td><td>0.401</td><td>0.728</td></tr><tr><td>DER++</td><td>0.233</td><td>0.234</td><td>0.395</td><td>0.667</td></tr><tr><td>Increformer</td><td></td><td>0.228</td><td>0.215</td><td>0.384</td><td>0.637</td></tr></table>

由表 2 可知，LSTM、Transformer 和 Informer 的评估指标结果显著偏高。LSTM 作为一种经典的神经网络模型，在时序数据预测问题上有效解决了训练过程梯度消失和梯度爆炸的问题。而Transformer和Informer基于自注意力机制能有效提取时序信息，减小预测误差。但对于在线预测场景，这 3个模型的评估指标表现不佳。这是因为它们仅在预热阶段采用离线批处理模式训练模型，随着预测数据的增加，模型无法有效预测新的数据分布。对于股票价格这类非平稳、波动性强的时序数据而言，当预测数据远多于训练数据时，预测性能显著降低。IncLSTM是基于LSTM改进的增量式集成模型，其在线预测阶段不断学习新数据并使用多个模型组合预测。此外，DWE-IL 与 OnlineTCN 也分别在 ELM 和 TCN 网络架构上进行了模型的增量训练，大大提升了预测效果。然而，集成模型受到基模型及其相关参数影响，当数据分布差异较大时，基模型之间的不同预测结果降低了在线预测阶段的预测性能。ER与 $\mathrm { D E R + + }$ 采用回放机制使用缓冲区存储以前的部分数据，并在学习新数据时穿插旧样本。结合 Transformer捕获时间序列的特征，提高了模型在线预测阶段的长期预测能力。相较于经典 Transformer模型评估指标均得到了有效提升，在沪深300指数与中证500指数两个数据集上，Increformer的各项评价指标上表现最优。较于其他模型，它具有更强的特征提取能力以及长期预测性能，预测效果显著优于上述基线模型。

图2、图3分别为上述模型对沪深300指数与中证500指数在线预测阶段的部分预测结果。图中展示了收盘价的预测值和真实值，并经过标准化处理。在图 2 和图 3 中，LSTM、Transformer 和 Informer的预测结果在部分波段波动较大，甚至出现趋势相反的情况。这是因为部分数据的分布模式在预热阶段并未出现，静态模型无法适应新数据的变化趋势。IncLSTM、DWE-IL 和 OnlineTCN 由于增量学习的设置，表现有所提高，但曲线整体拟合度不高，在中证500指数上出现曲线变化幅度较大时在拐点预测效果不佳的问题，仍有较大提升空间。基于回放机制的ER和 $\mathrm { D E R + + }$ 方法由于采用 Transformer 作为主干网络，预测效果与真实值较为接近，且总体趋势拟合程度较高。综上，Increformer 拟合效果最优，预测值与真实值两条曲线最为接近。

# 3.3.2 个股预测分析

在实际环境中，个股股票价格的随机波动较大，且受多种因素影响，预测较为困难。为进一步验证模型的准确性，从中证500指数成分股中选取平安银行（000001.SZ），从沪深300指数成分股中选取中文传媒（600373.SH）和国药股份（600511.SH），采用 Incre-former以及基线模型对这3支股票进行个股价格预测。多次实验计算在线预测阶段评估指标的平均值，结果如表 3\~表 5所示。

![](images/8ad63b7526b351328129525ff2f164dd7689ddccb9cfe70e5722ca916e2d2e9d.jpg)  
图 2 沪深 300指数各模型预测结果比较

![](images/43b5eb3aa1c21fd88fef470f883772a1cd7204846ec138d860b873f58ab97a5c.jpg)  
Fig.2 Prediction results of 000300.SH by each model   
图 3 中证 500指数各模型预测结果比较  
Fig.3 Prediction results of 399905.SZ by each model

# 表 3 不同模型对平安银行价格预测结果

Table 3 Prediction results of 000001.SZ by each model   

<table><tr><td>Model</td><td>MAE</td><td>MSE</td><td>RMSE</td><td>MAPE</td></tr><tr><td>LSTM</td><td>0.416</td><td>0.764</td><td>0.729</td><td>1.746</td></tr><tr><td>Transformer</td><td>0.321</td><td>0.479</td><td>0.594</td><td>1.656</td></tr><tr><td>Informer</td><td>0.319</td><td>0.221</td><td>0.407</td><td>0.971</td></tr><tr><td>IncLSTM</td><td>0.327</td><td>0.218</td><td>0.416</td><td>0.981</td></tr><tr><td>DWE-IL</td><td>0.302</td><td>0.206</td><td>0.395</td><td>0.952</td></tr><tr><td>OnlineTCN</td><td>0.299</td><td>0.235</td><td>0.382</td><td>0.839</td></tr><tr><td>ER</td><td>0.190</td><td>0.076</td><td>0.222</td><td>0.555</td></tr><tr><td>DER++</td><td>0.146</td><td>0.047</td><td>0.169</td><td>0.528</td></tr><tr><td>Increformer</td><td>0.132</td><td>0.043</td><td>0.157</td><td>0.526</td></tr></table>

# 表 4 不同模型对中文传媒价格预测结果

Table 4 Prediction results of 600373.SH by each model   

<table><tr><td>Model</td><td>MAE</td><td>MSE RMSE</td><td>MAPE</td></tr><tr><td>LSTM</td><td>0.641</td><td>1.029 0.814</td><td>0.972</td></tr><tr><td>Transformer</td><td>0.552</td><td>0.863 0.735</td><td>0.945</td></tr><tr><td>Informer</td><td>0.527</td><td>0.842 0.706</td><td>0.795</td></tr><tr><td>IncLSTM</td><td>0.406</td><td>0.526 0.615</td><td>0.781</td></tr><tr><td>DWE-IL</td><td>0.356</td><td>0.452 0.523</td><td>0.618</td></tr><tr><td>OnlineTCN</td><td>0.289</td><td>0.235 0.338</td><td>0.439</td></tr><tr><td>ER</td><td>0.268</td><td>0.226 0.336</td><td>0.401</td></tr><tr><td>DER++</td><td>0.275</td><td>0.214 0.324</td><td>0.417</td></tr><tr><td>Increformer</td><td>0.253</td><td>0.172 0.298</td><td>0.372</td></tr></table>

# 表 5 不同模型对国药股份价格预测结果

Table 5 Prediction results of 600511.SH by each model   

<table><tr><td>Model</td><td>MAE</td><td>MSE</td><td>RMSE MAPE</td></tr><tr><td>LSTM</td><td>0.375</td><td>0.409 0.496</td><td>1.643</td></tr><tr><td>Transformer</td><td>0.364</td><td>0.367</td><td>0.469 1.488</td></tr><tr><td>Informer</td><td>0.324</td><td>0.334 0.434</td><td>1.265</td></tr><tr><td>IncLSTM</td><td>0.327</td><td>0.256</td><td>0.417 1.368</td></tr><tr><td>DWE-IL</td><td>0.213</td><td>0.132</td><td>0.305 0.964</td></tr><tr><td>OnlineTCN</td><td>0.176</td><td>0.092</td><td>0.203 0.782</td></tr><tr><td>ER</td><td>0.164</td><td>0.108 0.249</td><td>1.046</td></tr><tr><td>DER++</td><td>0.163</td><td>0.082 0.193</td><td>0.773</td></tr><tr><td>Increformer</td><td>0.160</td><td>0.081</td><td>0.188 0.759</td></tr></table>

从个股预测结果可以看出，相比于上述的基线模型，Increformer在各项评估指标上性能均为最佳。在增量场景下，它能够有效地对非平稳、高波动性的股价数据进行预测，并能够及时学习新数据动态调整模型参数，以拟合实际股价趋势。Increformer具有良好的泛化性与鲁棒性，能够为投资者的投资决策提供重要参考，从而提高投资收益。

此外，相比于表 2 的股指预测结果，Increformer模型在平安银行股价序列上的MSE误差结果最小，中文传媒股价序列的MAPE最小，在国药股份价格序列的MAPE偏高，但其他3个评价指标显著低于股指序列。从各评价指标结果的综合情况比较，本文所提模型在个股的预测效果更好，能够有效学习不同个股股票价格序列的趋势与波动。

# 3.4 消融实验

为验证所提Increformer模型的预测效果以及各功能模块的作用与效果，使用沪深 300指数数据集进行消融实验。通过移除模型中的不同模块得到变体模型，并根据评价指标MAE和MSE的计算进行对比分 析 。 首 先 ，针 对 持 续 注 意 力 模 块 ，将 其 替 换 为Transformer 和 Informer 的 注 意 力 模 块 ，包 括 Log-Sparse attention 和 ProbSparse attention。其次，将持续归一化模块去除，仅进行层归一化（LN）。最后，更换增量训练方法为EWC、ER与 $\mathrm { D E R + + }$ 。在替换各个模块的过程中，控制其他部分保持不变。实验对比结果如表 6所示。

表 6 Increformer 消融实验结果  
Table 6 Increformer ablation experiments   

<table><tr><td>Model</td><td>MAE</td><td>MSE</td></tr><tr><td>LogSparse attention</td><td>0.230</td><td>0.290</td></tr><tr><td>ProbSparse attention</td><td>0.232</td><td>0.288</td></tr><tr><td>Increformer (LN)</td><td>0.231</td><td>0.293</td></tr><tr><td>Increformer (ER)</td><td>0.245</td><td>0.315</td></tr><tr><td>Increformer (DER++)</td><td>0.237</td><td>0.305</td></tr><tr><td>Increformer (EWC)</td><td>0.228</td><td>0.285</td></tr><tr><td>Increformer</td><td>0.225</td><td>0.274</td></tr></table>

由表6可知，Increformer的持续注意力机制性能最优，比 LogSparse attention 的 MAE 和 MSE 分别降低了 $2 . 1 7 \%$ 和 $5 . 5 1 \%$ ，验证了所提模块更适用于在线预测场景以及对流式数据的处理，能够更有效地捕捉新数据的趋势特征。此外，去除持续归一化机制后，误差指标结果增大，这表明该模块对数据平稳化以及模型持续学习新数据的知识有一定帮助。对于增量学习方法的替换，可以看出EWC与TS-EWC的预测性能较好，但TS-EWC的计算开销更小，模型的更新效率更高。当数据的新模式出现得越多，即网络参数要学习的任务数不断增加时，相比于 EWC，TS-EWC 的性能就越好。综上所述，Increformer 的各个模块减小了模型的计算开销，增强了模型捕捉数据时序维度关联和特征维度关联的能力，有效提升了在线预测框架的整体性能。

# 4 结束语

为提升股票价格预测精度，针对现有深度学习模型结构随着预测时间增加、新数据不断出现，对非平稳股票价格在线预测效果不佳的问题，提出一种基于增量学习算法与Transformer框架的在线预测模型Increformer。该模型引入了持续自注意力机制挖掘特征变量之间的时序依赖关系，采用持续归一化机制为模型提供增量场景下的数据平稳化方案。同时，基于TS-EWC算法保证模型的动态更新，更有效地预测新数据。实验结果表明，Increformer的预测性能及拟合程度显著优于现有 LSTM、Transformer、In-former、IncLSTM、DWE-IL、OnlineTCN、ER以及DER $^ { + + }$ 等经典深度学习和增量学习方法。此外，实验选取的多种股指和个股股票数据验证了模型的普适性。由于金融时间序列受政治、社会、经济以及心理等多方面影响，难以进行准确的预测。本文只采用了部分技术指标进行特征提取与预测，为进一步提高模型的预测准确性，未来考虑融合财经新闻、股评舆情以及投资者情绪等相关因素。对Increformer模型的参数设置进行优化，并加入可解释性强的优化算法。为金融领域提供更为准确且实际的参考。

# 参考文献：

[1] BOX G E P, JENKINS G M, REINSEL G C, et al. Time series analysis: forecasting and control[M]. New York: John Wiley & Sons, 2015.   
[2] FRANCQ C, ZAKOIAN J M. GARCH models: structure, statistical inference and financial applications[M]. New York: John Wiley & Sons, 2019.   
[3] CAO L, GU Q. Dynamic support vector machines for nonstationary time series forecasting[J]. Intelligent Data Analysis, 2002, 6(1): 67-83.   
[4] KUMAR M, THENMOZHI M. Forecasting stock index movement: a comparison of support vector machines and random forest[J]. SSRN Electronic Journal, 2006. DOI: 10.2139/ssrn.876544.   
[5] 梁宏涛, 刘硕, 杜军威, 等. 深度学习应用于时序预测研究 综述[J]. 计算机科学与探索, 2023, 17(6): 1285-1300. LIANG H T, LIU S, DU J W, et al. Review of deep learning applied to time series prediction[J]. Journal of Frontiers of Computer Science and Technology, 2023, 17(6): 1285- 1300.   
[6] LEE M C, CHANG J W, HUNG J C, et al. Exploring the effectiveness of deep neural networks with technical analysis applied to stock market prediction[J]. Computer Science and Information Systems, 2021, 18(2): 401-418. [7] LU W, LI J, WANG J, et al. A CNN-BiLSTM-AM method for stock price prediction[J]. Neural Computing and Applications, 2021, 33: 4741-4753. [8] VASWANI A, SHAZEER N, PARMAR N, et al. Attention is all you need[C]//Advances in Neural Information Processing Systems 30, Long Beach, Dec 4- 9, 2017: 5998- 6008. [9] ZHOU H, ZHANG S, PENG J, et al. Informer: beyond efficient transformer for long sequence time- series forecasting [C]//Proceedings of the 2021 AAAI Conference on Artificial Intelligence. Menlo Park: AAAI, 2021: 11106-11115.   
[10] WU H, XU J, WANG J, et al. Autoformer: decomposition transformers with auto-correlation for long-term series forecasting[C]//Advances in Neural Information Processing Systems 34, Dec 6-14, 2021: 22419-22430.   
[11] DING Q, WU S, SUN H, et al. Hierarchical multi-scale Gaussian transformer for stock movement prediction[C]//Proceedings of the 29th International Joint Conference on Artificial Intelligence, Yokohama, Jan 7-15, 2021: 4640-4646.   
[12] DAIYA D, LIN C. Stock movement prediction and portfolio management via multimodal learning with transformer [C]//Proceedings of the 2021 IEEE International Conference on Acoustics, Speech and Signal Processing. Piscataway: IEEE, 2021: 3305-3309.   
[13] KIRKPATRICK J, PASCANU R, RABINOWITZ N, et al. Overcoming catastrophic forgetting in neural networks[J]. Proceedings of the National Academy of Sciences, 2017, 114(13): 3521-3526.   
[14] ALJUNDI R, KELCHTERMANS K, TUYTELAARS T. Task- free continual learning[C]//Proceedings of the 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition. Piscataway: IEEE, 2019: 11254-11263.   
[15] BUZZEGA P, BOSCHINI M, PORRELLO A, et al. Dark experience for general continual learning: a strong, simple baseline[C]//Advances in Neural Information Processing Systems 33, Dec 6-12, 2020: 15920-15930.   
[16] MALLYA A, LAZEBNIK S. Packnet: adding multiple tasks to a single network by iterative pruning[C]//Proceedings of the 2018 IEEE Conference on Computer Vision and Pattern Recognition. Washington: IEEE Computer Society, 2018: 7765-7773.   
[17] WANG X, HAN M. Online sequential extreme learning machine with kernels for nonstationary time series prediction [J]. Neurocomputing, 2014, 145: 90-97.   
[18] YU H, DAI Q. DWE- IL: a new incremental learning algorithm for non- stationary time series prediction via dynamically weighting ensemble learning[J]. Applied Intelligence,

[19] WANG H, LI M, YUE X. IncLSTM: incremental ensemble LSTM model towards time series data[J]. Computers & Electrical Engineering, 2021, 92: 107156.

[20] WOO G, LIU C, SAHOO D, et al. CoST: contrastive learning of disentangled seasonal- trend representations for time series forecasting[EB/OL]. [2023-11-12]. https://arxiv.org/abs/ 2202.01575.

[21] HUANG T, CHEN P, ZHANG J, et al. A transferable time series forecasting service using deep transformer model for online systems[C]//Proceedings of the 37th IEEE/ACM International Conference on Automated Software Engineering. Piscataway: IEEE, 2022: 1-12.

[22] IOFFE S, SZEGEDY C. Batch normalization: accelerating deep network training by reducing internal covariate shift [C]//Proceedings of the 32nd International Conference on Machine Learning, Lille, Jul 6-11, 2015: 448-456.

[23] WU Y, HE K. Group normalization[C]//Proceedings of the 15th European Conference on Computer Vision. Cham: Springer, 2018: 3-19.

陈东洋（1997—），男，江西南昌人，硕士研究生，主要研究方向为增量学习、迁移学习，数据挖掘等。

![](images/66f0e24da365754c414d9a8a4db355a262921002888fff8c10141d198957eb4d.jpg)

CHEN Dongyang, born in 1997, M.S. candidate. His research interests include incremental learning, transfer learning, data mining, etc.

![](images/7287cf2fd06d4995645f35dcfeb3cab862938f8806c64a7cfc1dd568319fbecf.jpg)

毛力（1967—），男，江苏无锡人，硕士，教授，硕士生导师，主要研究方向为人工智能、机器学习等。

MAO Li, born in 1967, M.S., professor, M.S. supervisor. His research interests include artificial intelligence, machine learning, etc.

# 2024 CCF全国高性能计算学术年会征文通知

由中国计算机学会主办，中国计算机学会高性能计算专业委员会、华中科技大学共同承办，北京并行科技股份有限公司共同协办的“2024 CCF全国高性能计算学术年会（CCF HPC China 2024）”将于2024年9月24日至26日在武汉·中国光谷科技会展中心召开。全国高性能计算学术年会是中国一年一度高性能计算领域的盛会，为相关领域的学者提供交流合作、发布最前沿科研成果的平台，将有力地推动中国高性能计算的发展。

征文涉及的领域包括但不限于：高性能计算机体系结构、高性能计算机系统软件、高性能计算环境、高性能微处理器、高性能计算机应用、并行算法设计、并行程序开发、大数据并行处理、科学计算可视化、云计算和网格计算相关技术及应用、 $\mathrm { A I ^ { + } }$ Science、量子计算、State of Practice最佳实践，以及其他高性能计算相关领域。本次大会设置“CCF HPC China 2024超算年度最佳应用”Track，评选 CCF HPC China 2024超算年度最佳应用。

会议录用的中文论文将分别推荐到《计算机研究与发展》（EI）、《计算机学报》（EI）、《计算机科学与探索》（正刊）、《计算机工程与科学》（正刊）、《计算机科学》（正刊）、《国防科技大学学报》（EI 正刊）和《数据与计算发展前沿》（正刊）等刊物上发表。会议英文论文直接投稿到 CCF Transactions on High Performance Computing（CCF THPC）期刊，录用通知日期与大会一致，录用论文由作者在 HPC China会议进行论文报告后在 CCF THPC发表。会议还将评选优秀论文和优秀论文提名奖各 5名。

投稿须知：本届大会接收中英文投稿。作者所投稿件必须是原始的、未发表的研究成果、技术综述、工作经验总结或技术进展报告。请登录 https://easychair.org/conferences/?conf=hpcchina2024 的会议投稿系统链接进行投稿，首次登录请注册。

投稿要求：论文模版下载地址为 https://gitee.com/hpcchina/template，中文/英文 word 模版为 word-cn-en.doc，中文/英文 latex 模版为 latex-cn-en.zip。

“超算最佳应用”Track 请单独使用 best-application-latex-cn.zip 中文模版或者 best-application-latex-en.zip 英文模版。

会议将邀请知名院士、学者做大会特邀报告，举行学术报告和分组交流，还将进行高性能计算专题研讨、高性能计算新技术与新产品展示等活动，并同期现场举办“PAC2024全国并行应用挑战赛”决赛。本次会议邀请了国内外知名超算中心主任参加，并举行形式多样、不同主题的论坛研讨。从中您能了解到国内外高性能计算的最新动态，获取对您个人的职业发展有益的各类信息。欢迎从事高性能计算及相关研究的同仁踊跃投稿。

论文提交截止日期：2024年7月31日论文录用通知日期：2024年 8月 31日正式论文提交日期：2024年 9月 05日

联 系 人：袁良、李希代  
联系电话：136-9305-6420  
电子邮箱：hpcchina@gmail.com、lixidai@ict.ac.cn