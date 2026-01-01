# 基于优化LSTM模型的股价预测方法

周章元，何小灵（浙江科技学院理学院，杭州310012)

摘要:为了提高金融时序预测的准确性及泛化性，文章提出了基于主成分分析法和注意力机制来优化长短时记忆模型(PCA-Attention-LSTM)的消费行业板块指数预测方法。首先对指数日常数据生成技术指标，然后通过主成分分析法提取重要特征，根据长短时记忆神经网络(LSTM)学习输入特征的内部变化规律，并利用注意力机制计算LSTM隐层状态的不同权重，最后结合注意力权重和LSTM神经网络进行指数预测。结果表明，优化后的LSTM模型对消费行业板块指数走势具有较强的预测能力。此外，在预测方法的基础上引入了股票的异同移动平均线和均线指标，提供了一种每日轮动自动捕捉交易点的短频量化交易策略。

关键词：长短时记忆网络；技术指标；主成分分析法；注意力机制；行业指数；量化交易中图分类号：F832.51;TP183 文献标识码：A 文章编号:1002-6487(2023)06-0143-06

# 0引言

随着人工智能、大数据等前沿技术的兴起，机器学习方法被广泛应用于股价预测以及交易决策的研究中"，传统的机器学习包括决策树、逻辑回归和支持向量机等，尹湘锋等(2021)2使用不同核函数的支持向量回归对股价进行预测。王燕和郭元凯(2019)P基于改进的XGBoost模型对多只股票进行预测，发现该模型展现出较好的拟合效果。现如今机器学习中的神经网络模型在股票预测方面取得了比传统机器学习模型更优的效果，神经网络模型按层次结构大致可分为两类。第一类为浅层神经网络[4，包括MP神经网络和BP神经网络，在股票预测方面该类模型作为早期的神经网络做出了杰出的贡献，但浅层神经网络结构单一，存在以下三个问题：一是容易过拟合；二是存在局部极值问题；三是网络中的神经元过多容易造成梯度消失或梯度爆炸。第二类为深度神经网络(DNN)[5，相比浅层神经网络，DNN对输入变量的形式没有限制，拟合效果更好，且通过DNN中的tanh激活函数能显著解决梯度爆炸和梯度消失的问题。王超(2021)[使用深度学习模型分析股票技术指标，对中国行业指数进行了实证研究。Singh和Srivastava(2017)使用深度神经网络对谷歌公司股票进行技术分析，比较准确地预测了股票日频价格。贺毅岳等(2020)8使用长短时记忆模型对指数的高精度预测,大幅提升了择时信号的准确度。从现有文献来看，深度学习在金融数据分析中的应用研究方兴未艾。

综上所述，股价预测算法对预测精度有较高的要求。本文基于深度学习LSTM算法对技术分析方法进行改进，在数据处理方面，引入主成分分析法提取出更有效的特征变量，在最大限度地保留原始数据信息的同时，提高学习速率，加入注意力机制提升优化效果。此外，将股票的MACD指标和五日均线考虑在内，通过预测结果构建量化交易策略。

# 1模型构建与交易策略

# 1.1 LSTM模型

LSTM网络是在递归神经网络（RNN)的拓扑结构上进行改进得来的[9。作为RNN的变体，LSTM模型通过引入可控自循环，有效解决了RNN因网络层数增多和时间流逝而带来的梯度消失问题，其巧妙的设计结构，特别适合处理延迟和时序间隔较长的任务。LSTM神经网络结构不同于其他深度学习算法之处在于其特殊的神经元细胞状态，可在长期状态下学习需要记录和遗忘的序列数据信息。细胞单元内部由遗忘门、输入门和输出门三大门限单元构成[°。整体结构如图1所示。

![](images/9e21140b40e70455a97d550b5b7a6c936b4cfd141a0da175075c11dbe20074fa.jpg)  
图1LSTM网络结构单元示意图

(1)遗忘门。LSTM模型开始运行的第一步是确定在细胞状态中需要舍弃哪些干扰信息，可以通过读取序列数据中的 $h ^ { t - 1 }$ (cid:) $x ^ { t }$ ,并给予0到1之间的不同权重 $W _ { f }$ 和偏置$b _ { f }$ 来决定对于序列数据信息的保留程度。遗忘门的控制函数为：

# 财经纵横

$$
f ^ { ( t - 1 ) } = \delta ( { } ^ { t } W _ { f } [ h ^ { ( t - 1 ) } , x ^ { t } ] + b _ { f } )
$$

(2)输入门。输入门控制对新输入序列数据信息加入细胞元状态的处理程度，首先根据Sigmoid函数确定信息的更新程度，如公式(2)所示；其次，决定更新内容的多少，如公式(3)所示；最后，将两部分相结合，丢掉在遗忘门中已舍弃的信息并加入更新后的信息，储存到长期状态中，如公式(4)所示。

$$
\begin{array} { r l } & { i ^ { t } = \sigma ( W _ { i } [ h ^ { ( t - 1 ) } , x ^ { t } ] + b _ { i } ) } \\ & { c ^ { t } = \sigma ( W _ { c } [ h ^ { ( t - 1 ) } , x ^ { t } ] + b _ { c } ) } \\ & { C ^ { t } = i ^ { t } \ast c ^ { t } + f ^ { ( t ) } \ast C ^ { t - 1 } } \end{array}
$$

其中， $W _ { i } \setminus W _ { c }$ 代表相应的权重， $b _ { i } \setminus b _ { c }$ 代表相应的偏置， $C ^ { t }$ 表示当前的单元状态值。

(3)输出门。LSTM结构基于细胞状态决定序列信息的输出程度，细胞状态的输出部分由sigmoid层判断，并将细胞元的长期状态用tanh函数处理后，再与之前的输出部分相乘，最终得到确定输出的序列信息[.12]。用公式表示为：

$$
o ^ { t } = \sigma ( W _ { o } [ h ^ { ( t - 1 ) } , x ^ { t } ] + b _ { o } )
$$

$$
h ^ { t } = o ^ { t * } \operatorname { t a n h } ^ { - 1 } ( c ^ { t } )
$$

其中， $W _ { o }$ 和 $b _ { o }$ 分别代表输出门的权重和偏置， $h _ { t }$ 为当前单元的输出值。

LSTM的创新之处在于，随着细胞元状态 $C _ { ( t - 1 ) }$ 从左至右贯穿LSTM结构，序列数据先经过遗忘门筛选掉部分信息，再通过输入门添加需要新增的数据，而长期状态 $C _ { ( t ) }$ 直接输出。在LSTM结构的每个节点，不断有数据筛选和增加，并且可以通过输出门结构来判断长期状态中的有效信息和无效信息，进行过滤进而形成短期状态 $H _ { \left( t \right) }$

# 1.2主成分分析法(PCA)

主成分分析法在减少需要分析的指标的同时，可将原变量的信息尽可能保留。由于金融变量中各指标之间存在一定的相关关系，因此，可以用较少的综合指标来涵盖原指标中的大量信息。先对指标进行标准化处理，再采集$P$ 维随机向量 $\pmb { x } = ( X _ { 1 } , X _ { 2 } , \cdots , X _ { p } ) ^ { \operatorname { T } }$ ，其中， $X _ { i } = ( X _ { i 1 } , X _ { i 2 } , \cdots$ $X _ { i p } ) ^ { \mathrm { T } }$ ，标准化变换[³]的公式如下：

$$
Z _ { i , j } = \frac { x _ { i j } - \bar { x } _ { j } } { S _ { j } }
$$

$$
\bar { x } _ { j } = \frac { \sum _ { i = 1 } ^ { n } x _ { i j } } { S _ { j } }
$$

$$
S _ { j } ^ { 2 } { = } \frac { { \sum _ { i = 1 } ^ { n } ( x _ { i j } - \bar { x } _ { j } ) } ^ { 2 } } { n - 1 }
$$

接着获得标准化阵 $Z$ 。根据标准化阵 $Z$ 求相关系数矩阵：

$$
r _ { i j } = \frac { \sum { z _ { k j } } ^ { \ast } z _ { k j } } { n - 1 }
$$

$$
\pmb { R } = [ r _ { i j } ] _ { p } x p = \frac { Z ^ { \mathrm { T } } Z } { n - 1 }
$$

其中， $i = 1 , 2 , \cdots , n ; j = 1 , 2 , \cdots , p$

对样本相关矩阵 $\pmb { R }$ 的特征方程 $| { \pmb R } - \lambda I p | = 0$ 进行求解，主成分由 $p$ 个特征值来确定。根据特征值大于0.85固定 $m$ 值，并且要获取超过 $8 5 \%$ 的信息， $b _ { j }$ 是本文获得的一组标准量，主成分通过标准化后的指标转化获得。

$$
U _ { i j } = z _ { i } ^ { \mathrm { T } } b _ { j }
$$

将得到的主成分分别称为第一主成分到第 $p$ 主成分。综合评价是通过对 $m$ 个成分进行混合平均，每个主成分权数为方差贡献率，最终的评价值为主成分的加权求和。

# 1.3注意力机制

注意力机制（Attention)[4能够对原本的输入数据进行调整,根据不同的需求得到更符合当前模型的新的输入数据。Attention机制的核心思想是合理分配模型对于目标信息的注意力，降低或忽略无关信息，并放大所需的重要信息。Attention机制通过调整输入特征的比例，即给予不同的权重参数，突出重要的影响特征向量，抑制无用的特征向量，优化模型学习并作出更优的选择，同时，不会增加模型的计算量。综上所述，本文通过Attention机制来优化模型的性能。Attention机制的结构如图2所示。

![](images/8971aa0d0456b313d32befd90bbbc3de5fdd9524ccca41a37df1f8542c471ba9.jpg)  
图2Attention机制单元结构示意图

在图2中， $\pmb { h } _ { 1 } , \pmb { h } _ { 2 } , \cdots , \pmb { h } _ { t }$ 为输入特征值对应的隐藏层的状态值； $x _ { 1 } , x _ { 2 } , x _ { 3 } , \cdots , x _ { t }$ 为输入的特征值； $\pmb { h } _ { c }$ 为最后节点输出的隐藏层状态值； $a _ { t }$ 为历史输入的隐藏层状态对应的当前输入的权重值。Attention机制的计算公式如下：

$$
e _ { t } = u \operatorname { t a n h } ( w h _ { t } + b )
$$

$$
a _ { t } = { \frac { \exp ( e _ { t } ) } { \sum _ { i = 1 } ^ { n } \exp ( e _ { i } ) } }
$$

$$
s _ { t } = \sum _ { t = 1 } ^ { n } e _ { t } a _ { t }
$$

其中， $w$ 与 $^ b$ 分别为权值参数与偏置， $e _ { t }$ 为 $t$ 时刻输入向量 $\pmb { h } _ { t }$ 所决定的注意力概率分布值， $s _ { t }$ 为最终输出的特征。

1.4 Attention-LSTM神经网络模型

Attention-LSTM神经网络模型[1516]的结构如下页图3所示，具体说明如下：

(1)输入层。将中证消费行业指数数据预处理并进行主成分分析，通过输入层输入至模型中。假定批量输入的数据长度为 $m$ ，可用 $X { = } [ x _ { 1 } . . . x _ { t } . . . x _ { m } ] ^ { \mathrm { T } }$ 表示输入向量。

(2)LSTM层。经过主成分分析提取特征向量输入至LSTM层，用于学习数据特征关系。本文采用一层LSTM结构，学习消费指数的涨跌特性。LSTM层的输出向量为

# 财经纵横

![](images/ead1f2cae3d980ba4cb60075fd480a572f2eab4255730b754976fae1066cbd85.jpg)  
图3Attention-LSTM神经网络模型

$\pmb { H } _ { L }$ 。假定输出向量长度为 $j$ , $\pmb { H } _ { L } { = } [ h _ { L 1 } . . . h _ { L t } . . . h _ { L j } ] ^ { \mathrm { T } } .$

(3) Attention层。Attention层的输入为经过LSTM层计算的隐层状态 $\pmb { H } _ { L }$ ,注意力权值 $a _ { _ t }$ 的计算如式(13)、式(14)所示，记在 $t$ 时刻Attention层的输出为 $S _ { t }$ ，如式(15)所示。

(4)输出层。将Attention层的输出向量输入至输出层，输出层通过全连接层得到消费指数的预测值。假定输出层预测的步长为 $n$ , $\pmb { Y } { = } [ y _ { 1 } . . . y _ { t } . . . y _ { n } ] ^ { \mathrm { T } }$ 。计算公式为：

$$
\scriptstyle { Y = f ( W _ { _ { Y } } \cdot S + b _ { _ { Y } } ) }
$$

其中， $b _ { _ Y }$ 为输出层偏置； $W _ { _ { Y } }$ 为输出层权值矩阵； $f$ 为全连接层激活函数。

# 1.5量化交易策略

在上述预测模型基础上，本文提出了一种交易策略，该策略在得到模型的预测结果后引入均线指标和平滑异同移动平均线(MACD)，通过这两种指标构成量化交易策略"7]。MACD是当前使用最多且有效性最强的指标之一。在该交易策略中，买入信号有两个：一是基于模型的预测股价高于当前五日均线的 $1 \%$ ；二是MACD中的离差值DIF向上突破离差值的短期平均值DEA，且两者均大于0。策略的卖出信号为DIF和DEA都在0轴以下且DIF向下突破DEA。

# 2实证分析

基于PCA-Attention-LSTM模型的消费行业指数预测整体流程如图4所示。

![](images/2e1824c0f9ea6a526ac8c4b1279c0a62988d8efdc33556fc58ddff3a9f4561ce.jpg)  
图4整体流程图

整体流程主要由四个环节构成：(1)根据行业指数的价格序列计算每个交易日的技术指标；(2)由于指标众多，且某些指标之间相关性较高，故对33个指标进行主成分分析以减少数据维度，生成模型的输入因子，降低模型运行时间;(3)将数据分为训练集和测试集，将新因子输入模型进行训练，并进行超参数优化，学习率采用Adam算法进行更新，批次和神经元个数使用网格搜索寻优；(4)将测试集的数据输入模型对指数进行预测。

# 2.1指标选取与数据来源

本文所采用的数据来自Wind数据库，截取指数的时间为2010-01-04至2021-07-15，其中，原始指标包括开盘价、最高价、最低价、收盘价、成交额、涨跌幅、振幅、换手率、总市值、市盈率、市净率、市现率、市销率、股息率和市盈率相对增长比率。为了使数据更加充分，本文还通过技术分析指标计算第三方包Talib生成了18个技术指标，这些技术指标可为预测股价短期波动提供有力支持，例如在股市中使用最多的均线指标。为了充分涵盖短期的价格波动，选择了5日、10日和20日共3个均线指标，还包括变动率指标、抛物线指标、真实波动幅度均值、能量潮等，最终所输入的因子指标详细情况如表1所示。

表1 15个原始指标和 18个技术指标  

<table><tr><td>指标</td><td>符号</td><td>指标</td><td>符号</td><td>指标</td><td>符号</td></tr><tr><td>开盘价</td><td>Open</td><td>市现率</td><td>PCTTM</td><td>真实波动幅度均值</td><td>ATR</td></tr><tr><td>收盘价</td><td>Close</td><td>市销率</td><td>PSTTM</td><td>能量潮</td><td>OBV</td></tr><tr><td>最高价</td><td>High</td><td>股息率</td><td>DYTTMd</td><td>主导周期指标</td><td>HT_DC</td></tr><tr><td>最低价</td><td>Low</td><td>市盈率相对增长比率</td><td>PEG</td><td>平均价格函数</td><td>AVG</td></tr><tr><td>成交额</td><td>Amt</td><td>5日均线</td><td>5MA</td><td>中位数价格</td><td>MED</td></tr><tr><td>涨跌幅</td><td>ct_ch</td><td>10日均线</td><td>10MA</td><td>平均趋向指数</td><td>ADX</td></tr><tr><td>振幅</td><td>Swing</td><td>20日均线</td><td>20MA</td><td>阿隆振荡</td><td>AROON</td></tr><tr><td>换手率</td><td>Turn</td><td>平滑异同移动平均线</td><td>MACD</td><td>顺势指标</td><td>CCI</td></tr><tr><td>总市值</td><td>MV</td><td>12日指数移动平均</td><td>EMA12</td><td>钱德动量摆动</td><td>CMO</td></tr><tr><td>市盈率</td><td>PETTM</td><td>26日指数移动平均</td><td>EMA26</td><td>动向指标</td><td>DX</td></tr><tr><td>市净率</td><td>PBLF</td><td>抛物线指标</td><td>SAR</td><td>变动率指标</td><td>ROC</td></tr></table>

# 2.2数据预处理

# 2.2.1异常值处理

对于消费行业指数的开盘价、收盘价、最高价、最低价和涨跌幅等，从Wind数据库获取指标数据进行描述性统计，发现数据并无缺失值，数据总长度皆为2802。从数据的描述性统计来看，各方面数据呈现的分布皆正常，唯有PEG 指标出现较大的离群值，PEG指标的均值为1.056，而最小值为-5482，远超三个标准差之外，在统计学上可判别其为异常值，故将该点转化为平均值。

# 2.2.2数据归一化

在建模过程中，为消除数据间的量纲影响并解决变量间相关性过高的问题以及提升模型的运行速度，对原始数据进行线性变换，方法选用 $\operatorname* { m i n - m a x }$ 标准化，将33个指标数据值映射到区间[0,1],转换函数如下：

$$
x ^ { * } = \frac { x - m i n } { m a x - m i n }
$$

其中，max为样本数据的最大值，min为样本数据的最小值， $x$ 为需要被标准化处理的原始值。处理后的部分数据如下页表2所示。

# 2.2.3主成分分析结果

为了使数据的质量更高，减小变量间的相关性，本文结合主成分分析法对33个指标提取主成分，得到其每个成分相应的特征值、方差百分比和累计贡献率，根据贡献率从大到小进行排列，结果如下页表3所示。

# 财经纵横

表2 标准化后的数据  

<table><tr><td>时间</td><td>Open</td><td>High</td><td>Low</td><td>Close</td><td>Amt</td><td>…</td></tr><tr><td>2010-04-02 2010-04-06</td><td>0.3436 0.3495</td><td>0.3393 0.3456</td><td>0.3675 0.3661</td><td>0.3549 0.3543</td><td>0.0839 0.0784</td><td>… …</td></tr><tr><td>2010-04-07</td><td>0.3444</td><td>0.3373</td><td>0.3614</td><td>0.3494</td><td>0.0599</td><td>…</td></tr><tr><td>2010-04-08</td><td>0.3387</td><td>0.3312</td><td>0.3525</td><td>0.3386</td><td>0.0734</td><td>…</td></tr><tr><td>2010-04-09</td><td>0.3303</td><td>0.3307</td><td>0.3542</td><td>0.3473</td><td>0.0586</td><td>…</td></tr><tr><td>2010-04-12</td><td>0.3405</td><td>0.3343</td><td>0.3509</td><td>0.3398</td><td>0.0913</td><td>…</td></tr><tr><td>2010-04-13</td><td>0.3307</td><td>0.3365</td><td>0.3492</td><td>0.3506</td><td>0.1021</td><td>…</td></tr><tr><td>2010-04-14</td><td>0.3421</td><td>0.3372</td><td>0.3636</td><td>0.3538</td><td>0.0750</td><td>…</td></tr><tr><td>2010-04-15</td><td>0.3454</td><td>0.3394</td><td>0.3605</td><td>0.3514</td><td>0.0727</td><td>…</td></tr><tr><td>…</td><td>…</td><td>…</td><td>…</td><td>…</td><td>…</td><td>…</td></tr></table>

表3  
特征值及贡献率  

<table><tr><td rowspan=2 colspan=1>编号</td><td rowspan=1 colspan=3>初始特征值</td><td rowspan=1 colspan=3>提取载荷平方和</td></tr><tr><td rowspan=1 colspan=1>特征值</td><td rowspan=1 colspan=1>方差百分比(%)</td><td rowspan=1 colspan=1>累积贡献率(%)</td><td rowspan=1 colspan=1>特征值</td><td rowspan=1 colspan=1>方差百分比(%)</td><td rowspan=1 colspan=1>累积贡献率(%)</td></tr><tr><td rowspan=2 colspan=1>12</td><td rowspan=2 colspan=1>17.8163.835</td><td rowspan=2 colspan=1>53.98811.621</td><td rowspan=1 colspan=1>53.988</td><td rowspan=1 colspan=1>17.816</td><td rowspan=1 colspan=1>53.988</td><td rowspan=1 colspan=1>53.988</td></tr><tr><td rowspan=1 colspan=1>65.609</td><td rowspan=1 colspan=1>3.835</td><td rowspan=1 colspan=1>11.621</td><td rowspan=1 colspan=1>65.609</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>2.634</td><td rowspan=1 colspan=1>7.982</td><td rowspan=1 colspan=1>73.591</td><td rowspan=1 colspan=1>2.634</td><td rowspan=1 colspan=1>7.982</td><td rowspan=1 colspan=1>73.591</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>1.900</td><td rowspan=1 colspan=1>5.758</td><td rowspan=1 colspan=1>79.349</td><td rowspan=1 colspan=1>1.900</td><td rowspan=1 colspan=1>5.758</td><td rowspan=1 colspan=1>79.349</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1.320</td><td rowspan=1 colspan=1>3.999</td><td rowspan=1 colspan=1>83.348</td><td rowspan=1 colspan=1>1.320</td><td rowspan=1 colspan=1>3.999</td><td rowspan=1 colspan=1>83.348</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>0.995</td><td rowspan=1 colspan=1>3.015</td><td rowspan=1 colspan=1>86.364</td><td rowspan=1 colspan=1>0.995</td><td rowspan=1 colspan=1>3.015</td><td rowspan=1 colspan=1>86.364</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>0.921</td><td rowspan=1 colspan=1>2.790</td><td rowspan=1 colspan=1>89.154</td><td rowspan=1 colspan=1>0.921</td><td rowspan=1 colspan=1>2.790</td><td rowspan=1 colspan=1>89.154</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>0.808</td><td rowspan=1 colspan=1>2.450</td><td rowspan=1 colspan=1>91.603</td><td rowspan=1 colspan=1>0.808</td><td rowspan=1 colspan=1>2.450</td><td rowspan=1 colspan=1>91.603</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>0.795</td><td rowspan=1 colspan=1>2.410</td><td rowspan=1 colspan=1>94.013</td><td rowspan=1 colspan=1></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>0.460</td><td rowspan=1 colspan=1>1.393</td><td rowspan=1 colspan=1>95.406</td><td rowspan=1 colspan=1></td></tr></table>

由表3可知，主成分1的特征值最高，为17.816,排名第二的主成分特征值为3.835，排名第三的特征值为2.634，最后一个主成分的特征值为0.460。在统计学上，一般将特征值大于0.8的称为有效主成分，其中有8个主成分的特征值大于0.8，这8个主成分的累计贡献率达到$9 1 . 6 0 3 \%$ ，这些主成分的信息已经涵盖原数据超过 $9 0 \%$ 的信息，因此，本文选取前8个主成分作为训练数据，这8个主成分载荷值的部分数据如表4所示。

表4 主成分载荷值  

<table><tr><td></td><td>主成分1</td><td>主成分2</td><td>…</td><td>主成分7</td><td>主成分8</td></tr><tr><td>Open</td><td>0.993</td><td>−0.058</td><td>…</td><td>-0.020</td><td>0.020</td></tr><tr><td>High</td><td>0.993</td><td>-0.054</td><td>…</td><td>−0.009</td><td>0.022</td></tr><tr><td>Low</td><td>0.992</td><td>-0.054</td><td>…</td><td>−0.011</td><td>0.018</td></tr><tr><td>Close</td><td>0.993</td><td>−0.050</td><td>…</td><td>0.000</td><td>0.021</td></tr><tr><td>Amt</td><td>0.801</td><td>0.241</td><td>…</td><td>0.030</td><td>−0.029</td></tr><tr><td>pct_chg</td><td>0.031</td><td>0.368</td><td>…</td><td>0.828</td><td>0.022</td></tr><tr><td>swing</td><td>0.179</td><td>-0.024</td><td></td><td>0.139</td><td>0.214</td></tr><tr><td>turn</td><td>0.081</td><td>0.571</td><td>…</td><td>0.025</td><td>-0.046</td></tr><tr><td>MV</td><td>0.981</td><td>−0.069</td><td>…</td><td>0.007</td><td>0.008</td></tr><tr><td>PETTM</td><td>0.582</td><td>0.258</td><td>…</td><td>−0.012</td><td>-0.046</td></tr><tr><td>PBLF</td><td>0.854</td><td>0.083</td><td>&quot;</td><td>-0.003</td><td>-0.074</td></tr><tr><td>PCTTM</td><td>0.752</td><td>0.262</td><td>…</td><td>0.011</td><td>-0.220</td></tr><tr><td>PSTTM</td><td>0.961</td><td>0.024</td><td>…</td><td>−0.011</td><td>0.039</td></tr><tr><td>…</td><td>…</td><td>…</td><td>…</td><td>…</td><td>…</td></tr></table>

# 2.3结果分析

本文使用2010-01-04至2021-07-15的消费行业指数进行实证分析。将前文处理后的特征序列按照训练集和测试集进行划分，经过处理后数据集的长度为2744，时间是2010-04-02至2021-07-14，本文把数据集前 $80 \%$ 的数据划分为训练集，后 $20 \%$ 的数据划分为测试集，则训练集数据共有2193条，测试集数据共有548条；以2010-04-02至2019-04-10数据进行训练，用2019-04-11至2021-07-14的数据进行测试。预测模式是：在 $T$ 日取当日和前 $x$ 个交易日的8个主成分构造状态变量，用优化后的LSTM模型进行分析，以预测 $T + 1$ 日的指数价格，依此类推进行滚动，如图5所示。

![](images/7d5f0fdaf2596099cfde82565657ddcd5b6e249b7d9820de7d5c4a1319479446.jpg)  
图5股价预测方法

在对长短时记忆模型进行训练时，模型经历的迭代次数过少则无法获得准确的预测结果，因此，需要将训练的数据集在相同的神经网络中传递若干次。为了找到合适的迭代次数，本文对PCA-Attention-LSTM模型进行了多次训练，不同迭代次数相对应的误差统计如图6所示。

![](images/315e5c19a7195927c49c10f3c45f3ea149a59124ffaa48193c2e76af4f678737.jpg)  
图6不同迭代次数下的损失函数折线图

由图6可知，随着模型迭代次数增加，机器在不断学习的情况下得到了良好的效果，均方根误差在前200个epochs中快速下降，在随后的几百次迭代中也缓慢下降，大约在第400个epoch中逐渐收敛并趋于稳定，结合本文实验的需要，设定500个eopchs作为实验迭代次数。

关于模型其他参数的确定，隐藏节点数和层数直接决定了模型的非线性拟合能力，在其数目足够多的环境下，理论上可逼近任意非线性分布特性的数据集，但同时可能引发过拟合问题;隐藏节点数没有固定的计算方法，一般

# 财经纵横

参照如下经验公式：

hidden_siz $e = { \sqrt { m + n } } + l$

其中， $m$ 为输入变量数， $n$ 为输出变量数，1为可调正常整数，取值范围为[1,10]。

在学习速率方面，采用 $\mathrm { A d a m }$ 优化算法，将学习速率设为0.001，beta1设为 $0 . 9 , { \mathrm { b e t a } } 2$ 设为0.999，epsilon设为 $^ { 1 \ast }$ $1 0 ^ { 8 }$ ，加入dropout层，dropout设为0.05，迭代次数设为500个epochs。在这些参数下，使用网格搜索算法对批量大小batch_size进行寻优，通过算法得出batch_size的最优参数为 $1 4 0$ 。接下来对模型输入时间步长进行选择，因为是短期预测，故本文分别将前一日、前两日至前十日的数据输入模型进行训练来预测下一日的股价，模型输入层窗口大小分别为 $1 ^ { * } 8 , 2 ^ { * } 8 , \cdots , 1 0 ^ { * } 8$ ，输出层维度为1，记录每个模型的平均绝对误差，结果如表5所示。

表5 不同时间窗口下LSTM模型效果  

<table><tr><td rowspan="2">窗口大小</td><td colspan="5">神经元个数</td></tr><tr><td>10</td><td>11</td><td>…</td><td>28</td><td>29</td></tr><tr><td>1*8</td><td>0.040734</td><td>0.042805</td><td>…</td><td>0.037855</td><td>0.030692</td></tr><tr><td>2*8</td><td>0.05733</td><td>0.050006</td><td>…</td><td>0.051107</td><td>0.036410</td></tr><tr><td>3*8</td><td>0.049273</td><td>0.074284</td><td>…</td><td>0.049465</td><td>0.046814</td></tr><tr><td>4*8</td><td>0.043837</td><td>0.089205</td><td>…</td><td>0.056024</td><td>0.053589</td></tr><tr><td>5*8</td><td>0.031644</td><td>0.063473</td><td>…</td><td>0.019870</td><td>0.014418</td></tr><tr><td>6*8</td><td>0.029451</td><td>0.063887</td><td>…</td><td>0.041480</td><td>0.017853</td></tr><tr><td>7*8</td><td>0.036733</td><td>0.067521</td><td>…</td><td>0.018387</td><td>0.018222</td></tr><tr><td>8*8</td><td>0.044406</td><td>0.065288</td><td>…</td><td>0.022447</td><td>0.020813</td></tr><tr><td>9*8</td><td>0.055084</td><td>0.078240</td><td>…</td><td>0.043221</td><td>0.029235</td></tr><tr><td>10*8</td><td>0.021193</td><td>0.015692</td><td>…</td><td>0.020992</td><td>0.030345</td></tr></table>

由表5可知，从时间窗口的角度来看，效果最好的是$5 ^ { * } 8$ 时间窗口，神经元个数为25，平均绝对误差最小，为0.01356，随后分别是使用前10日神经元个数为17、前6日神经元个数为22，MAE分别为0.01427、0.015457。可以看出，股价的前5日数据对短期股价具有较强的敏感性，能够及时地反映股价短期的资金动向，5日数据也正好是非节假日时期股票一周的可交易时间，因此，接下来取前5日的股价数据矩阵来预测后一日收盘价。

为验证本文提出的PCA-Attention-LSTM模型在股价预测精度方面的优势，采用单LSTM模型、PCA-LSTM模型、Attention-LSTM模型、PCA-Attention-LSTM模型这4种模型作为对比模型。其中，LSTM模型采用的是原始数据，加入PCA的是对原始数据生成技术指标并转换为新特征输入模型。经过多次实验后，记录每种模型对测试集进行指数预测的最优结果，不同模型预测结果对比如表6所示。

表6  
不同模型对比结果  

<table><tr><td>模型</td><td>MAE</td><td>RMSE</td><td>MAPE</td></tr><tr><td>LSTM</td><td>0.01750</td><td>0.02408</td><td>3.08136</td></tr><tr><td>PCA-LSTM</td><td>0.01456</td><td>0.01958</td><td>2.62528</td></tr><tr><td>Attention-LSTM</td><td>0.01360</td><td>0.01949</td><td>2.27992</td></tr><tr><td>PCA-Attention-LSTM</td><td>0.01239</td><td>0.01673</td><td>2.15186</td></tr></table>

由表6可知，本文提出的PCA-Attention-LSTM效果最好，评价指标最小，预测性能最优；相对于单LSTM神经网络模型、PCA-LSTM模型和加入注意力机制的Atten-tion-LSTM，MAE分别降低了 $2 9 . 2 \%$ (cid:) $1 4 . 9 \%$ 和 $8 . 9 \%$ , RMSE分别降低了 $4 8 . 5 5 \% , 3 6 . 7 2 \%$ 和 $3 6 . 4 3 \%$ ，MAPE分别降低了$3 0 . 1 7 \%$ (d:) $1 8 . 0 3 \%$ 和 $5 . 6 2 \%$ 。本文提出的PCA-Atten-tion-LSTM模型，在不改变输入样本数据结构的基础上实现降维，减少了输入层的维度，使神经网络的结构更加简单，在提高训练效果的同时，对比其他3个模型大幅度减少了测试误差。将4个模型的预测值和中证消费指数测试集收盘价反归一化后绘图，部分结果如图7所示。

![](images/c47e972acec4c5f1eb874c184c2dc0aa5ee1db6adb5fadcb4e3983fe1727849b.jpg)  
图7PCA-Attention-LSTM模型预测结果

本文对指数价格进行涨跌类型的分类，设置分类标准如下：若下一日指数上涨，标签为1，若下一日指数下跌，标签为0，并将模型预测的指数价格与该日收盘价做对比，进行分类。对分类完成后的真实标签和预测标签构建混淆矩阵，如表7所示。

表7  
混淆矩阵   

<table><tr><td rowspan="2">真实值</td><td colspan="2">预测值</td></tr><tr><td>跌</td><td>涨</td></tr><tr><td>跌</td><td>95</td><td>151</td></tr><tr><td>涨</td><td>113</td><td>201</td></tr></table>

由表7可知，模型的准确率为 $5 2 . 8 6 \%$ ，相比随机猜测涨跌各 $5 0 \%$ ,模型的准确率要高出 $2 . 8 6 \%$ ；召回率为$6 4 . 0 1 \%$ ,表示指数在所有真实上涨的日期中，模型预测结果也为上涨的概率，召回率越高，模型盈利的概率就越大；精确率为 $5 7 . 1 0 \%$ ，表明所有预测上涨的结果中预测成功概率较大，从混淆矩阵角度来看，PCA-Attention-LSTM模型的效果也是较为优异的。

将模型的预测结果作为买卖信号进行交易，通过前文提出的交易策略选择合适的交易点以获得较高的收益回报、夏普比率以及较低的最大回撤率。本文选择基于py-thon的历史回测交易框架backtrader，初始资金设为100万，时间为2019-03-27至2021-07-15，策略标的为中证消费指数，将基于PCA-Attention-LSTM模型的交易策略称为策略1，简单持有中证消费指数策略称为策略2,两种策略的比较结果如表8所示。

表8 两个模型的历史回测  

<table><tr><td></td><td>策略1</td><td>策略2</td></tr><tr><td>夏普比率</td><td>1.534</td><td>0.874</td></tr><tr><td>年化复合收益率(%)</td><td>42.66</td><td>33.72</td></tr><tr><td>最大回撤率(%)</td><td>14.92</td><td>23.79</td></tr><tr><td>总资产（元）</td><td>2202306.22</td><td>1907377.68</td></tr><tr><td>总收益率(%)</td><td>120.23</td><td>90.74</td></tr></table>

# 财经纵横

由表8可知，基于PCA-Attention-LSTM模型的策略1远优于买入并被动持有的策略2，该策略的夏普比率为1.534，而买入并持有策略只有0.874;基于模型的交易策略年化复合收益率高达 $4 2 . 6 6 \%$ ，比简单持有策略高了约9个百分点；最大回撤率为 $1 4 . 9 2 \%$ ,将最大回撤率降低了约9个百分点；总资产翻倍达到220多万，总收益为 $1 2 0 . 2 3 \%$ ,高于简单持有策略近30个百分点。由此可见，相比单纯持有指数，基于优化后的LSTM模型进行指数买卖有更优异的结果，这说明模型不仅在拟合效果上表现良好，而且在金融交易方面也具有一定的实用价值。

# 3结束语

本文针对金融时间序列预测问题，提出一种基于PCA-Attention-LSTM的指数预测模型。首先，该模型消除了输入特征的相关性，减少了模型神经网络的输入层数，在提升输入数据精简度的同时，也简化了整体网络结构。其次，通过引入注意力机制使模型能够更多地关注相关性更高的特征，增加了模型的预测精度。进一步对最终的预测结果进行涨跌类型分类，计算混淆矩阵，模型的准确率、精确率和召回率都大于 $5 0 \%$ ,其中，召回率为 $6 4 \%$ ,远高于随机猜测。最后，将预测结果进行历史回测，结果表明相比简单持有中证消费指数，基于PCA-Attention-LSTM模型的交易策略能取得更高的策略收益和更小的最大回撤。这说明本文提出的模型对行业指数涨跌类型具有一定的预测能力，可作为预测指数走势或实盘交易的参考。

# 参考文献：

[1]Zhang X, Zhang Y J, Wang S Z, et al. Improving Stock Market Prediction via Heterogeneous Information Fusion [J].Knowledge-based Systems,2018,(143).   
[2]尹湘锋,崔浩锋,文雪婷.基于两类核函数的TSVR在股价预测中的 比较[J].统计与决策,2021,(12).   
[3]王燕,郭元凯.改进的XGBoost模型在股票预测中的应用[J].计算机 上柱与应用,2019,23(20).   
[4]Asteris P G, Mokos V G. Concrete Compressive Strength Using Artificial Neural Networks [J].Neural Computing and Applications,2020,32 (15).   
[5]李仪,林建君，朱习军.基于改进DNN的糖尿病预测模型设计[J].计 算机工程与设计,2021,42(5).   
[6]王超.深度学习在行业指数技术分析中的应用研究[J].管理评论， 2021,33(3).   
[7]Singh R, Srivastava S. Stock Prediction Using Deep Learning [J].Multimedia Tools and Applications,2017,76(18).   
[8]贺毅岳,高妮,韩进博,等.基于长短记忆网络的指数量化择时研 究[J].统计与决策,2020,(23).   
[9]Bukhari A H, Raja M A Z, Sulaiman M, et al. Fractional Neuro-sequential ARFIMA-LSTM for Financial Market Forecasting [J].IEEE Access,2020,(8).   
[10]汪定,邹云开，陶义，等.基于循环神经网络和生成式对抗网络的口 令猜测模型研究[J].计算机学报,2021,44(8).   
[11]刘亮,蒲浩洋.基于LSTM的多维度特征手势实时识别[J].计算机 科学,2021,48(8).   
[12]Yadav A, Jha C K, Sharan A. Optimizing LSTM for Time Series Prediction in Indian Stock Market [J].Procedia Computer Science,2020, (167).   
[13]宋丽娜.基于情感分析和PCA-LSTM模型的股票价格预测[J].中 国管理信息化,2021,24(21).   
[14]蒙懿，徐庆娟.基于CNN-BiLSTM和注意力机制的股票预测[J].南 宁师范大学学报(自然科学版),2021,38(4).   
[15]Lin Y, Yan Y, Xu J, et al. Forecasting Stock Index Price Using the CEEMDAN-LSTM Model [J].The North American Journal of Economics and Finance,2021,(57).   
[16]Jin Z, Yang Y, Liu Y. Stock Closing Price Prediction Based on Sentiment Analysis and LSTM [J].Neural Computing and Applications, 2020,32(13).   
[17]Nahil A, Lyhyaoui A. Short-term Stock Price Forecasting Using Kernel Principal Component Analysis and Support Vector Machines: The Case of Casablanca Stock Exchange [J].Procedia Computer Science,2018,(127).

(责任编辑/易永生)