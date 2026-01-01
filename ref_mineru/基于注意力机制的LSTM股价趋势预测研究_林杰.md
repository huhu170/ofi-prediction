# 基于注意力机制的LSTM 股价趋势预测研究

林傑康慧琳

（同济大学经济与管理学院，上海200092)

摘 要：针对中国股票市场，提出了一种基于注意力机制的 LSTM 股价趋势预测模型。选取 42只中国上证 50 从 2009 年到 2017 年的股票数据为实验对象，根据股票市场普遍认可的经验规则，分别对每个技术指标进行量化处理得到股票涨跌的趋势数据，并和交易数据混合作为预测模型的输入，然后使用基于注意力机制的 LSTM 模型提取股价趋势特征进行预测。实验结果表明：引入股票离散型趋势数据到预测模型中，能够在已有交易数据和技术指标的基础上提升预测精确度，与传统的机器学习模型 SVM 和单一的 LSTM 模型相比，基于注意力机制的 LSTM 模型具有更好的预测能力。

关键词：股价趋势预测；LSTM;注意力机制中图分类号：TP183/F830.9 文献标志码：A

# Attention-Based LSTM for Stock Price Movements Prediction

LIN Jie KANG Huilin (School of Economics &. Management, Tongji University, Shanghai 200092, China)

Abstract: This paper adresses problem of stock price movements prediction for china stock markets. We present an Attention-Based LSTM approach to predict stock price movements. Nine years of historical data from 2009 to 2017 of 42 stocks of SSE 50 are selected for experimental evaluation. According to the empirical rules generall accepted by the stock market, the stock technical indicators are quantified to obtain the stock price movements prediction and together with the trading data as input to the prediction model. Then, we use Attention-Based LSTM to extract important features for prediction. The experimental re sults suggest that introducing stock discrete trend data into the prediction model can achieve higher predic tion accuracy based on trading data and technical indicators. Experimental results also show that the Attention-Based LSTM model outperforms both traditional machine learning model SVM and the single LSTM model on overall performance.

Key words: stock price movements prediction; LSTM; attention mechanism

股票市场被称为“经济晴雨表”，它的波动与市场经济的兴衰息息相关。股价趋势预测是一个经典问题，一直受到学术界和工业界的广泛关注。Fama(1970)的有效市场假说（eficientmarkethypothesis)指出，股票市场是“有效信息”市场。也就是说，如果能把股票交易数据进行有效处理并使用合适的算法，就可以预测股票的涨跌趋势，这一理论成为后续股票预测研究工作的基础和依据。

目前越来越多的学者开始投入股票预测研究中，试图用各种方法和模型来解决金融时间序列数据非线性、非平稳、高噪声，以及股票市场规则高复杂性的问题。近年来，随着大数据与云计算的兴起，深度学习算法取得了巨大突破。深度神经网络是一个高度复杂的非线性人工智能系统，适合处理多影响因素、不稳定的非线性难题，它能够从数据中提取抽象特征，并能在不依赖计量经济学假设和金融专业知识的情况下识别隐藏的非线性关系，这使得深度学习可能成为解决股价趋势预测问题的突破口。

早在20世纪90年代，应用神经网络来预测股价趋势便受到国外学术界的广泛关注。White（1988）首次将BP神经网络模型应用于股票市场时间序列的处理和预测中，使用IBM公司股票日收益率为实证研究的对象。Kolarik和Rudorfer（1994）将ANN模型和ARIMA模型比较，发现ANN模型预测效果更优。近年来，国内外学者对神经网络预测股票市场问题的研究也取得了进展。Bildirici等（2009)学者将ARCH/GARCH模型和ANN模型相结合，选取伊斯坦布尔股票市场三十年的股票历史数据作为研究对象，发现GARCH和ANN的混合模型比APGARCH模型具有更好的预测效果，但处理海量数据比较棘手。Hammad等（2007）利用多层BP神经网络分析了约旦证券市场的股票价格，通过实证分析发现相比较于统计学和计量经济学分析方法，BP神经网络预测效果更优，但是文章中并未提及传统BP神经网络容易陷入局部极小值的问题。

虽然人工神经网络相较于其他模型有较好的预测能力，但经过多年的模型优化，人工神经网络及其各种改进的模型仍不能完全摆脱易陷入局部最小值的缺陷，研究人员开始寻找新的解决思路。近年来，深度学习算法在语音识别、自然语言处理、计算机视觉等诸多领域均取得了巨大成功。考虑到时序关系在金融领域是普遍存在的，加上深度学习中LSTM对时间序列有着良好的处理能力，研究人员开始将LSTM模型应用到金融时间序列问题的研究中。例如，Chen等（2015）使用LSTM预测中国股票回报率，根据投资回报率将标签分为7类，利用LSTM模型对股票回报率进行预测。与随机预测方法相比，LSTM模型提高了股票回报率预测的准确率。Fischer（2018)使用LSTM预测了金融市场的表现，形成了短期的投资策略，包括直接预测股价和投资组合等。

目前将LSTM模型应用于金融时间序列的研究偏少，同时LSTM模型用于金融时间序列预测的结果并不令人满意。注意力机制（attention mechanism)最早由Treisman等（1980)提出，通过计算注意力概率分布从众多信息中选出关键信息，对关键性输入进行突出，实现对传统模型的优化。已有实验表明，融入注意力机制的LSTM比单一的LSTM在机器翻译、情感分类、关系分类等问题中有更好的表现。例如，Wang等（2016）提出的AT-LSTM模型，在LSTM的隐藏层中加入了注意力机制，提高了文本情感分类的精确度。

因此，本文提出了一种基于注意力机制的LSTM股价趋势预测模型，用来提高股价趋势预测的精度。首先，介绍了实验使用的数据集，着重描述了连续型技术指标和离散型技术指标。然后，详细描述了本文提出的基于注意力机制的LSTM预测模型架构。最后进行两个对比实验，验证了引入股票离散型趋势数据和本文提出的基于注意力机制的LSTM股价趋势预测模型的有效性。

# 1数据集描述

从大智慧客户端获取了42只中国上证50从2009年1月至2017年12月的股票交易数据，每条交易数据都有开盘价、收盘价、最高价、最低价、交易量等信息，共计83716条交易数据，并根据交易数据计算得到股票的技术指标。选取每支股票2016年和2017年的数据作为测试集，其余部分用来模型训练。42只股票数据共同构成实验的数据集，共同反映市场情况。

# 1.1数据预处理

# 1.1.1数据标准化

股票交易数据各变量数值处于不同的量级，如果不进行标准化处理，大量级变量对结果的影响会覆盖小量级变量对结果的影响，从而丢失小量级变量所包含的信息。需注意的是，由于开盘价、收盘价、最高价和最低价4者间具有相关性，因此对这4个变量统一进行标准化处理。本文采用min-max标准化法，如下所示：

$$
x ^ { \prime } { _ { i } } = \frac { x _ { i } - x _ { \operatorname* { m i n } } } { x _ { \operatorname* { m a x } } - x _ { \operatorname* { m i n } } }
$$

其中： $\mathbf { X } _ { \mathrm { i } }$ 为待标准化处理的原始数据； $\mathbf { X } _ { m a x }$ 和$\mathbf { X } _ { m i n }$ 分别代表待标准化处理的原始数据序列中的最大值和最小值； $\mathbf { x } _ { \mathrm { ~ i ~ } } ^ { \prime }$ 为经过min-max标准化之后得

到的目标数据，它的取值范围是[0，1]。

# 1.1.2 涨跌标签label

$$
\mathrm { l a b e l } = \left\{ \begin{array} { l l } { 1 , \mathrm { O p n p r c _ { t + 1 } } \geqslant \mathrm { O p n p r c _ { t } } } \\ { 0 , e l s e } \end{array} \right.
$$

其中，Opnprct为第t天的开盘价，如果第 $\mathrm { t } + 1$ 天的开盘价 $\mathrm { O p n p r c } _ { \mathrm { t } + 1 }$ 大于等于第t天的开盘价Opnprct，那么第t天的标签为“上涨”，标记为“1”，反之为“下跌”，标记为“0”。

# 1.2技术指标介绍

1.2.1连续型技术指标—实际时序序列

使用以下7个技术指标用于股价趋势预测：MA、K、D、WR、MACD、RSI和 $C C I$ 。根据表1中的公式，通过股票交易数据，可以计算得到这7个技术指标值。通过观察表1可知，技术指标是连续型数值，它们的值在[-1，1]，不会出现大量级覆盖小量级信息的情况。

表1技术指标及计算公式  

<table><tr><td>技术指标</td><td>计算公式</td></tr><tr><td>MA(取10)</td><td>(Ct + Ct−1 +  +Ct−9)/10</td></tr><tr><td></td><td>Stochastic K % (Ct − LLt−(n−1) )/(HHt−(n−1) −LLt−(n−1) ) × 100</td></tr><tr><td>Stochastic D%</td><td>(( ∑ Kt−i)/10)% i=0</td></tr><tr><td>Williams R%</td><td>(HHt−n − Ct )/(HHt−n − LLt−n ) X 100</td></tr><tr><td>MACD</td><td>2 MACD(n)t−1 + (DIFFt −MACD(n)t−1) n+1</td></tr><tr><td>RSI</td><td>n−1 i=0 i=0</td></tr><tr><td>CCI</td><td>(Mt −SMt)/0.015Dt</td></tr></table>

$C _ { t } \setminus L _ { t } , . \ H _ { t }$ 分别代表第 $t$ 天的收盘价、最低价和最高价。其中，$D I F F _ { t } { = } E M A ( 1 2 ) _ { t } { - } E M A ( 2 6 ) _ { t }$ 。EMA是指数移动平均值，EMA$\mathbf { \Phi } ( k ) _ { t } = E M A ( k ) _ { t - 1 } + \propto ( C _ { t } - E M A ( k ) _ { t - 1 } )$ , $\infty$ 是平滑因子， $\infty = 2 / ( k$ $+ 1 )$ 。 $L L _ { t }$ 和 $H H _ { t }$ 分别表示过去 $t$ 天中的最低值和最高值。其中，$\begin{array} { r } { M _ { t } = ( H _ { t } + L _ { t } + C _ { t } ) / 3 , S M _ { t } = ( \sum _ { i = 1 } ^ { n } M _ { t - i + 1 } ) / n , D _ { t } = ( \sum _ { i = 1 } ^ { n } | M _ { t - i + 1 } - 1 ) / n . } \end{array}$ $- S M _ { t } \mid ) / n$ $U P _ { t }$ 和 $D W _ { t }$ 分别代表 $t$ 时间段股票上涨和下跌

# 1.2.2离散型技术指标——股价趋势预测

根据现有的专家知识经验，对1.2.1中的股票技术指标(连续型)进行量化处理得到各指标对股票涨跌趋势的预测，它反映的是各指标对未来股价“上涨”和“下跌”的判断。在这里，把它称为离散型技术指标。其中，用“ $+ 1$ ”代表“上涨”，“-1”代表“下跌”。下文将详细介绍每个离散型技术指标涨跌判断的专家规则。

(1)移动平均线(MA)

移动平均线又称均线，将一定时期内的股票(指数)收盘价加以平均得到。将某一股票不同时间的平均值连接起来，形成一根MA，用以观察股票变动趋势。有专家经验知识指出：若当前价高于其移动平均线，则产生购买信号，预示未来行情看涨；若当前价低于其移动平均线，则产生出售信号，预示未来行情看跌。因此，量化规则如下：如果当前收盘价Clsprc高于MA，行情涨跌趋势为“涨”，标记为“ $^ +$ $1 ^ { \mathfrak { s } }$ ；如果当前收盘价Clsprc低于 $M A$ ，行情涨跌趋势为“跌”，标记为“-1”。在股票市场，常用线有5天、10天、30天、60天、120天和240天的指标。本文中使用10日均线，即MA2。

(2)随机指标 $( K , D )$

$K$ 和 $D$ 通过最近几个交易日的最高价、最低价和收盘价的波动，来估计未来的涨跌趋势，精准反映股票在一段时间内的随机振幅，是进行中短期趋势波段分析研判的较佳的技术指标。有专家经验知识指出：一般当月 $K , D$ 值在低位时逐步进场吸纳，即产生购买信号。因此，量化规则如下：如果当前 $K$ 值大于前一天，行情涨跌趋势为“涨”，标记为“ $+ 1 ^ { \prime \prime }$ ;如果当前 $K$ 值小于前一天，行情涨跌趋势为“跌”，标记为“-1”。 $D$ 也遵从此规定。

(3)威廉指标(WR)

利用“最后一周期”的最高价、最低价、收市价，计算当日收盘价的摆动点，度量市场处于超买还是超卖状态。有专家经验知识指出： $W R$ 属于摆动类反向指标，即股价上涨，WR指标向下，股价下跌，WR指标向上。因此，量化规则如下：如果当前WR值小于前一天，行情涨跌趋势为“涨”，标记为“ $+ 1 ^ { \prime \prime }$ ;如果当前WR值大于前一天，行情涨跌趋势为“跌”，标记为“-1”。WR1是10天买卖强弱指标；WR2是6天买卖强弱指标。本文中使用WR1。

(4)指数平滑移动平均线(MACD)

MACD由快的指数移动平均线（EMA12）减去慢的指数移动平均线（EMA26)得到。有专家经验知识指出：MACD从负数转向正数，是购买的信号；MACD从正数转向负数，是售出的信号。因此，量化规则如下：如果当前MACD值大于前一天，行情涨跌趋势为“涨”，标记为“ $+ 1 ^ { \mathfrak { n } }$ ；如果当前MACD值小于前一天，行情涨跌趋势为“跌”，标记为“-1”。

(5)相对强弱指数(RSI)

相对强弱指数RSI以数字计算的方式来量化分析市场买卖意向和实力。相对强弱指数RSI认为，在一个正常的股市中，多空买卖双方的力量必须得到均衡，股价才能稳定。有专家经验知识指出：RSI指标的取值范围在 $0 \sim 1 0 0 , 7 0$ 以上可认为是超买，30以下可认为是超卖。在正常区间(30，70)内，若向上的力量较大，则 $R S I$ 曲线上升，是购买的信号，若向下的力量较大，则 $R S I$ 曲线下降，是售出的信号。因此，量化规则如下：若当前 $R S I$ 值小于30，行情涨跌趋势为“涨”，标记为“ $+ 1$ ”;若当前RSI值大于70，行情涨跌趋势为“跌”，标记为“-1”；如果RSI值在 $3 0 \sim 7 0$ ,当前 $R S I$ 值大于前一天，行情涨跌趋势为“涨”，标记为“ $+ 1 ^ { \prime \prime }$ ;若当前RSI值小于前一天，行情涨跌趋势为“跌”，标记为“-1”。

![](images/c80883850410bde1467cfe45c5df0b96c907fa46cd3f617bddd80097151ff761.jpg)  
图1 Attention-Based LSTM神经网络架构

(6)随顺市势指标(CCI)

CCI是专门测量股价是否已超出常态分布范围，波动于正无穷大和负无穷大之间的指标。有专家经验知识指出：在100以上可认为是超买，-100以下可认为是超卖。在振荡区 $( - 1 0 0 , + 1 0 0 )$ ，若向上的力量较大，是购买的信号，若向下的力量较大，则是售出的信号。因此，量化规则如下：若当前CCI值小于-100，行情涨跌趋势为“涨”，标记为“ $+ 1$ ”;若当前CCI值大于100，行情涨跌趋势为“跌”，标记为“-1”;如果CCI值在 $- 1 0 0 \sim + 1 0 0$ ，当前CCI值大于前一天，行情涨跌趋势为“涨”，标记为“ $+ 1 ^ { \prime \prime }$ ,若当前CCI值小于前一天，行情涨跌趋势为“跌”，标记为“-1”。

# 2预测模型

本文所提出的基于注意力机制的LSTM的股价趋势预测模型架构如图1所示。它包括数据准备层、输入层、长短时记忆网络层、注意力机制层、全连接层和输出层。数据准备层完成股票混合数据集的构建；长短时记忆网络层充分发挥LSTM的优势，保持股票数据信息并提取其特征；注意力机制层用来识别股票涨跌最主要的特征；最后经过全连接层，使用Softmax分类器得出分类结果。

# 2.1数据准备层

预测模型的输入包括交易数据和技术指标两部分，技术指标又分为连续型(详见1.2.1)和离散型（详见1.2.2)，预测模型的输出是“涨跌”预测标签。将前 $t$ 天的股票数据依次输入Attention-BasedLSTM 网络中，预测第 $t + 1$ 天的涨跌情况。将模型预测得到的结果与真实的涨跌情况进行对比，来调整预测模型。

# 2.2长短时记忆网络层

用数据准备层的输出数据作为长短时记忆网络层的输入，充分发挥LSTM的时间序列处理优势，保持股票信息并提取其特征。长短时记忆网络(long short term memory，LSTM）是一种改进的RNN模型，它解决了RNN训练过程中梯度爆炸或者梯度消失等问题，所有的RNN都有一种重复的神经网络模块链式的形式。在标准的RNN中，这个重复的模块只有一个非常简单的结构，例如一个tanh层或者sigmoid层。与单一tanh循环体结构不同，LSTM是一种拥有三个“门”的特殊网络结构，它们分别是忘记门（forgetgate)、输入门（inputgate)和输出门（outputgate)。忘记门负责选择忘记过去无用信息;输入门负责确定有用的新信息被存放在细胞状态中；输出门决定输出信息。LSTM单元结构如图2所示。

记忆模块进行状态更新和信息输出的过程如下：

（1）LSTM的核心是Cell：“CellState”即细胞状态，是随着时间而变化的整个模型的记忆传输带。传送带本身是无法控制哪些信息是否被记忆，起控制作用的忘记门、输入门和输出门。

（2）遗忘状态信息，负责选择忘记过去的无用信息。读取当前时刻的输入 $\boldsymbol { \mathcal { X } } _ { t }$ 和上一个时刻的记忆单元状态信息 $h _ { t - 1 }$ ，然后通过sigmoid函数来输出一个[0，1]的值，用来说明历史信息需要被保留的程度。

$$
f _ { t } = \sigma ( W _ { f } \bullet [ h _ { t - 1 } , x _ { t } ] + b _ { f } )
$$

![](images/f01e7952afe5528f30a70e23677fa132936bb2af67c8a3e6f45ce19c27f46f10.jpg)  
图2LSTM单元结构

（3）更新状态信息，将有用的新信息存放在细胞状态中。首先，计算输入门的值 $i _ { t }$ 。输入门的作用是控制当前时刻的数据输入如何影响记忆单元状态值。然后，再计算当前时刻 $t$ 的候选记忆单元信息 $\widetilde { C } _ { t }$ ，里面包含新的待添加信息。最后，将旧细胞状态 $C _ { t - 1 } * f _ { t }$ （用于遗忘）与新的候选信息 $i _ { t } * \widetilde { C } _ { t }$ 进行合并，确定更新的信息：

$$
i _ { t } = \sigma ( W _ { i } \bullet [ h _ { t - 1 } , x _ { t } ] + b _ { i } )
$$

$$
\widetilde { C } _ { t } = \operatorname { t a n h } \ ( W _ { c } \bullet [ h _ { t - 1 } , x _ { t } ] + b _ { C } )
$$

$$
C _ { t } = f _ { t } * C _ { t - 1 } + i _ { t } * \widetilde { C } _ { t }
$$

（4）输出信息。先确定状态的哪个部分将被输出，最后通过输出门的值与记忆单元状态信息经过tanh变换之后得到当前时刻的记忆单元输出信息：

$$
o _ { t } { = } \sigma ( W _ { o } \cdot [ h _ { t - 1 } , x _ { t } ] { + } b _ { o } )
$$

$$
h _ { t } = o _ { t } \ast t a n h \mathrm { ~ } ( C _ { t } )
$$

# 2.3注意力机制层

本文在分类方法中加入了注意力机制层，它能够更好地捕捉股票数据中的有效信息，抓住核心关键数据信息，克服了标准LSTM模型在每步预测时由于使用了相同的状态向量导致在预测时无法充分学习序列编码的细节信息的问题。 $H \in R ^ { d \times N }$ 表示由LSTM网络输出向量 $[ h _ { 1 } , \cdots , h _ { N } ]$ 所组成的矩阵，$N$ 表示句子长度，本文中指代预测周期 $T$ ，用 $T$ 的数据预测第 $T + 1$ 天的涨跌趋势，注意力机制最终会产生一个注意力权重向量 $\alpha$ 和 $r$ :

$$
\begin{array} { l } { M { = } \operatorname { t a n h } \ ( W _ { h } \ H ) } \\ { \alpha { = } \operatorname { s o f t m a x } ( \boldsymbol { w } ^ { \mathrm { { T } } } \ M ) } \\ { r { = } H _ { \alpha } { } ^ { \mathrm { { T } } } } \end{array}
$$

其中， $W \in R ^ { d \times N } , \alpha \in R ^ { N } , r \in R ^ { d }$ 。 $W _ { h } \in R ^ { d \times d }$ 和$\tau ^ { \mathrm { T } } \in R ^ { d }$ 是后续模型需要训练的参数矩阵。为了强调LSTM的隐藏层数据，最终注意机制输出的LSTM的隐藏层数据 $H$ 和 $r$ 的拼接。最终的输出向量表示如下：

$$
h ^ { \star } = \operatorname { t a n h } \ ( W _ { \mathit { p } } \ r + W _ { \mathit { x } } \ h _ { \mathit { N } } )
$$

其中： $h ^ { \ast } \in R ^ { d } : W _ { \rho }$ 和 $W _ { x }$ 是后续模型需要训练的参数矩阵；输出变量 $h ^ { * }$ 最后经过一个全连接层和Softmax分类器实现股票涨跌的预测。

# 3实验结果与分析

为了检验本文提出的离散型技术性指标的效果，以及基于注意力机制的LSTM预测模型的作用，设置了以下两组实验。实验一，为检验本文提出的离散型技术性指标的效果，使用三种不同的输入数据，分别使用基于注意力机制的LSTM模型进行股价趋势预测实验；实验二，为了检验本文提出的基于注意力机制的LSTM模型的效果，使用基本交易数据(7维) $^ +$ 离散型技术指标（7维)为输入数据，分别使用Attention-BasedLSTM模型、SVM模型和LSTM模型进行股价趋势预测实验。

本文的实验环境为Ubuntu16.04操作系统，采用Tensorflow开源平台作为深度学习平台，采用Python3.5编写实验程序，利用Python调用Ten-sorflow框架来实现基于注意力机制的LSTM模型的搭建。在SVM分类对比实验中，采用Python的第三方机器学习库Scikit-learn来实现传统机器学习的分类方法。

# 3.1模型参数

在传统机器学习SVM分类方法中，设置Ker-nel为Polynomia，目标函数的惩罚系数C为1，de-gree设置为1。在LSTM模型和Attention-BasedLSTM模型中，设置droupout rate为0.5。在训练的时候采用小批量随机梯度下降法，以减少训练损失，mini-batch 设置为64。

# 3.2评价指标

实验采用的评估分类模型的指标为模型分类的准确率（Accuracy)和F值(F-measuree）。计算这两个评估统计指标需要先计算精确率(Precision)和召回率(Recall)。精确率和召回率公式如下：

$$
P r e c i s i o n _ { p o s i t i v e } = \frac { T P } { T P + F P }
$$

$$
P r e c i s i o n _ { n e g a t i v e } = \frac { T N } { T N + F N }
$$

$$
R e c a l l _ { t o s i t i v e } = \frac { T P } { T P + F N }
$$

$$
R e c a l l _ { n e g a t i v e } = \frac { T N } { T N + F P }
$$

其中： $T P$ 表示预测上涨趋势，实际也为上涨的样本个数； $F P$ 表示预测下跌趋势，实际也为上涨的样本个数； $T N$ 表示预测下跌趋势，实际也为下跌的样本个数； $F N$ 表示预测下跌趋势，实际也为上涨的

样本个数。

Accuracy和 $\mathrm { F } \cdot$ measure公式如下:

$$
A c c u r a c y = \frac { T P + T N } { T P + F N + T N + F P }
$$

$$
F - m e a s u r e { = } \frac { 2 \times P r e c i s i o n \times R e c a l l } { P r e c i s i o n + R e c a l l }
$$

其中：Precision 是 $P r e c i s i o n _ { t o s i t i t e }$ 和 Precisionegative的加权平均；Recall是 $R e c a l l _ { \mathit { \phi o s i t i t e } }$ 和 $R e c a l l _ { \mathit { n e g a t i v e } }$ 的加权平均。

# 3.3对比实验分析

# 3.3.1不同输入数据集的预测对比实验

为检验本文提出的离散型技术性指标的效果，实验分别采用以下三种输入数据集：a．基本交易数据（7维）；b.基本交易数据（7维） $^ +$ 连续型技术指标(7维）;C.基本交易数据（7维) $^ +$ 离散型技术指标（7维）。本实验中，用前20天的股票数据去预测第21天的股票涨跌，使用基于注意力机制的LSTM网络作为预测模型。实验结果如表2所示。

使用BASIC $^ +$ 离散型指标数据的数据集在基于注意力机制的LSTM模型中是最优的，将近 $56 \%$ 的正确率。相较于只使用交易数据，正确率提升将近 $2 . 5 \%$ ，相较于BASIC $^ +$ 连续型输入，正确率有$1 \%$ 的提高，实验结果验证了加入离散型技术指标的有效性。离散型技术指标数据作为模型的输入时，它向模型输入了各个技术指标所感知的趋势信息，相较于输入股票实际的技术指标，这是向前迈进的一步。实验小结：引入股票离散型趋势数据到预测模型中，能够在已有交易数据和技术指标的基础上提升预测精确度。

# 3.3.2 Attention-Based LSTM模型与 SVM、 LSTM模型的对比实验

为了检验本文提出的含有注意力机制的LSTM模型的效果，分别使用Attention-BasedLSTM模型、SVM模型和LSTM模型进行股价涨跌预测实验。本实验中，用前20天的股票数据去预测第21天的股票涨跌，使用基本交易数据(7维) $^ +$ 离散型技术指标(7维)为输入数据。实验结果如表3所示。

表3列出了含有离散型技术指标混合数据集分别在SVM、LSTM和LSTM $^ +$ Attention模型的分类正确率和 $F$ 值。从实验结果可以看出，SVM在本次试验中表现最差，使用LSTM模型进行股票预测，相较于SVM模型，预测的正确率提高了 $3 \%$ 。实验结果表明了LSTM模型在股票预测中的有效性。Attention-Based LSTM模型表现最为出色，正确率提升将近 $5 \%$ 。实验结果表明，与传统的机器学习模型SVM和单一的LSTM模型相比，基于注意力机制的LSTM模型具有更好的预测能力。

表2不同输入数据集的预测对比实验结果  

<table><tr><td rowspan="3">股票涨跌预测 分类模型</td><td colspan="2">BASIC</td><td>BASIC+连续型</td><td></td><td>BASIC+离散型</td><td></td></tr><tr><td>准确率</td><td>F值</td><td>准确率</td><td>F值</td><td>准确率</td><td>F值</td></tr><tr><td>LSTM+ Attention</td><td>0.523</td><td>0.527</td><td>0.536</td><td>0.534</td><td>0.548</td><td>0.542</td></tr></table>

表3Attention-Based LSTM模型与SVM、LSTM模型  
的对比实验结果  

<table><tr><td>股票涨跌预测分类模型</td><td>SVM</td><td>LSTM</td><td>LSTM+ Attention</td></tr><tr><td>准确率</td><td>0.503</td><td>0.529</td><td>0.548</td></tr><tr><td>F值</td><td>0.504</td><td>0.531</td><td>0.542</td></tr></table>

# 4总结与展望

针对金融时间序列动态不稳定性以及长期依赖的特性，本文提出了一种基于注意力机制的LSTM股价趋势预测模型。实验结果表明引入股票离散型趋势数据到预测模型中，能够在已有交易数据和技术指标的基础上提升预测精确度，与传统的机器学习模型SVM和单一的LSTM模型相比，基于注意力机制的LSTM模型具有更好的预测能力。未来可以进一步从丰富专家经验规律特征，引入宏观经济指标，加入金融新闻与评论、网民情绪语料等方面来优化本模型。

# 参考文献：

[ 1 ] MALKIEL B G, FAMA E F. Efficient capital markets: a review of theory and empirical work[J]. The journal of Finance, 1970, 25(2): 383-417.   
2] WHITE H. Economic prediction using neural networks : the case of IBM daily stock returns[C]//Neural Networks, 1988,IEEE International Conference on. IEEE,1988:451-458.   
[ 3 ] KOLARIK T, RUDORFER G. Time series forecasting using neural networks[J]. Time Series and Neural Networks, 1994, 12(3):86–94.   
4 BILDIRICI M, ERSIN Ö Ö. Improving forecasts of GARCH family models with the artificial neural networks: an application to the daily returns in Istanbul Stock Exchange[J]. Expert Systems with Applications,2009, 36(4): 7355–7362.   
5 HAMMAD A, ALI S M A, HALL E L. Forecasting the Jordanian stock price using artificial neural network [J]. Intelligent Engineering Systems through Artificial Neural Networks, 2007, 17.   
6 GRAVES A, MOHAMED A, HINTON G. Speech recognition with deep recurrent neural networks [C]//Acoustics, speech and signal processing （icassp), 2013 ieee international conference on. IEEE, 2013:6645–6649.   
[ 7 ] WANG S, JIANG J. Learning natural language inference with LSTM [J]. arXiv preprint arXiv: 1512. 08849,2015.   
[8] SUN Y, WANG X, TANG X. Deeply learned face representations are sparse, selective, and robust [C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 2892-2900.   
[9] CHEN K, ZHOU Y, DAI F. A LSTM-based method for stock returns prediction: a case study of China stock market[C]//Big Data （Big Data), 2015 IEEE International Conference on. IEEE， 2015： 2823-2824.   
[10] FISCHER T, KRAUSS C. Deep learning with long short-term memory networks for financial market predictions[J]. European Journal of Operational $\mathrm { R e ^ { - } }$ search, 2018, 270(2): 654-669.   
[11] TREISMAN A M, GELADE G. A feature-integration theory of attention[J]. Cognitive psychology, 1980,12(1):97–136.   
[12] LUONG M T, PHAM H, MANNING C D. Effective approaches to attention-based neural machine translation [J ]. arXiv preprint arXiv: 1508. 04025,2015.   
[13] WANG Y, HUANG M, ZHAO L. Attention-based lstm for aspect-level sentiment classification [C]// Proceedings of the 2016 conference on empirical methods in natural language processing. 2016: 606–615.   
[14] ZHOU P, SHI W, TIAN J, et al. Attention-based bidirectional long short-term memory networks for relation classification [C]//Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). 2016, 2: 207-212.   
[15] KARA Y, BOYACIOGLU M A, BAYKAN Ö K. Predicting direction of stock price index movement using artificial neural networks and support vector machines: the sample of the Istanbul Stock Exchange [J]. Expert systems with Applications, 2011, 38 (5)：5311-5319.   
[16 HOCHREITER S, SCHMIDHUBER J. Long shortterm memory[J]. Neural computation, 1997, 9(8): 1735-1780.   
[17] MIKOLOv T, KARAFI? T M, BURGET L, et al. Recurrent neural network based language model [C]//Eleventh Annual Conference of the International Speech Communication Association. 2010.   
[18 ABADI M, BARHAM P, CHEN J, et al. Tensorflow: a system for large-scale machine learning[C]// OSDI. 2016,16: 265-283.