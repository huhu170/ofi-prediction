# 基于优化 LSTM模型的股票预测

# 胡聿文

江西财经大学统计学院南昌330013

摘要 股票预测研究一直是困扰投资者的难题。以往，投资者采用传统分析方法如K 线图、十字线等方法来预测股票走势，但随着科技的进步和经济市场的发展，以及经济政策的变动，股票的价格走势受到越来越多方面因素的干扰，仅靠传统的分析方法远远不能解析出股票价格波动中隐藏着的重要信息，因此预测精度大打折扣。为了提高股票价格的预测精度，提出一种基于PCA和LASSO的LSTM神经网络股票价格预测模型。采用2015—2019 年平安银行(00001)五大类技术指标数据，通过PCA 和LASSO 方法对五大类技术分析指标进行降维筛选，再使用 LSTM 模型进行平安银行股票收盘价预测，对比前两种模型和单纯使用 LSTM 模型的预测效果稳定性及准确性。结果表明，相比于LASSO-LSTM 模型和 LSTM模型，PCA-LSTM 模型能够大幅削减数据冗余，并且获得了更优异的预测精度。

关键词：平安银行；技术指标；LSTM;PCA;LASSO

中图法分类号TP183

# Stock Forecast Based on Optimized LSTM Model

HU Yu-wen

College of Statistics, Jiangxi University of Finance and Economics, Nanchang 330013,China

Abstract Stock forecasting resarch has always been a problem that plagued investors. In the past, investors used traditional a nalysis methods such as K-line diagrams and Yin-Yang lines to predict stock trends. However, with the advancement of science and technoly, the development of ecnomic marketnd changes in economic policies, the price trend of a stock is distured y varius actors.Traditioal nalysis ethds are far frobeig able to aale the iforation in volaityof a stock.  predic tion auracy is greatly reduced. In order to improve the prediction accuracy of stok prices, this paper proposes a stock price pre diction model based on PCA,LASSO, nd LSTM neural networks. Based on the data of the five major categories of technical indcators of Ping An Bank(0001) from 2015 to 2019, the five major categories of indicators are reduced and screened using the PCA and LASSO methods,and the LSTM model is used to predict the closing price of Ping An Bank's stock, compared with he stability and accuracy of the previous two methods and using LSTM alone. The experimental results show that the PCA-LSTM model significantly reduces data redundancy and obtains beter prediction accuracy than the LASSO-LSTM model and LSTM model.

Keywords Ping An bank, Technical index, LSTM, PCA, LASSC

金融股票的预测一直是金融领域的研究热点，在方法上，大致可分为线性预测模型和非线性预测模型。其中，线性预测模型主要有移动平均自回归模型（ARIMA)、广义自回归条件异方差模型（GARCH)[1]、指数广义自回归条件异方差（EGARCH)[2]和整积的广义自回归条件异方差（IGARCH)[3]等。上述线性预测模型作为早期的股票预测模型，对整个金融股票的预测发展起到了举足轻重的推动作用。但鉴于金融时间序列的高噪声和非线性特征，想要通过线性预测模型去精准预测金融股票价格依旧十分困难。随着计算机科技的高速发展以及深度学习研究的日益精进，机器学习领域的神经网络被日益广泛地应用于股票预测并且已经取得了比线性预测模型更加高效和准确的预测结果。 $\mathrm { C a o ^ { [ 4 ] } }$ 使用BP神经网络和灰色GARCH-BP模型所预测出的精度要明显优于GARCH模型。由于神经网络预测模型具有显著的非线性，我们把神经网络模型归为非线性预测模型。神经网络分为两大类。第一类人是工神经网络（ANN），如MP神经网络和

BP神经网络。ANN作为早期的神经网络模型，在股票预测方面亦做出了杰出贡献：Deng[5]提出DAE-BP模型对股票先进行DAE降维，再使用BP神经网络进行股价预测，取得了不错的预测结果。可惜的是，ANN模型结构过于单一，存在以下问题：1)过拟合，导致模型的泛化能力大大减弱；2)存在局部极值问题，即梯度下降过程中达到局部极小值点就停止工作，不能精准下降至全局极小值点，导致模型预测能力大大减弱；3）优化过程中容易因为神经元权重过多、过繁，导致梯度消失或者梯度爆炸问题，最终使神经网络模型预测失效。第二类神经网络则是相对于ANN来说，更深层次、更高效的深度神经网络模型（DNN)，如卷积神经网络（CNN）、循环神经网络(RNN)和长短期记忆神经网络(LSTM)。这一类神经网络模型是当前研究金融预测领域最高效、前沿的预测模型，其具有多方面优势。1)对输入变量的形式没有限制，与预测问题可能相关的信息均可被作为模型输入，这一点极大满足了股票市场容易被各式各样的信息所干扰影响的特征。2)有效拟合输入变量间的非线性复杂关系，提高样本拟合程度，同时通过神经元权重循环使用原理，大大减少了神经元权重的数量，有效防止过拟合现象。3)通过DNN中tanh激活函数，能够显著解决ANN中的梯度爆炸和梯度消失问题。影响股票价格波动的因素有公司财务面、基本面和相关政策以及投资者心理情绪。投资界有一句行话：关于股市的一切影响因素都可以在股价中体现，但是投资者心理情绪对股票在一段时间内的波动具有举足轻重的影响。Zheng在各自研究结果中明确表示投资者的投资情绪与股票收益存在显著的关联性[6-7]。因此，本文在股票价格影响因素选取中创新性地将消费者情绪和财务数据、基本面数据等多种数据融合，通过深度学习LASSO方法和PCA分析法对影响股票价格的多种因素进行降维筛选[8-9]，使各输入数据之间的相关性最大化，再分别导入目前最前沿的LSTM神经网络模型[10-14]进行预测，并进行准确性和稳定性的对比，发现其中最高效的股票模型预测方法。

# 1研究方法

# 1.1 LASSO

实证分析中，通常会设置尽可能多的自变量，选取自变量时容易出现偏主观意愿的疏漏，从而导致实证分析失真。而LASSO方法是一个能够客观筛选有效变量并且解决多重共线性等问题的估计方法。它是1997年由Tibshirani提出的一种压缩估计方法，通过构造一个惩罚函数，让回归系数的绝对值之和在小于一个常数的约束条件下，使得回归模型残差平方和最小，产生严格等于零的回归系数，从而有效解决回归模型中的多重共线性问题[15]。

LASSO方法是在普通线性回归模型中增加 $L _ { 1 }$ 惩罚项，普通线性模型的LASSO估计为：

$$
\begin{array} { l } { { \wedge _ { \mathrm { { L a s s o } } } = \underset { \beta \in { \cal R } ^ { d } } { \arg \operatorname* { m i n } } \parallel Y - X \beta \parallel ^ { 2 } } } \\ { { \mathrm { s . ~ t . ~ } \displaystyle \sum _ { j = 1 } ^ { d } | \beta _ { j } | \leqslant t , t \geqslant 0 } } \end{array}
$$

等价于：

$$
\stackrel { \wedge } { \beta } _ { \scriptscriptstyle \mathrm { L a s s o } } = \arg \operatorname* { m i n } _ { \beta \in { \cal R } ^ { d } } ( \mathrm { ~ \| ~ } Y - X \beta \| { \mathrm { ~ } } ^ { 2 } + \lambda \sum _ { j = 1 } ^ { d } | \beta _ { j } | )
$$

其中， $t$ 与 $\lambda$ 一一对应，为调节系数。

令 $\cdot t _ { 0 } = \sum _ { j = 1 } ^ { d } \mid \bigwedge _ { \beta _ { j } } ^ { \wedge } \left( O L S \right) \mid$ ,当 $t { < } t _ { 0 }$ 时，一部分系数就会被压缩至0，从而降低 $X$ 的维度，达到减小模型复杂度的目的。例如：如果 $t = t _ { 0 } / 2$ ,粗略地说，模型中非零系数个数就会由 $d$ 大约减少至 $d / 2$ 。

最后通过对调节系数 $\lambda$ 的控制，可实现变量筛选。

1.2 PCA

主成分分析法是一种降维的统计方法，它借助于一个正交变换，将其分量相关的原随机向量转化成其分量不相关的新随机向量，这在代数上表现为将原随机向量的协方差阵变换成对角形阵，在几何上表现为将原坐标系变换成新的正交坐标系，使之指向样本点散布最开的 $\boldsymbol { \phi }$ 个正交方向；然后对多维变量系统进行降维处理，使之能以一个较高的精度转换成低维变量系统，再通过构造适当的价值函数，进一步把低维系统转化成一维系统。

(1)原始指标数据的标准化采集 $\boldsymbol { \phi }$ 维随机向量 $x = ( x _ { 1 }$ ,$x _ { 2 } \ : , x _ { 3 } \ : , \cdots , x _ { p } \ : ) ^ { \mathrm { T } } \ : , \tau$ $n$ 个样品 $x _ { i } = ( x _ { i 1 } , x _ { i 2 } , x _ { i 3 } , \cdots , x _ { i p } ) ^ { \mathrm { T } } , i = 1$ ,

$2 , \cdots , n ( n > p )$ ，构造样本阵，对样本阵元进行如下标准化变换： $Z _ { i j } = { \frac { x _ { i j } - { \bar { x } } _ { j } } { s _ { j } } }$ xj−xi, i=1,2,…,n; j=1,2,…, φ。其中,xj =${ \frac { \sum _ { i = 1 } ^ { n } x _ { i j } } { n } } , { s _ { j } } ^ { 2 } = { \frac { \sum _ { i = 1 } ^ { n } ( x _ { i j } - x _ { j } ^ { - } ) ^ { 2 } } { n - 1 } }$ 。从而得标准化阵 $Z$ 。

(2)对标准化阵 $Z$ 求相关系数矩阵：

$$
R { = } [ r _ { i j } ] _ { \scriptscriptstyle P \times \scriptscriptstyle P } { = } \frac { Z ^ { \mathrm { T } } Z } { n - 1 }
$$

其中 $, r _ { i j } = \frac { \sum z _ { k j } \mathrm { ~ . ~ } z _ { k j } } { n - 1 } , i , j = 1 , 2 , \cdots , _ { \mathit { p } } { _ { \mathrm { ~ } } }$

(3)解样本相关矩阵 $R$ 的特征方程：

$| R - \lambda I _ { P } | = 0$ 得 $\boldsymbol { \phi }$ 个特征根，确定主成分。按 $\frac { \underset { j = 1 } { \overset { m } { \sum } } \lambda _ { j } } { \underset { j = 1 } { \overset { p } { \sum } } \lambda _ { j } } \geqslant 0$ -85确定 $m$ 值，使信息的利用率达 $8 5 \%$ 以上。对每个 $\lambda _ { j } \ : , j = 1$ ,$2 , \cdots , m$ ，解方程组 $R b { = } \lambda _ { j } b$ 得单位特征向量 $\mathbf { \partial } \cdot \mathbf { \partial } b _ { j } ^ { o }$ 。

(4)将标准化后的指标变数转换为主成分 ${ U _ { i j } } = _ { z _ { i } ^ { \mathrm { T } } } b _ { j } ^ { \circ } , j =$ $1 , 2 , \cdots , m$ 。其中， $U _ { 1 }$ 称为第一主成分， $U _ { 2 }$ 称为第二主成分，以此类推， $U _ { p }$ 称为第 $\boldsymbol { \phi }$ 主成分。

(5)对 $m$ 个主成分进行综合评价。

对 $m$ 个主成分进行加权求和，即得最终评价值，权数为每个主成分的方差贡献率。

# 1.3 LSTM

长短时记忆神经网络（LongShort-termMemoryNet-works，LSTM)是一种特殊的RNN类型，可以学习长期依赖信息。RNN神经网络模型一直被广泛用于语言识别和文本分类等多个研究领域[16]。相比于人工神经网络模型(ANN)而言，RNN神经网络模型可以循环利用神经元的权重参数，能够很好地将历史数据相关信息应用到预测中去。然而，RNN神经网络模型的误差反向传播算法只是像ANN神经网络模型中一样简单，权重的重复利用能够带来好处，也会带来很大弊端，例如梯度爆炸和梯度消失问题，即对历史数据的长期依赖性问题无法有效解决。为解决这两大难题，机器学习科研工作者们研究出长短时记忆神经网络模型(LSTM)[17]，如图1所示。

![](images/1f5e30a1410d22ad8e6a7f9a89c53f27f03a2f2c26fda2e41ecc1b5bade03f78.jpg)  
图1LSTM模型神经元结构示意图 Fig. 1 Neuron structure of LSTM model

LSTM模型相较于RNN模型最明显的改进是增加了1个细胞状态C和3个阀门，3个阀门分别是遗忘门 $f$ 、输出门$o$ 和输入门i。在LSTM模型误差反向传播校正权重时，有些误差可以直接通过输入门传递给下一层神经元，有些误差则可以通过遗忘门去进行数据遗忘，这样就解决了梯度爆炸与消失的难题，即有效地处理历史数据中相关信息的冗余等问题[18-19]。本文研究的股票价格预测是典型的时序问题，且某一个时刻的价格受前一时刻和历史多时刻价格影响，所以选择LSTM模型进行股票价格预测。

# 2股票价格预测的实证分析

# 2.1数据来源及指标选取

文章所采用的数据为2015年1月5日至2020年2月7日的平安银行(0000001)股票数据(数据来源于通达信金融终端)，共1240条数据。其中 $80 \%$ 作为训练集用于训练模型，其余 $20 \%$ 作为测试集中来验证模型的泛化能力。

在指标选取的过程中，应尽可能全方面地考虑影响因素，全方位地对问题进行分析，尤其对股票价格波动这种影响因素较多且各因素之间并不呈明显线性关系的难题，在指标选择过程中更应该精准筛选。相较于其他研究，实验选用股价的开盘价、最高最低价、成交量以及一般技术指标OBV、KDJ、BIAS等常见的技术指标，本文创新性地添加了最前沿的CCI，MFI，MTM等若干股价判断技术指标以及准确反映投资者心理情绪的PSY指标。这些技术指标能够多方位地涵盖股价波动的潜在信息，具有很强的股价解释性。为了更清楚地对这57个技术指标进行理解，表1进行了详细说明。

表1技术指标变量解释  
Table 1 Explanation of technical index variables   

<table><tr><td>一级指标</td><td>二级指标</td><td>三级指标</td><td>变量</td><td>解释</td></tr><tr><td rowspan="10"></td><td rowspan="2">BRAR</td><td>AR</td><td>Z1</td><td rowspan="2">判断行情热点</td></tr><tr><td>BR</td><td>Z2</td></tr><tr><td rowspan="5">CR</td><td>CR.CR</td><td>Z3</td><td rowspan="4">买入卖出信号</td></tr><tr><td>CR. MA1</td><td>Z4</td></tr><tr><td>CR.MA2</td><td>Z5</td></tr><tr><td>CR.MA3</td><td>Z6</td></tr><tr><td rowspan="2"></td><td>CR. MA4</td><td>Z7</td></tr><tr><td>VR.VR</td><td>Z8</td><td rowspan="2">买入卖出信号</td></tr><tr><td rowspan="2">VR</td><td>VR. MAVR</td><td>Z9</td></tr><tr><td>MASS. MASS</td><td>Z10</td><td rowspan="2">买入卖出信号</td></tr><tr><td rowspan="2">MASS</td><td>MASS. MAMASS</td><td>Z11</td></tr><tr><td>PSY. PSY</td><td>Z12</td><td rowspan="2">研究投资者对股市 涨跌产生心理波动</td></tr><tr><td rowspan="2">PSY</td><td>PSY. PSYMA</td><td>Z13</td></tr><tr><td>NVI.NVI</td><td>Z14</td><td rowspan="2">追踪大户资金流向</td></tr><tr><td rowspan="2">NVI PVI</td><td>NVI. MA</td><td>Z15</td></tr><tr><td>PVI. PVI</td><td>Z16</td><td rowspan="2">追踪散户资金流向</td></tr><tr><td rowspan="2">CCI</td><td>PVI. MA</td><td>Z17</td></tr><tr><td></td><td>CCI. CCI</td><td>Z18 判断信号</td></tr><tr><td rowspan="19">超买超 卖型</td><td rowspan="3">KDJ</td><td>KDJ.K</td><td>Z19</td><td rowspan="3">买进卖出信号</td></tr><tr><td>KDJ.D</td><td>Z20</td></tr><tr><td>KDJ.J</td><td>Z21</td></tr><tr><td rowspan="2">MFI MTM</td><td>MFI. MFI</td><td>Z22</td><td rowspan="2">短线买进卖出信号 买入卖出信号</td></tr><tr><td>MTM.MTM</td><td>Z23</td></tr><tr><td rowspan="2">OSC</td><td>MTM. MTMMA</td><td>Z24</td><td rowspan="2">多/空头市场</td></tr><tr><td>OSC. OSC</td><td>Z25</td></tr><tr><td rowspan="2">ROC</td><td>OSC. MAOSC</td><td>Z26</td><td rowspan="2">判断信号 超买超卖界定信号</td></tr><tr><td>ROC. ROC</td><td>Z27</td></tr><tr><td rowspan="2">SKDJ</td><td>ROC. MAROC</td><td>Z28</td><td rowspan="2">买入卖出信号</td></tr><tr><td>SKDJ.K</td><td>Z29</td></tr><tr><td rowspan="2">BIAS</td><td>SKDJ.D</td><td>Z30</td><td rowspan="2">判断股价拉升</td></tr><tr><td>BIAS. BIAS1</td><td>Z31</td></tr><tr><td rowspan="2">WR</td><td>BIAS. BIAS2</td><td>Z32</td><td rowspan="2">拉回信号</td></tr><tr><td>BIAS. BIAS3</td><td>Z33</td></tr><tr><td rowspan="2"></td><td>WR. WR1</td><td>Z34</td><td rowspan="2">股价转强转弱信号</td></tr><tr><td>WR.WR2</td><td>Z35</td></tr></table>

（续表）  

<table><tr><td>一级指标</td><td>二级指标</td><td>三级指标</td><td>变量</td><td>解释</td></tr><tr><td rowspan="9">趋势型</td><td rowspan="4">DMI</td><td>DMI. PDI</td><td>Z36</td><td rowspan="4">多/空头市场</td></tr><tr><td>DMI. MDI</td><td>Z37</td></tr><tr><td>DMI. ADX</td><td>Z38</td></tr><tr><td>DMI. ADXR</td><td>Z39</td></tr><tr><td rowspan="2">DPO</td><td>DPO. DPO</td><td>Z40</td><td rowspan="2">多/空头市场 判断信号</td></tr><tr><td>DPO. MADPO</td><td>Z41</td></tr><tr><td rowspan="3">MACD</td><td>MACD. DIF</td><td>Z42</td><td rowspan="3">买入卖出信号</td></tr><tr><td>MACD. DEA</td><td>Z43</td></tr><tr><td>MACD. MACD</td><td>Z44</td></tr><tr><td rowspan="6">成交量型</td><td rowspan="3">AMO</td><td>AMO. AMOW</td><td>Z45</td><td rowspan="3">判断热门/ 非热门股票</td></tr><tr><td>AMO. AMO1</td><td>Z46</td></tr><tr><td>AMO. AMO2</td><td>Z47</td></tr><tr><td rowspan="2">OBV</td><td>OBV. OBV</td><td>Z48</td><td rowspan="2">短线买入卖出信号</td></tr><tr><td>OBV. MAOBV</td><td>Z49</td></tr><tr><td>VOL</td><td>VOL. MAVOL1</td><td>Z50</td><td rowspan="2">判断热门/ 非热门股票</td></tr><tr><td>VOL</td><td>VOL. MAVOL2</td><td>Z51</td></tr><tr><td rowspan="6">股价</td><td>开盘价</td><td>开盘价</td><td>Z52</td><td>开盘时股票价格</td></tr><tr><td>最高价</td><td>最高价</td><td>Z53</td><td>股票当日最高价格</td></tr><tr><td>最低价</td><td>最低价</td><td>Z54</td><td>股票当日最低价格</td></tr><tr><td rowspan="2">成交量</td><td>成交量</td><td>Z55</td><td>股票当日成交量 股价10日均线</td></tr><tr><td>MA.MA1</td><td>Z56</td><td></td></tr><tr><td>MA</td><td>MA.MA5</td><td>Z57</td><td>股价120日均线</td></tr></table>

# 2.2数据的筛选

表1所列5个一级指标、28个二级指标和57个三级指标描述了股价的波动影响因素。由于指标个数较多，为了不给网络运行带来负担，提高LSTM神经网络的预测能力，本文分别采用主成分分析法和LASSO回归法对57个指标进行筛选。

# (1)LASSO回归法

对数据进行标准化处理，接下来利用LASSO模型对57个指标数据进行分析，以进行变量筛选。运行结果显示，模型91步可以得到LASSO的全部解。根据最小角回归原理，选择 $C _ { P }$ 统计量最小值对应的拟合方程，第80步的 $C _ { P }$ 值最小为45.04705，模型最优。根据LASSO结果，筛除变量为0的MACD. DEA,OBV. OBV,BRAR. BR, BRAR. AR, NVI. MA,PVI.PVI,KDJ.K,ROC.ROC和SKDJ.K9个变量，选取50个变量进行后续的神经网络模型预测。

图2为LASSO模型筛选结果。图中右侧数据分别代表相应自变量。

![](images/9f481f6889f3104d2db0fa7e658d78eb2c71944f49c2d067d0331e9868a33e4a.jpg)  
图2LASSO模型筛选结果  
Fig. 2 Screening results of lasso model

# (2)主成分分析法

通过RStudio软件对平安银行57个指标数据进行标准化处理，然后画出碎石图。由图3可知，特征值大于1的主成分个数有9个。

![](images/caa02f1e219597b5cac295b3f365998dc8d213dcf71aef971b73c99698a07e0c.jpg)  
图3碎石图  
Fig. 3 Scree plot

对数据进行提取主成分，并对提取出的主成分结果取$8 5 \%$ 贡献率，我们得到8个主成分。为了使主成分分析法解释更高效并且最小化LSTM神经网络模型的数据冗余，我们保留数值大于0.5的参数数据。处理后的8个主成分如下：

id: $x _ { 1 } = 0 . 7 7 7 ~ Z _ { 7 } + 0 . 5 9 0 ~ Z _ { 8 } + 0 . 6 9 1 ~ Z _ { 9 } + \cdots + 0 . 8 2 6 ~ Z _ { 5 9 }$ d: $x _ { 2 } = 0 . 5 4 4 Z _ { 1 } + 0 . 5 6 5 Z _ { 2 } + 0 . 5 2 3 Z _ { 3 } + \cdots + 0 . 9 2 3 Z _ { 2 2 }$ d: $x _ { 3 } = 0 . 6 3 5 ~ Z _ { 3 2 } + 0 . 5 7 2 ~ Z _ { 3 3 } + 0 . 5 7 7 ~ Z _ { 4 2 } + \cdots + 0 . 6 3 2 ~ Z _ { 5 5 }$ $\alpha _ { 4 } = 0 . 8 0 0 ~ Z _ { 3 6 } + 0 . 9 1 7 ~ Z _ { 3 7 } + 0 . 5 8 4 ~ Z _ { 4 2 } + \cdots + 0 . 6 9 1 ~ Z _ { 5 8 }$ … $\alpha _ { 8 } = 0 . 7 5 1 Z _ { 1 } + 0 . 7 4 4 Z _ { 2 } + 0 . 7 6 1 Z _ { 3 } + \cdots + 0 . 8 0 3 Z _ { 6 }$

# 2.3预测方法及思路

为客观比较LASSO-LSTM和PCA-LSTM之间的预测效率，我们加入单纯LSTM模型的预测结果，进行三者对比。

# （1)LSTM神经网络方法及预测思路

LSTM神经网络用历史1240个交易日的数据信息对股票价格走势进行预测。输入数据为未做数据筛选的所有57个参数变量，输出数据为历史股价下一日的收盘价预测值。

(2)LASSO-LSTM模型方法及预测思路

通过LASSO回归法构造惩罚函数，将历史1240个交易日内的57个参数变量进行去共线性筛选，留下的50个参数变量具有低共线性、高相关度等特征，再将其当作输入变量输入LSTM神经网络模型中，输出变量是当日历史数据的下一日收盘价预测值。

# (3)PCA-LSTM模型方法及预测思路

通过PCA分析法，从57个原始数据提取出8个主成分用于LSTM模型输入，这8个主成分因子分别以不同的参数系数囊括了57个历史数据的信息，显著地精简了神经网络模型输入端，同时又不丢失重要数据信息。输出变量是当日历史数据的下一日收盘价预测值。

# 2.4实证结果对比

现有运用神经网络进行股票预测的研究实验较多是将历史数据全部放入训练集中进行模型训练，将最新一天的股价作为预测数据[20-21]。以该方式预测可以得出非常拟合的预测结果图，是因为神经网络模型独特的多次校正权重原理能够让神经网络模型最大程度地拟合出与历史股价重合的预测股价，但这在实际炒股应用中的失真率非常大，毫无实际应用价值。因此，本文采用历史股价 $80 \%$ 的数据即前960日的历史股价作为训练集，剩余 $20 \%$ 的数据即第960日至1240日的历史股价作为预测集。不同的LSTM模型超参数，对LSTM模型的预测能力有着显著的影响，例如神经元层数及每层神经元中的神经元个数的改变会使模型运算繁琐度指数增长且影响最后的预测精度，学习率的千分位改变会显著影响模型在梯度下降时的效率与准确率[22]。

因此，在本次预测实验中设定了不同的LSTM模型超参数进行预测结果比较。在目前的LSTM研究文章中，LSTM模型神经元层数设为2层和3层时效率最高，神经元个数设置为8不变，学习率通常设置为0.001，这里另设对比学习率0.005，迭代次数设置为500和1000次，激活函数设置为LSTM模型常用的tanh函数不变。通过判断实验结果的预测精度以及预测股价和历史股价间走势的拟合程度，来确定何种方式的模型预测结果最有成效。预测精度评估分别采用均方误差函数（MSE）、均方根误差(RMSE)和平均绝对误差（MAE)，三者的值越小表明预测结果越精确。表2一表4是3种预测模型的实验结果。

表2LSTM在LSTM模型参数下的预测结果  
Table 2 Prediction results of LSTM under LSTM model parameters   

<table><tr><td>迭代 次数</td><td>激活 函数</td><td>学习率</td><td>神经元 层数</td><td>神经元 个数</td><td>MSE</td><td>RMSE</td><td>MAE</td></tr><tr><td>500</td><td>tanh</td><td>0.001</td><td>3</td><td>8</td><td>0.0078</td><td>0.1344</td><td>0.2280</td></tr><tr><td>1000</td><td>tanh</td><td>0.001</td><td>3</td><td>8</td><td>0.0014</td><td>0.0313</td><td>0.1754</td></tr><tr><td>500</td><td>tanh</td><td>0.005</td><td>3</td><td>8</td><td>0.0036</td><td>0.0573</td><td>0.1884</td></tr><tr><td>1000</td><td>tanh</td><td>0.005</td><td>3</td><td>8</td><td>0.0151</td><td>0.0343</td><td>0.1677</td></tr><tr><td>500</td><td>tanh</td><td>0.001</td><td>2</td><td>8</td><td>0.0092</td><td>0.0303</td><td>0.1822</td></tr><tr><td>1000</td><td>tanh</td><td>0.001</td><td>2</td><td>8</td><td>0.0017</td><td>0.0382</td><td>0.1467</td></tr><tr><td>500</td><td>tanh</td><td>0.005</td><td>2</td><td>8</td><td>0.0012</td><td>0.0351</td><td>0.2516</td></tr><tr><td>1000</td><td>tanh</td><td>0.005</td><td>2</td><td>8</td><td>0.0042</td><td>0.0490</td><td>0.2382</td></tr></table>

从表2单纯使用LSTM模型对股票的预测结果来看，在LSTM模型设置迭代次数为500次、激活函数为tanh、学习率为0.005、神经层为3层、神经元为8个时，MSE结果最小，为0.0012;在设置迭代次数500次、激活函数tanh、学习率0.001、神经层2层、神经元8个时，RMSE结果最小，为0.0303;在LSTM模型设置迭代次数1000次、激活函数tanh、学习率0.001、神经层3层、神经元8个时，MAE结果最小，为0.1467。图4一图6分别为单纯LSTM模型下，3个误差值最小条件时的预测图。

![](images/710daa73b939e6182d5d3fcc27a6a1765a56b05e2e8a2d2d1157457e1f961743.jpg)  
图4迭代500次、学习率0.005、3层神经元时的MSE Fig.4 MSE when iteration $= 5 0 0$ ,learning rate $= 0 . \ 0 0 5$ and neurons $; = 3$

![](images/2d782b88a738cad8b24a8ed2731dab2632647b7fafd1fa041f5768d5f3540dd3.jpg)  
图5迭代500次、学习率0.001、2层神经元时的RMSE Fig.5 RMSE when iteration $= 5 0 0$ ,learning rate $= 0 . 0 0 1$ and neurons $= 2$

![](images/5bcb3b30d3c1a622a3623efcce98bdfe137223bd4d6d140dfacd3f3d3a39c732.jpg)  
图6迭代1000次、学习率0.001、3层神经元时的MAE Fig.6 MAE when iteration $= 1 0 0 0$ ,learning rate $= 0 . 0 0 1$ and neurons $; = 3$

从图4一图6可以看出，单纯使用LSTM模型进行股价预测，以MSE和 $M A E$ 为误差函数做预测时，在预测集一端的误差非常大，预测股价呈现乱序、波动大等负面情况，无法拟合预测集的真实股价的走势和真实股价。相比前两者的预测结果，以RMSE为误差函数做预测时，预测股价则能够相对较好地贴合预测集中真实股价的走势，但是预测结果与预测集中第960日至第1240日的真实历史股价间的重合率过低，无法较好地拟合。

从表3LASSO-LSTM模型对股票预测结果来看，在LSTM模型设置迭代次数1000次、激活函数tanh、学习率0.001、神经层2层、神经元8个时，MSE结果最小，为0.0009；在设置参数为迭代次数1000次、激活函数tanh、学习率0.001、神经层3层、神经元8个时，RMSE结果最小，为0.0272;在LSTM模型设置迭代次数500次、激活函数tanh、学习率0.001、神经层3层、神经元8个时，MAE结果最小，为0.1398。图7一图9分别为单纯LSTM模型下，3个误差值最小条件时的预测图。从图7一图9可以看出，尽管LAS-SO-LSTM模型在以MSE，RMSE，MAE三者为误差函数时取得的预测精度有较为可观的结果，但是从预测图的走势来看，LSTM模型所得的预测值呈现杂乱性、无序性并且与历史真实股价严重偏离的现象，因此，LASSO-LSTM模型在股价预测应用上是远远不能付诸于实际中的。

表3LASSO-LSTM在LSTM模型参数下的预测结果  
Table 3 Prediction results of lasso LSTM under LSTM model parameters   

<table><tr><td>迭代次数</td><td>激活函数</td><td>学习率</td><td>神经元层数</td><td>神经元个数</td><td>MSE</td><td>RMSE</td><td>MAE</td></tr><tr><td>500</td><td>tanh</td><td>0.001</td><td>3</td><td>8</td><td>0.0039</td><td>0.0419</td><td>0.1398</td></tr><tr><td>1000</td><td>tanh</td><td>0.001</td><td>3</td><td>8</td><td>0.0050</td><td>0.0272</td><td>0.2209</td></tr><tr><td>500</td><td>tanh</td><td>0.005</td><td>3</td><td>8</td><td>0.0098</td><td>0.0535</td><td>0.1793</td></tr><tr><td>1000</td><td>tanh</td><td>0.005</td><td>3</td><td>8</td><td>0.0025</td><td>0.0423</td><td>0.1492</td></tr><tr><td>500</td><td>tanh</td><td>0.001</td><td>2</td><td>8</td><td>0.0013</td><td>0.0605</td><td>0.1965</td></tr><tr><td>1000</td><td>tanh</td><td>0.001</td><td>2</td><td>8</td><td>0.0009</td><td>0.0327</td><td>0.2346</td></tr><tr><td>500</td><td>tanh</td><td>0.005</td><td>2</td><td>8</td><td>0.0062</td><td>0.0616</td><td>0.2441</td></tr><tr><td>1000</td><td>tanh</td><td>0.005</td><td>2</td><td>8</td><td>0.0039</td><td>0.0607</td><td>0.1856</td></tr></table>

![](images/c86fa808fa245a681cb8be67cae8ee307fd814318efd0052a88e2999c1842db8.jpg)  
图7迭代1000次、学习率0.001、2层神经元时的MSE Fig.7 MSE when iteration $= 1 0 0 0$ ,learning rate $= 0 , 0 0 1$ and neurons ${ \it \Omega } = 2 { \it \Omega }$

![](images/ebe48b2403f77b983892c6ce29d7bf60bbe7086d83420767139eaea39de32d32.jpg)  
图8迭代1000次、学习率0.001、3层神经元时的RMSE Fig.8 RMSE when iteration $\scriptstyle 1 = 1 0 0 0$ ,learning rate $= 0 , 0 0 1$ and neurons $; = 3$

从表4PCA-LSTM模型对股票预测结果来看，在LSTM模型设置迭代次数1000次、激活函数tanh、学习率0.001、神经层2层、神经元8个时，MSE结果最小，为0.0005；在LSTM模型设置迭代次数500次、激活函数tanh、学习率0.001、神经层3层、神经元8个时，RMSE结果达到最小值

0.0283；同时，在此条件下， $M A E$ 也为最小值0.2101。值得一提的是，MSE的值也很可观，仅比最小0.0005略微高出0.0002。图10一图12分别为PCA-LSTM模型下，3个误差值最小值条件时的预测图。

![](images/e9eae48c358afadf03612a22d69a99d44343ef693b070e7823ba1997042eed91.jpg)  
图9迭代500次、学习率0.001、3层神经元的时MAE Fig.9 MAE when iteration $= 5 0 0$ ,learning rate $= 0 , 0 0 1$ and neurons $= 3$

表4PCA-LSTM在LSTM模型参数下的预测结果  
Table 4 Prediction results of PCA-LSTM under LSTM mode parameters   

<table><tr><td>迭代 次数</td><td>激活 函数</td><td>学习率</td><td>神经元 层数</td><td>神经元 个数</td><td>MSE</td><td>RMSE</td><td>MAE</td></tr><tr><td>500</td><td>tanh</td><td>0.001</td><td>3</td><td>8</td><td>0.0007</td><td>0.0283</td><td>0.2101</td></tr><tr><td>1000</td><td>tanh</td><td>0.001</td><td>3</td><td>8</td><td>0.0012</td><td>0.0629</td><td>0.2221</td></tr><tr><td>500</td><td>tanh</td><td>0.005</td><td>3</td><td>8</td><td>0.0071</td><td>0.0501</td><td>0.3365</td></tr><tr><td>1000</td><td>tanh</td><td>0.005</td><td>3</td><td>8</td><td>0.0019</td><td>0.0423</td><td>0.2308</td></tr><tr><td>500</td><td>tanh</td><td>0.001</td><td>2</td><td>8</td><td>0.0017</td><td>0.0513</td><td>0.2532</td></tr><tr><td>1000</td><td>tanh</td><td>0.001</td><td>2</td><td>8</td><td>0.0005</td><td>0.0629</td><td>0.2160</td></tr><tr><td>500</td><td>tanh</td><td>0.005</td><td>2</td><td>8</td><td>0.0060</td><td>0.0584</td><td>0.2355</td></tr><tr><td>1000</td><td>tanh</td><td>0.005</td><td>2</td><td>8</td><td>0.0015</td><td>0.0314</td><td>0.2814</td></tr></table>

![](images/c712a6f6a78691b9c2595d2d8032b563c73017e720bab194ad9493123f0197fb.jpg)  
图10迭代1000次、学习率0.001、2层神经元时的MSEFig. 10 MSE when iteration $= 1 0 0 0$ ,learning rate $= 0 , 0 0 1$ andneurons ${ \it \Omega } = 2 { \it \Omega }$

![](images/b3f1c1ce31dc153bc6dfe3ad67babdab67100c6c4f1c9c74b5240a56488a07f5.jpg)  
图11迭代500次、学习率0.001、3层神经元时的RMSE Fig. 11 RMSE when iteration $= 5 0 0$ ,learning rate $= 0 . \ 0 0 1$ and neurons $; = 3$

![](images/86fd3194b229d47af5f864f985abbd7916ce067c3bbb0052f77a7e133659c7a8.jpg)  
图12迭代500次、学习率0.001、3层神经元时的MAE Fig. 12 MAE when iteration $= 5 0 0$ ,learning rate $= 0 . 0 0 1$ and neurons $; = 3$

从图10一图12来看，PCA-LSTM模型无论是MSE还是RMSE作为误差函数的预测精度都很卓效，在实际预测中，预测模型所得的预测股价与第960日至第1160日之间的真实股价近乎重合，且与第1160日至第1240日之间的历史股价走势高度拟合。而PCA-LSTM模型在以MAE为误差函数的预测实验中，也有较好的拟合结果。

结束语本文使用LSTM模型、LASSO-LSTM模型和PCA-LSTM模型对上证股市的平安银行股票进行股价预测，通过对比3种预测模型，发现PCA-LSTM模型对平安银行的股票价格预测有很强的泛化能力，其预测效果也是三者中最为精确高效的。目前市面上较多股市投资团队仅将股票的相关影响因素单纯应用LSTM模型进行预测，影响因素数据繁杂冗余，导致模型预测精度大打折扣。而PCA-LSTM模型在数据预测前将股票的相关影响因素数据进行主成分分析，对数据进行因子旋转提炼，提取出以最少主成分数量代表最多影响因素数量的主成分作为LSTM神经网络模型的输入变量，能够让预测模型更加高效，从而具有较好的预测效果。

# 参考文献

[1] YANG Q,CAO X B. Analysis and prediction of stock price

vaseu un aruagarcn muuer L.rracuce anu Unuersanuug un Mathematics,2016,46(6) :80–86.   
[2] HUANG L J, JIN T X. Stability of Islamic stock market based on egarch-m model [J]. Gansu Academic Journal of theory, 2019 (6) :107–115.   
[3] FANG J. Empirical research on VaR Measurement of China's stock market: semi parametric method based on igarch [J]. Fi nancial Theory and Teaching, 2018(3) :15-18.   
[4] CAO X,SUN H B. Stock price prediction based on Grey GARCH model and BP neural network [J]. Software, 2017, 38(11):126-131.   
[5] DENG J K, WAN L, HUANG N N. Research on Stock Forecasting Based on dae-bp neural network [J]. Computer Engineering and Application,2019,55(3) :126-132.   
[6] ZHENG G J. Stock time series analysis and prediction based on Internet investor sentiment [D]. Hangzhou:Zhejiang University of technology, 2019.   
[7] LIANG X Z. Research on the relationship between investor sentiment and stock returns [D]. Beijing : Beijing Jiaotong University,2017.   
[8] WANG J Z. Application of improved adaptive lasso method in stock market [J]. Mathematical Statistics and Management, 2019,38(4):750–760.   
[9] YU H H,CHEN R D,ZHANG G P. A SVM Stock Selection Model within PCA[J]. Procedia Computer Science, 2014,31.   
[10] REN T Y. An Empirical Study of Stock Return and Investor Sentiment Based on Text Mining and LSTM[C] / Proceedings of the 2019 4th International Conference on Social Sciences and Economic Development(ICSSED 2019). 2019.   
[11] JIA M Z, HUANG J, PANG L H, et al. Analysis and Research on Stock Price of LSTM and Bidirectional LSTM Neural Network[C] // Proceedings of the 3rd International Conference on Computer Engineering, Information Science &. Application Technology(ICCIA 2019). 2019.   
[12] YANG Q, WANG C W. Global stock index prediction based on deep learning LSTM neural network [J]. Statistical Research, 2019,36(3):65-77.   
[13] PENG Y,LIU Y H,ZHANG R F. Modeling and analysis of stock price forecasting based on LSTM [J]. Computer Engineering and Application,2019, 55(11) :209–212.   
[14] CHEN J, LIU D X, WU D S. Research on stock index prediction method based on feature selection and LSTM model [J]. Computer Engineering and Application,2019,55(6) :108-112.   
[15] TIBSHIRANI R. Regression shrinkage and selection via the lasso:A retrospective[J]. Journal of the Royal Statistical Society: Series B(Statistical Methodology) , 2011, 73(3) :273–282.   
[16] HAN H, LIU G L, SUN T Y, et al. Text sentiment analysis based on multi attention level neural network [J]. Computer Engineering and Application.   
[17] PEI D W, ZHU M. Stock price prediction based on multi factor and multi variable long term and short term memory network [J]. Computer System Applications,2019, 28(8) :30–38.

[18] ZENG A, NIE W J. Stock recommendation system based on deep bidirectional LSTM J]. Computer Science, 2019,46（10): 84– 89.

[19] REN J, WANG J H, WANG C M, et al. Stock forecasting system based on elstm-1 model [J]. Statistics and Decision, 2019,35 (21):160-164.

[20] FENG Y X, LI Y M. Research on prediction model of CSI 300 index based on LSTM neural network [J]. Practice and Understanding of Mathematics, 2019, 49(7) :308–315.

[21] LI S S. Securities selection based on long term memory neural network [D]. Zhengzhou:Zhengzhou University, 2019.

[22] XU T T. Research on stock price rise and fall prediction based on LSTM neural network model [D]. Shanghai: Shanghai Normal University,2019.

![](images/ba2736c268cd87dcfb2471c0625717d3111959ec6408c2065f7cedd65f4e49ed.jpg)

HU Yu-wen, born in 1996, postgraduate. His main research interests include mathematical statistics, financial statistics and deep learning.

# （上接第136页）

[8] WANG H, CHEN X, TIAN S Z, et al. SAR image recognition based on few-shot learning[J]. Computer Science, 2020, 47(5 ) : 124–128.

[9] YU Y, WANG B, ZHANG L. Hebbian-based neural networks for bottom-up visual attention and its applications to ship detection in SAR images[J]. Neurocomputing, 2011, 74(11):2008– 2017.

[10] CHEN LF, LIU Y Z, ZHANG P, et al. Road extraction algorithm of multi-feature migh-resolution SAR image based on Multi-Path RefineNet[J]. Computer Science, 2020, 47(3) : 156– 161.

[11] ZHANG Z F,ZHAO Z, WEI J J, et al. A method for ship detection in SAR imagery based on improved human visual attention system[J]. Science of Surveying and Mapping, 2017, 42(4) : 108– 112.

[12] ZHANG M C, WU X Q, WANG P W. An adaptive approach for target detection in SAR images[J]. Computer Engineering and Applications, 2006(10) : 200–203.

[13] ZHAO Q H, WANG X, LI Y, et al. Ship detection optimization method in SAR imagery based on multi-feature weighting[J]. Journal on Communication,2020,41(3) :91-101.

[14] DU L, WANG Z C, WANG Y,et al. Survey of research progress on target detection and discrimination of single-channel SAR images for complex scenes[J]. Journal of Radars, 2020, 9(1) : 34– 54.

[15] CHEN S, WANG H, XU F, et al. Target classification using the deep convolutional networks for SAR images[J]. IEEE Transactions on Geoscience and Remote Sensing, 2016, 54(8): 4806- 4817.

[16] BENTES C, VELOTTO D, TINGS B. Ship classification in TerraSAR-X images with convolutional neural networks[J]. IEEE Journal of Oceanic Engineering,2017, PP(99) :1-9.

[17] NIU L. Sidelobe removal method for ship target based on SAR images[J]. Radar Science and Technology, 2018, 16(2) : 197– 200.

[18] GAO D, HAN S, VASCONCELOS N. Discriminant saliency, the detection of suspicious coincidences, and applications to visual recognition[J]. IEEE Transactions on Pattern Analysis 8. Machine Intelligence,2009,31(6) :989–1005.

[19] WANG X H. Moment technique and its applications in image processing and recognition[D]. Xi'an : Northwestern Polytechnical University, 2002.

[20] XIONG W, XU Y L,CUI Y Q, et al. Geometric feature extractionof ship in high-resolution synthetic aperture Radar images [J]. Acta Photonica Sinica, 2018,47(1) :55–64.

[21] ZHEN Y,LIU W,CHEN J H, et al. Geometric structure feature extraction ofship target in high-resolution SAR image[J]. Journal of Signal Processing, 2016,32(4) :424-429.

[22] PAPATSOUMA I,FARMAKIS N. Approximating Symmetric Distributions via Sampling and Coefficient of Variation[J]. Communications in Statistics,2020,49(1):61–77.

[23] AI J, YANG X, SONG J, et al. An adaptively truncated clutter statistics-based two-parameter CFAR detector in SAR imagery [J]. IEEE Journal of Oceanic Engineering, 2017, PP(99) :1-13.

[24] XIONG W,XU Y 1, YAO L B, et al. A new ship target detection algorithm based on SVM in high resolution SAR images[J]. Remote Sensing Technology and Application, 2018, 33 (1): 119– 127.

[25] ZHAO Z,JI K, XING X, et al. Adaptive CFAR detection of ship targets in high resolution SAR imagery[J]. Proceedings of Spie the International Society for Optical Engineering, 2013, 8917: 89170L-89170L-8.

![](images/b3ce125118ed5afb3aed751ec76f59e2671c0f9dceee0fe016c0c7dc3177559f.jpg)

YAN Jun, born in 1962, Ph.D. His main research interests include remote sen sing technology and application and so on.

![](images/cee97a0a3c83377bfa7d279d9cd5bb3181d42e1663ed2d8dab0824268c244ea3.jpg)

LU Lin-lin, born in 1984, Ph. D, associate professor. Her main research interests include urban remote sensing, remote sensing information extraction and classification.