# 基于深度学习和小波分析的 LSTM-Wavelet 模型股价预测

李 梦,黄章杰,徐健晖重庆工商大学 数学与统计学院,重庆

摘 要:针对股价数据具有高噪声、非线性和非平稳性等特征,使得股价精确预测非常困难的问题,提出小波-长短记忆网络( )模型应用于股价预测。 首先,利用小波( )分解降低金融时间序列的不稳定性,并分析小波系数的细节特征;接着,发挥长短记忆网络( )模型的优势,深层挖掘小波系数中的长期依赖关系,对分解后的各层小波系数分别建模预测;最后进行预测小波系数的数据重构。 使用中石油近两年的股价数据进行实证分析,以每个交易日的开盘价、最高价、最低价、交易量为特征输入,预测当日中石油的收盘价。 结果表明:相较于标准 模型和小波- ( - )模型,提出的 - 模型有更好的预测效果; 通过小波分析将复杂股票数据,分解为长短记忆网络( )容易识别的小波系数,根据各层小波系数不同的数据特征进行分层预测,提高了预测精度。

关键词:股价预测;小波分解;LSTM 模型;LSTM-Wavelet 模型 中图分类号: 文献标识码:A doi:10. 16055 / j. issn. 1672-058X. 2023. 0002. 015

# Stock Price Prediction with LSTM-Wavelet Model Based on Deep Learning and Wavelet Analysis

LI Meng, HUANG Zhangjie, XU Jianhui School of Mathematics and Statistics, Chongqing Technology and Business University, Chongqing 400067, China

Abstract: Accurate stock price prediction is very difficult due to the high noise, non-linearity, and non-smoothness of stock price data. A wavelet-long and short-term memory network ( LSTM-Wavelet) model was proposed for stock price prediction. First, wavelet decomposition was used to reduce the instability of financial time series, and the detailed features of wavelet coefficients were analyzed. Then, taking advantage of the long and short-term memory network (LSTM) model, the long-term dependencies in the wavelet coefficients were mined deeply and the predictions were modeled separately for each layer of the decomposed wavelet coefficients. Finally, the data reconstruction of the predicted wavelet coefficients was performed. Based on the empirical analysis of PetroChina’ s stock price data for the past two years, the opening price, high price, low price, and trading volume of each trading day were used as the characteristic inputs to predict the closing price of PetroChina on that day. The results showed that the proposed LSTM-Wavelet model had better prediction results compared with the standard LSTM model and the wavelet-ARIMA (ARIMA-Wavelet) model; the complex stock data were decomposed into wavelet coefficients easily recognized by Long Short Term Memory (LSTM) network through wavelet analysis, the stratified prediction was performed based on the data characteristics with different wavelet coefficients in each layer, and the prediction accuracy has been improved.

Keywords:stock price prediction; wavelet decomposition; LSTM model; LSTM-Wavelet model

# 1 引 言

金融市场的蓬勃发展 使得股价预测成为研究的热点和难点 它的研究可以带给投资者信心 并规避交易风险 对于市场交易具有重要指导作用 然而股价数据的时间序列具有高噪声 非线性和非平稳性等特征使得股价的预测非常困难 常用的股价预测模型主要有时间序列模型和基于机器学习的算法 时间序列模型包括自回归滑动平均模型 [1] 自回归条件异方差模型 [2] 广义自回归条件异方差模型[3]等 由于股市价格具有时间上的连续性在不考虑外界因素的影响下 采用时间序列模型能够较好地预测股市价格 当自变量和因变量之间的关系具有时变性时 传统的时间序列预测模型很难进行精准建模 此外 时间序列模型有平稳序列建模等前提假设 而股票数据具有非线性和非平稳性等特征 这进一步限制了时间序列模型在股市预测中的应用[4] 为了更好刻画金融时间序列的非线性 佟伟民等[5] 构建了小波分析方法与 模型相结合的 -模型 通过小波分析方法使得时间序列数据结构清晰明了 再与 模型结合进行预测 取得了较好的效果 但是由于 模型本身的问题 如没有考虑外界影响因素和被预测对象间的因果关系[6] 使得模型仍然很难精确预测高噪声 非平稳性的时间序列

将机器学习算法应用于金融数据预测主要有支持向量机 随机森林 算法等 如陈亚男等[7]提出 Bagging-SVM 模型用于股价趋势的预测 刘玉敏等[8]提出 - 模型用于成分股价格趋势预测 由于机器学习对模型的函数形式并未做出严格的平稳序列建模假设 对变量之间的相互作用以及参数的统计分布假设相较于传统计量方法更为宽松[9] 因此能够更好地用于非线性数据的分析与建模能够较好地处理非平稳数据的预测问题 但是股市容易受到企业内外部等多种因素的影响 金融时间序列还有维度高等特征 这使得传统机器学习算法很难精确预测高维度数据 随着人工智能的高速发展 神经网络模型被应用于高维度金融数据预测 如綦方中等[10]提出了 神经网络与 降维相结合的模型用于股价预测,陈祥一[11] 将卷积神经网络用于沪深 300 指数的预测 杨青等[12] 构建了深层 模型用于全球股指预测研究 这些神经网络模型能够从多维角度分析数据特征 对数据的深层次特征有较好的学习能力 因此有较好的预测效果 等 [13]将 模型与各种广义自回归条件异方差模型相结合 提出了一种新的混合长短期记忆模型来预测股票价格波动;Chen 等[14]采用 LSTM 模型对中国股票收益率进行建模和预测。

特别值得注意的是 模型 在学习异质性信息过程中强化有效因素能力突出 对于非线性时间序列有较好的拟合和预测能力 但股票数据具有高噪声 数据波动剧烈等特征 使用单一的神经网络模型应用股价预测,容易忽略不同特征对输出结果的影响[15]。 为此,本文构造了 模型应用于股价预测 通过小波分解获取特征更加明显的小波系数 利用 模型分层预测 以期实现对股票收盘价的精确预测 本文模型主要优点 通过小波分解将数据分解成结构更为简单变化趋势更加明显的序列 以方便建模 由于 具备良好的细节特征提取能力 将 模型应用于小波系数的分层建模 可根据每层系数特征 得到更加精确的预测结果,避免使用原始数据单一建模时容易造成的模型过拟合情况

# 2 理论基础

# 2. 1 小波分析

年 法国物理学家 首次提出了 小波分析 的概念 并与法国物理学家 共同提出了连续小波变换体系 其空间基具有伸缩平移不变性 为小波理论的形成奠定了基础[16] 年 在构造正交小波基时提出了多分辨分析 ( Multi - Resolution的概念 其主要思想是将平方可积的实数空间$L ^ { 2 } ( R )$ 分解成一串具有不同分辨率的子空间序列,该子空间序列的极限便是 $L ^ { 2 } ( R )$ 然后将 $L ^ { 2 } ( R )$ 中的序列$f ( t )$ 描述为具有一系列近似函数的逼近极限 这一系列近似函数都是 $f ( t )$ 在这些子空间序列上的投影 根据这些投影 在不同分辨率子空间上来分析和研究 $f ( t )$ 的形态和特征 这个过程就叫作多分辨分析[17]

构造 $L ^ { 2 } ( R )$ 空间内一个子空间列 $\{ V _ { j } \} _ { j \in z }$ ,若空间$L ^ { 2 } ( R )$ 具有多分辨分析 则对于给定的 $f ( t )$ 有

$$
\begin{array} { r l } { f ( t ) = \displaystyle \sum _ { m = m _ { 0 } + 1 } ^ { \infty } \sum _ { n = - \infty } ^ { \infty } } & { < f , \psi _ { m , n } > \psi _ { m , n } ( t ) + } \\ { \displaystyle \sum _ { m = - \infty } ^ { m _ { 0 } } \sum _ { n = - \infty } ^ { \infty } } & { < f , \psi _ { m , n } > \psi _ { m , n } ( t ) } \end{array}
$$

其中,

$$
\psi _ { { m , n } } ( t ) = 2 ^ { - \frac { m } { 2 } } \psi \left( \frac { t { - n } 2 ^ { m } } { 2 ^ { m } } \right) = 2 ^ { - \frac { m } { 2 } } \psi ( 2 ^ { - m } t { - n } )
$$

为二进离散小波

基函数 $\boldsymbol { \phi } _ { m , n } ( t )$ 定义为

$$
{ \ s { \theta } _ { m , n } } ( t ) = 2 ^ { - \frac { m } { 2 } } \theta ( 2 ^ { - m } t { - } n )
$$

用基函数 $\boldsymbol { \phi } _ { m , n } ( t )$ 代替式(1)中加号前的部分,则有

$$
f ( t ) = \displaystyle \sum _ { n = - \infty } ^ { \infty } < f , \phi _ { m , n } > \phi _ { m , n } ( t ) + 
$$

令, $C _ { m _ { 0 } , n } = < f , \theta _ { m , n } > , d _ { m , n } = < f , \psi _ { m , n } >$ ,则有

$$
f ( t ) = \sum _ { n = - \infty } ^ { \infty } c _ { m _ { 0 } , n } \phi _ { m , n } ( t ) + \sum _ { m = - \infty } ^ { m _ { 0 } } \sum _ { n = - \infty } ^ { \infty } d _ { m , n } \psi _ { m , n } ( t )
$$

式 中 右边第一部分为 $f ( t )$ 在尺度 $2 ^ { m _ { 0 } }$ 时的低频部分记为 $A _ { { { m } _ { 0 } } }$ ,第二部分为相应的高频部分,记为 $D _ { { \scriptscriptstyle m } }$ ,所以有

$$
{ A _ { m _ { 0 } - 1 } } = { A _ { m _ { 0 } } } + { D _ { m _ { 0 } } }
$$

$$
f ( t ) = A _ { { \scriptscriptstyle m } _ { 0 } } + \sum _ { m = - \infty } ^ { m _ { 0 } } D _ { m } = A _ { { \scriptscriptstyle m } _ { 0 } - 1 } + \sum _ { m = - \infty } ^ { m _ { 0 } - 1 } D _ { m }
$$

由式 可以看出 多分辨分析只对低频部分进行进一步分解 高频部分则没有考虑

# 2.2 长短期记忆(LSTM)神经网络

长短期记忆 神经网络是一种多用于处理可变长序列的神经网络,由 Hochreiter 等[18] 提出,LSTM框架如图1 所示。

![](images/9360399064a2a97eff5a4330d31764fd0dc0a0c3330fcde4a678c4bc8c1b9bf4.jpg)  
图 1 LSTM 框架  
Fig. 1 Framework of LSTM

LSTM 通过 $c _ { t - 1 }$ 到 $c _ { t }$ 的传输带解决了传统循环神经网络 在处理长序列数据时容易产生的梯度消失和爆炸问题 加入门控 遗忘门 输入门 输出门设置来控制序列信息的遗忘与更新。 数据 $\pmb { a } _ { t } ^ { u }$ 输入LSTM 后,首先进入遗忘门与,上一状态信息 $\pmb { u } _ { t - 1 }$ 通过sigmoid 激活函数输出系数矩阵 $f _ { t }$ ,数学表达式为

$$
\boldsymbol { f } _ { t } = \sigma ( \boldsymbol { W } _ { t } \cdot [ \boldsymbol { u } _ { t - 1 } , \boldsymbol { a } _ { t } ^ { u } ] + \boldsymbol { b } _ { f } )
$$

式 中 $W _ { f }$ 和 $b _ { f }$ 分别表示遗忘门的权重参数矩阵和偏置参数 接着 $\pmb { a } _ { t } ^ { u }$ 与 $\pmb { u } _ { t - 1 }$ 进入输入门 通过 激活函数得到输出 $i _ { t }$ ;再将 ${ \pmb u } _ { t - 1 }$ 与 $\pmb { a } _ { t } ^ { u }$ 经过 tanh 激活函数创建一个新值向量 $\pmb { g } _ { t }$ ,利用 $i _ { t }$ 和 $\pmb { g } _ { t }$ 的信息共同控制序列信息的更新 数学表达式为

$$
\boldsymbol { i } _ { t } = \sigma ( \mathbf { \boldsymbol { W } } _ { i } \cdot \left[ \mathbf { \boldsymbol { u } } _ { t - 1 } , \mathbf { \boldsymbol { a } } _ { t } ^ { u } \right] + \boldsymbol { b } _ { i } )
$$

$$
{ \pmb g } _ { t } = \operatorname { t a n h } ( { \pmb W } _ { c } \cdot [ { \pmb u } _ { t - 1 } , { \pmb x } _ { t } ] + { \pmb b } _ { c } )
$$

其中 式 中 $\mathbf { } W _ { i }$ 和式 中 $W _ { c }$ 分别代表 和激活函数层的权重参数矩阵 $\pmb { b } _ { i }$ 和 $\pmb { b } _ { c }$ 分别代表和 激活函数层的偏置参数 此时传输带上的 $\pmb { c } _ { t - 1 }$ 通过遗忘门输出 $f _ { t }$ ,完成对原序列信息选择性的记忆 再加上输入门输出 $i _ { t }$ 与新值向量 ${ \pmb g } _ { t }$ 的乘积得到新的状态信息 $c _ { t }$ 其数学表达式为

$$
\pmb { c } _ { t } = \pmb { f } _ { t } \times \pmb { c } _ { t - 1 } + \pmb { i } _ { t } \times \pmb { g } _ { t }
$$

经过遗忘门和输入门的更新 数据进入输出门$\pmb { u } _ { t - 1 }$ 与 $\pmb { a } _ { t } ^ { u }$ 的信息经过 sigmoid 激活函数,得到输出门输出 $\mathbf { \delta } _ { \pmb { t } }$ 然后将 $\pmb { c } _ { t - 1 }$ 通过 变换与 $\mathbf { \delta } _ { \pmb { t } }$ 结合得到该时刻记忆单元的最终输出 $\pmb { u } _ { t }$ 同时传输到下一步 其数学表达式为

$$
\pmb { \sigma } _ { t } = \sigma ( \mathbf { \boldsymbol { W } } _ { o } \cdot \left[ \mathbf { \boldsymbol { u } } _ { t - 1 } , \pmb { a } _ { t } ^ { u } \right] + \pmb { b } _ { o } )
$$

$$
\pmb { u } _ { t } = \pmb { o } _ { t } \cdot \operatorname { t a n h } ( \pmb { c } _ { t } )
$$

其中 式 中 $W _ { o }$ 代表 激活函数层的权重参数矩阵, $\boldsymbol { b } _ { o }$ 代表 sigmoid 激活函数层的偏置参数。

# 3 基于 LSTM-Wavelet 模型的股价预测

# 3. 1 LSTM-Wavelet 模型构建

基于小波分解和 模型 本节提出如下- 模型应用于股票价格预测 其实现分为 个阶段 第一阶段对股票数据中的开盘价 收盘价 最高价 最低价 交易量 个维度的数据进行小波分解 使得平稳化后的数据更趋于市场原有规律 第二阶段将小波分解后的各层小波系数分别建立 模型进行预测 第三阶段将预测后各层小波系数进行小波重构实现对收盘价的预测 模型框架如图 所示

![](images/f5ba0de3e6340957e19e4a70638ce1e31272c8109fcc7ddcc31fb2e6b99c25a4.jpg)  
图 2 LSTM-Wavelet 模型框架  
Fig. 2 Framework of LSTM-Wavelet model

# 3.2 基于提出模型的股价预测

# 3 2 1 数据来源及描述

本节将使用 模型进行股票价格的预测 选取 $2 0 1 7 - 0 5 - 1 0 - 2 0 2 1 - 1 2 - 1 4$ 中石油共的股票交易数据作为训练集中石油共 的交易数据作为测试集 用以评估模型效果 选取开盘价 最高价 最低价 交易量 个维度的交易数据作为特征输入,收盘价作为标签。 本文基于 语言环境 并以 作为深度学习框架进行训练及预测 数据来自 数据库

# 3 2 2 数据的小波分解

本节主要利用小波分析方法中的多分辨分析对股票原始数据进行小波分解 第一阶段首先确定小波分解层数以及小波基函数 在实际应用中 分解层数越多 越有助于提取交易数据中的深层次信息 但是随着分解层数的增加 数据关键信息的损失也会相应增加数据本身的误差也会增大[19] 因此 小波分解层数一般不会超过 层 本文分解层数选择 层 根据过往的研究经验 小波基函数选取 小波[20] 对中石油股票数据的开盘价 收盘价 最高价 最低价 交易量进行层小波分解 分解结构图如图 所示

![](images/d5ce975259cc8a1939c1254d39580e5ca772eb104663bba2a456b6ec3cecc644.jpg)  
图 3 2 层小波分解结构图  
Fig. 3 Structure diagram of 2-layer wavelet decomposition

由图 可知 对原始数据序列 $f ( t )$ 进行 层小波分解后 $\mathbf { \mathscr { f } } ( t )$ 被分解为

$$
f ( t ) = A _ { 1 } + D _ { 1 } = A _ { 2 } + D _ { 2 } + D _ { 1 }
$$

图 中 $D _ { 1 } \setminus D _ { 2 }$ 分别为第一层 第二层分解得到的高频信号 ${ \bf \nabla } \cdot { \bf A } _ { 2 }$ 为第 层分解得到的低频信号 $A _ { 2 }$ 对应图中的第一层小波系数 $D _ { 2 }$ 对应图 中的第二层小波系数 $D _ { 1 }$ 对应图 中的第三层小波系数 图 显示了中石油股票数据分解后的各层小波系数 其中图图 分别对应开盘价 最高价 最低价 交易量 收盘价分解后的小波系数 每次小波分解会使得低频信号与高频信号平分数据量 因此 原始数据 $f ( t )$ 数据量为 $1 \ 1 2 1 , A _ { 2 }$ 层数据量为 $2 8 2 , D _ { 2 }$ 层数据量为$^ { 2 8 2 , D _ { 1 } }$ 层数据量为557。

![](images/d43bce6bec42bc768fc3e2c8012a1aee5408ac698d2b0c5377617ea909983d41.jpg)

![](images/55a0b567a70322b0a2370d2b273b08377e8106d849d5343fb535617332bfe7f3.jpg)  
图 4 中石油股票数据分解后的各层小波系数  
Fig. 4 Wavelet coefficients of each layer after the decomposition of PetroChina stock data

由图 可以看出 舍弃了高频数据的 $A _ { 2 }$ 层序列更为光滑 平稳 有利于模型的拟合预测 $D _ { 1 } \setminus D _ { 2 }$ 层中虽然包含了许多噪声和突变信息 但也隐藏着一些有用信息 通过对 $D _ { 1 } \setminus D _ { 2 }$ 层进行建模预测可以较大程度地提取其中的有效信息并去除噪声

# 3 2 3 模型的分层预测

为了保证各个序列之间不会受不同量纲的影响同时提升模型预测精度,训练模型之前选择对 $A _ { 2 } \setminus D _ { 1 }$ 、$D _ { 2 }$ 层分别做 MinMaxScaler 归一化处理,公式为

$$
x = \frac { x _ { i } - x _ { \operatorname* { m i n } } } { x _ { \operatorname* { m a x } } - x _ { \operatorname* { m i n } } }
$$

其中 ${ \bf \nabla } _ { x _ { i } }$ 为各层的第 $i$ 个数据值 $x _ { \mathrm { m i n } }$ 为各层的数据最小值, $x _ { \mathrm { m a x } }$ 为各层的数据最大值, $x$ 为各层数据经过归一化处理后位于 的值

设定 输入序列时间步长为 采用 $\operatorname { M i n i - }$ 方法训练 网络 由于 神经网络模块的层数越多 学习能力越强 但是层数过多又会造成网络训练难以收敛[21] 因此训练过程中网络的层数采用2层。 $A _ { 2 } \ l , D _ { 2 }$ 层中的数据量为 $D _ { 1 }$ 层中的一半 因此 $D _ { 1 }$ 层输入节点应多于 $A _ { 2 } \ l , D _ { 2 }$ 层。 设置 $A _ { 2 } \ l , D _ { 2 }$ 层中模型输入层节点数为 隐藏层节点数为 设置 $D _ { 1 }$ 层中模型输入层节点数为 120,隐藏层节点数为 12。 $A _ { 2 } \ l , D _ { 1 }$ 和 $D _ { 2 }$ 层中模型均设置输出层节点数为 损失函数采用 优化器采用 同时 个模型的隐藏层后面都加入了一个 层 使得前向传播时让神经元的激活值以指定的概率停止工作 从而增强模型的泛化性 防止过拟合 由于 $A _ { 2 }$ 层为分解得到的低频信号保留有更多的关键信息 $D _ { 1 } \setminus D _ { 2 }$ 层为分解得到的高频信号 其中噪声更多 因此设置 $A _ { 2 }$ 层模型 值设置 $D _ { 1 } \setminus D _ { 2 }$ 层模型 值 最后将预测后的数据进行反归一化 分层数据 模型股价预测流程如图5 所示。

![](images/dd6e6befcc6417b42cbc25d8ea18458d52f9146f4a94ef6b6d5546176d84dfbe.jpg)  
图 5 分层数据 LSTM 模型预测流程  
Fig. 5 Prediction process of layered data LSTM model

# 3 2 4 预测数据重构

对预测后各个小波层的数据进行小波重构以实现对中石油收盘价的预测 由于 模型中设置时间步长为 即以过去 个交易日数据预测下一个交易日收盘价 导致小波重构之前 $A _ { 2 } \setminus D _ { 1 } \setminus D _ { 2 }$ 各层缺少 组数据 实验选取训练集最后 组数据作为填充 以完成小波重构 重构后数据见表 第 列

# 3 2 5 短期预测结果对比分析

为了说明本文 模型的有效性 从预测性能与预测精度两方面对其比较 选取标准 模型和 - 模型[5]进行对比分析

表 是 个交易日中石油股价预测结果 第列分别是标准 模型 - 模型和本文模型的预测结果 在数据处理过程中- 模型和 - 模型都利用小波对股票历史数据做 层分解 在此基础上模型对分解后的各层小波系数分别建立模型 对各层小波系数进行预测 用得到的预测小波系数重构数据 最后得到预测结果 显然模型的预测结果更接近于真实值

表 1 20 个交易日中石油股价预测结果  
Table 1 Prediction results of PetroChina stock price for 20 trading days   

<table><tr><td>日期</td><td>收盘价</td><td>LSTM</td><td>I ARMA−Wavelet[5]</td><td>LSTM-Wavelet</td></tr><tr><td></td><td>/元</td><td>/元</td><td>/元</td><td>/元</td></tr><tr><td>2021/12/15</td><td>4.83</td><td>4.92</td><td>4.90</td><td>4.88</td></tr><tr><td>2021/12/16</td><td>4.94</td><td>4.94</td><td>5.01</td><td>4.95</td></tr><tr><td>2021/12/17</td><td>4.94</td><td>4.92</td><td>5.02</td><td>4.93</td></tr><tr><td>2021/12/20</td><td>4.85</td><td>4.96</td><td>4.98</td><td>4.90</td></tr><tr><td>2021/12/21</td><td>4.9</td><td>5.02</td><td>5.03</td><td>4.93</td></tr><tr><td>2021/12/22</td><td>4.89</td><td>5.03</td><td>4.99</td><td>4.96</td></tr><tr><td>2021/12/23</td><td>5.03</td><td>5.08</td><td>5.11</td><td>5.02</td></tr><tr><td>2021/12/24</td><td>4.94</td><td>5.09</td><td>5.09</td><td>5.05</td></tr><tr><td>2021/12/27</td><td>4.95</td><td>5.03</td><td>5.06</td><td>5.00</td></tr><tr><td>2021/12/28</td><td>4.93</td><td>5.01</td><td>5.01</td><td>4.98</td></tr><tr><td>2021/12/29</td><td>4.91</td><td>4.99</td><td>5.04</td><td>4.96</td></tr><tr><td>2021/12/30</td><td>4.9</td><td>5.00</td><td>5.07</td><td>4.97</td></tr><tr><td>2021/12/31</td><td>4.91</td><td>5.05</td><td>5.10</td><td>4.96</td></tr><tr><td>2022/1/4</td><td>4.94</td><td>5.08</td><td>5.06</td><td>5.02</td></tr><tr><td>2022/1/5</td><td>4.96</td><td>5.09</td><td>5.11</td><td>5.06</td></tr><tr><td>2022/1/6</td><td>4.98</td><td>5.20</td><td>5.21</td><td>5.10</td></tr><tr><td>2022/1/7</td><td>5.27</td><td>5.22</td><td>5.14</td><td>5.23</td></tr><tr><td>2022/1/10</td><td>5.3</td><td>5.21</td><td>5.21</td><td>5.26</td></tr><tr><td>2022/1/11</td><td>5.24</td><td>5.25</td><td>5.23</td><td>5.30</td></tr><tr><td>2022/1/12</td><td>5.33</td><td>5.30</td><td>5.20</td><td>5.32</td></tr></table>

图 是 个交易日的收盘价预测结果对比 其中黑色实线代表真实收盘价 黑色虚线代表 种模型对中石油收盘价的预测 由图 可知 种模型的预测结果与收盘价的真实趋势都较为接近 但标准 模型和 模型预测值普遍高于真实值 在时间序列预测普遍存在的滞后性方面 通过对比 种模型的预测结果与真实值处于相同趋势的位置 可以发现 模型在同一个趋势的大多数情况下距离真实值最近 可以认为 模型相比于另两个模型减轻了一定的滞后性 与真实值的变化趋势吻合度更高

![](images/03ece1ca6930bb03b6090bd5d056dbf6050f4766a1b1f48063c1058e8cec6a7f.jpg)  
图 6 20 个交易日的收盘价预测结果对比  
Fig. 6 Comparison of the closing price forecasts over 20 trading days

下面通过量化指标说明本文模型的有效性。 选取平均绝对误差( $R _ { \mathrm { { M A E } } }$ )和均方根误差( $R _ { \mathrm { R M S E } }$ )作为评估标准 公式如下

$$
R _ { _ { \mathrm { M A E } } } = \frac { 1 } { n } \sum _ { i \ 1 } ^ { n } \mid \ y _ { i } - \hat { y } _ { i } \mid
$$

$$
R _ { \mathrm { { R M S E } } } = { \sqrt { { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } { ( y _ { i } - { \hat { y } } _ { i } ) ^ { 2 } } } }
$$

其中, $n$ 为验证集序列长度, $y _ { i }$ 为收盘价的真实值, $\hat { \boldsymbol y } _ { i }$ 为收盘价的预测值 $R _ { \mathrm { M A E } }$ 表示预测值和观测值之间绝对误差的平均值 对所有个体差异在平均值上赋予的权重都相等, $R _ { \mathrm { { M A E } } }$ 越小代表精度越高; $R _ { \mathrm { R M S E } }$ 表示预测值和观测值之间差异的样本标准差 侧重描述样本的离散程度对极端的差异惩罚更多 $R _ { \mathrm { R M S E } }$ 越小代表精度越高 表是 个模型 个交易日中石油股价精确度比较

表 2 模型性能比较  
Table 2 Comparison of model performance   

<table><tr><td>模型</td><td>平均绝对误差(RMAE）均方根误差(RRMSE）</td><td></td></tr><tr><td>LSTM</td><td>0.091</td><td>0.106</td></tr><tr><td>ARMA−Wavelet</td><td>0.117</td><td>0.127</td></tr><tr><td>LSTM− W avelet</td><td>0.053</td><td>0.062</td></tr></table>

表 2 的预测结果显示:ARMA-Wavelet 模型的 $R _ { \mathrm { { M A E } } }$ 值和 $R _ { \mathrm { R M S E } }$ 值均为最大,LSTM 模型误差值居中,本文LSTM-Wavelet 模型误差最小，说明ARMA-Wavelet模型预测效果较差 本文 模型预测效果好。 分析其原因,主要是因为 ARIMA 时间序列模型以平稳序列建模为前提假设 而股票数据具有非线性和非平稳性 高噪声等特征 尽管 分解将数据分解成结构更为简单 数据变化趋势更加明显的序列 但是 模型仍然不能很好地预测股票价格 由于 具备良好的细节特征提取能力 本文- 模型将 模型应用于小波系数的分层建模 可根据每层系数特征 再通过数据融合得到总的预测结果 得到更加精确的预测结果 避免了使用原始数据单一建模时容易造成模型过拟合的情况。

# 4 结 论

本文针对股票市场随机变化性强 非线性显著的特点提出了基于深度学习和小波分解的 -预测模型 通过引入小波分解 使 模型充分发挥了自己的优势 对各层小波系数分别建模预测 既能有效刻画数据本身的非线性 又减轻了高噪声因素的影响 实验结果表明 该模型与 模型和模型相比 在预测性能与预测精度上均有提升

# 参考文献( References) :

[ 1 ] 张思奇, 马刚, 冉华. 股票市场风险、收益与市场效率— ARMA-ARCH-M 模型[ J] . 世界经济, 2000( 5) : 19—28. ZHANG Si-qi, MA Gang, RAN Hua. Stock market risk, return and market efficiency: ARMA-ARCH-M model [ J] . World Economy, 2000( 5) : 19—28.   
[ 2 ] BAILLIE R T, BOLLERSLEV T. A multivariate generalized ARCH approach to modeling risk premia in forward foreign exchange rate markets[ J] . Journal of International Money and Finance 1990( 9) : 309—324.   
[ 3 ] 韦艳华, 张世英. 金融市场的相关性分析———Copula-GARCH 模型及其应用[ J] . 系统工程, 2004( 4) : 7—12. WEI Yan-hua, ZHANG Shi-ying. Correlation analysis of financial market—Copula-GARCH model and its application[J]. Systems Engineering, 2004(4): 7—12.   
[ 4 ] 耿晶晶, 刘玉敏, 李洋, 等. 基于 CNN-LSTM 的股票指数预 测模型[ J] . 统计与决策, 2021, 37( 5) : 134—138. GENG Jing-jing, LIU Yu-min, LI Yang, et al. Stock index prediction model based on CNN-LSTM [ J] . Statistics and Decision, 201, 37( 5) : 134—138.   
[ 5 ] 佟伟民, 李一军, 单永正. 基于小波分析的时间序列数据 挖掘[ J] . 计算机工程, 2008( 1) : 26—29. TONG Wei-min, LI Yi-jun, SHAN Yong-zheng. Wavelet analysis based time series data mining [ J] . Computer Engineering, 2008(1): 26—29.   
[ 6 ] 鹿天柱, 钱晓超, 何舒, 等. 一种基于深度学习的时间序 列预测方法[ J] . 控制与决策, 2021, 36( 3) : 645—652. LU Tian-zhu, QIAN Xiao-chao, HE Shu, et al. A time series prediction method based on deep learning [ J] . Control and Decision, 2021, 36( 3) : 645—652.   
[ 7 ] 陈 亚 男, 薛 雷. 基 于 Bagging-SVM 的 股 票 趋 势 预 测 技 术[ J] . 电子测量技术, 2019, 42( 14) : 58—62. CHEN Ya-nan, XUE Lei. Stock trend prediction technology based on bagging SVM [ J] . Electronic Measurement Technology, 2019, 42 ( 14) : 58—62.   
[ 8 ] 刘玉敏, 李洋, 赵哲耘. 基于特征选择的 RF-LSTM 模型成 分股 价 格 趋 势 预 测 [ J ] . 统 计 与 决 策, 2021, 37 ( 1 ) : 157—160. LIU Yu-min, LI Yang, ZHAO Zhe-yun. Prediction of component stock price trend based on feature selection in RF-LSTM model [ J ] . Statistics and Decision, 201, 37 ( 1 ) : 157—160.   
[ 9 ] GHODDUSI H, CREAMER, GERMÁN G, et al. Machine learning in energy economics and finance: a review[ J] . Social Science Electronic Publishing, 2018, 81 ( 1 ) : 709— 727.   
[ 10] 綦方中, 林少倩, 俞婷婷. 基于 PCA 和 IFOA-BP 神经网络 的股价预测模型 [ J] . 计算机应用与软件, 2020, 37 ( 1 ) : 116—121,156. QI Fang-zhong, LIN Shao-qian, YU Ting-ting. Stock price prediction model based on PCA and IFOA-BP neural network[ J] . Computer Applications and Software, 2020, 37 (1) : 116—121, 156.   
[11] 陈祥一. 基于卷积神经网络的沪深 300 指数预测[ D] . 北 京: 北京邮电大学, 2018. CHEN Xiang-yi. Prediction of CSI 300 index based on convolutional neural network [ D ] . Beijing: Beijing University of Posts and Telecommunications, 2018.   
[12] 杨青, 王晨蔚. 基于深度学习 LSTM 神经网络的全球股票 指数预测研究[ J] . 统计研究, 2019, 36( 3) : 65—77. YANG Qing, WANG Chen-wei. Global stock index prediction based on deep learning LSTM neural network [ J] . Statistical Research, 2019, 36( 3) : 65—77.   
[ 13 ] KIM H Y, WON C H. Forecasting the volatility of stock price index: a hybrid model integrating LSTM with multiple GARCH-type models [ J] . Expert Systems with Applications, 2018, 103( 8) : 25—37.   
[ 14] CHEN K, ZHOU Y F, DAI A. LSTM-based method for stock returns prediction: a case study of China stock market [ C ] / / IEEE International Conference on Big Data. CA, USA: Santa Clara, 2015( 1) : 2823—2824.   
[ 15] GUOKUN L, CHANG W, YANG Y, et al. Modeling long-and short-term temporal patterns with deep neural networks [ C ] / / Proceedings of the 41th International ACM SIGIR Conference on Research & Development in Information Retrieval. New York: ACM, 2018( 1) : 95—104.   
[ 16] MORLET J, ARENS G, FORUTEAU E, et al. Wave propagation and samping theory and complex waves[ J] . Geophysics, 1982, 47( 2) : 222—236.   
[ 17] MALLET S G. A theory for multiresolution signal decomposition wavelet representation [ J] . IEEE Trans on PAMI, 1989, 11( 7) : 674—693.   
[ 18] HOCHREITER S, SCHMIDHUBER J. Long short-term memory[ J] . Neural Computation, 1997( 8) : 1735—1780.   
[ 19] 汤劼. 基于小波分析的金融时间序列研究与应用[ D] . 合 肥 中国科学技术大学 TANG Jie. Research and application of financial time series based on wavelet analysis[ D] . Hefei: University of Science and Technology of China, 2011.   
[ 20] 凡婷. 基于小波分析理论的 GARCH 模型在金融时间序列 中的应用研究[D]. 南充: 西华师范大学, 2018. FAN Ting. Research on the application of GARCH model based on wavelet analysis theory in financial time series [ D ] . Nanchong: China West Normal University, 2018.   
[21] 周颖, 沐年国. 基于神经网络研究人民币汇率对企债收益 的影响[ J] . 重庆工商大学学报( 自然科学版) , 2020, 37( 1) : 71—77. ZHOU Ying, MU Nian-guo. Research on the impact of RMB exchange rate on corporate bond income based on neural network [ J] . Journal of Chongqing Technology and Business University ( Natural Science Edition) , 2020, 37 ( 1) : 71—77.