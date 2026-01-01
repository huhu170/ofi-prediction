# 基于K-means-LSTM 模型的证券股价预测

肖田田

（安徽建筑大学数理学院，合肥230601)

摘要：鉴于股票数据具有非平稳、非线性等特征，传统的统计模型无法精准预测股票价格的未来趋势。针对这个问题，构建一种混合深度学习方法来提高股票预测性能。首先，通过将距离算法修改为DTW(动态时间归整)，令$K$ -means聚类算法拓展为更适用于时间序列数据的 $K$ means-DTW，聚类出价格趋势相似的证券；然后，通过聚类数据来训练LSTM(长短时记忆网络)模型，以实现对单支股票价格的预测。实验结果表明，混合模型 $K$ -means-LSTM表现出更好的预测性能，其预测精度和稳定性均优于单一LSTM模型。

关键词：股票价格预测； $K$ means；DTW（动态时间归整）； $K$ means-LSTM( $K$ 均值-长短时记忆网络)混合模型中图分类号：F830 文献标志码：A 文章编号： $1 6 7 1 - 1 8 0 7 ( 2 0 2 4 ) 0 3 - 0 2 1 0 - 0 6$

随着我国经济的蓬勃发展，越来越多的人积极融入金融投资领域，将其视为一种重要的理财手段。股票因其流动性强、投资收益高，一直是广大投资者青睐的投资方式。然而，股票数据表现出非平稳、非线性和高噪声等特点，同时受到多种因素综合影响。因此，建立一个科学合理且实用的股票价格预测模型变得尤为具有挑战性[1]。

传统的统计学模型，如自回归移动平均（autoregressive and moving average,ARMA)、自回归求和移动平均（autoregressive integrated moving average，ARIMA)已经被广泛应用于经济、社会等各个领域。吴玉霞和温欣[2]采用ARIMA模型对股票价格进行短期预测，取得了不错的效果。尽管传统统计模型在时间序列数据的预测中表现出一定的能力，但是也存在明显的局限性，这些模型本质上只能捕捉线性关系，无法捕捉非线性关系。这意味着它们在处理包含复杂非线性因素的数据时可能无法提供准确的预测。随着人工智能的快速发展，机器学习算法凭借其强大的特征提取和学习能力已广泛应用于自然语言处理、推荐系统、数据挖掘等领域[3]。邓烜堃等[4]采用BP(backpropagation)神经网络用于股票价格预测；邬春学和赖靖文[5]利用支持向量机预测股票价格。还有多种模型混合的方法，如崔文喆等[6]提出基于广义自回归条件异方差（generalized autoregressive conditional heteroskedasticity，GARCH）和BP神经网络模型的股票市场价格预测。尽管机器学习算法在处理非线性数据方面表现出显著的优势，但由于股票数据具有时序相关性，一般的机器学习算法在特征提取方面仍然存在一定的局限性。

随着大数据分析时代的来临，深度学习技术凭借其卓越的学习和泛化能力，在数据预测方面展现出超越传统机器学习的强大潜力[7]。循环神经网络（recurrentneuralnetwork，RNN）能够捕捉和记忆序列中的先前信息，适用于处理时间序列数据。然而，当处理长序列时，RNN面临着梯度消失问题。为了解决这一挑战，长短时记忆网络（longshort-term memory，LSTM）作为RNN的变种，通过引入门控机制，能够有效控制和更新信息流，更好地处理长期依赖关系。近年来，LSTM模型及其相关的混合和改进版本在多个领域广泛应用，用于解决分类和预测问题。LSTM模型为处理复杂、具有时序性的数据提供了更准确和可靠的解决方案。

以往的研究通常都是集中于单一股票的历史数据和相关特征，却忽略了不同股票之间的关联性。同一类股票之间往往具有相似的价格波动趋势，故提出了一种聚类数据的长短期记忆神经网络混合模型。首先，根据历史收盘价对15支证券股票进行聚类，聚类结果能够有效表达股票价格的历史相关性和趋势相关性；随后，利用聚类数据训练LSTM模型并进行预测；最后，将其结果与单一模型LSTM进行对比。这一研究方法不仅考虑股票之间的相关性，还能捕捉相关股票之间的内在联系，有望提供更准确的股票价格预测，有助于投资者做出更明智的决策。

# 1研究方法

# 1.1欧式距离

$K$ -means算法是一种常用的聚类分析方法，其默认的距离计算公式为欧式距离。对两个时间序列 $\pmb { x } = ( x _ { 1 } , x _ { 2 } , \pmb { \cdots } , x _ { n } ) , \pmb { y } = ( y _ { 1 } , y _ { 2 } , \pmb { \cdots } , y _ { n } )$ ，计算欧式距离为公式为

$$
d ( x , y ) = { \sqrt { \sum _ { i = 1 } ^ { n } ( x _ { i } - y _ { i } ) ^ { 2 } } }
$$

然而欧式距离在计算两个时间序列的距离时，存在一些限制。当两个时间序列的长度不等时，较长的一个时间序列会剩下无法匹配的点，此时欧式距离计算公式不再适用。此外，两个序列时间步对不齐，但总体的趋势是相似时，显然欧式距离按照点与点之间对应关系的计算方式无法满足研究需求。但动态时间归整（dynamic time warping，DTW)算法可以通过计算所有点之间的最小距离，实现一对多的匹配，解决时间轴上的失真[8]。

# 1.2动态时间规划(DTW)

设 $\pmb { x } = ( x _ { 1 } , x _ { 2 } , \pmb { \cdots } , x _ { n } ) , \pmb { y } = ( y _ { 1 } , y _ { 2 } , \pmb { \cdots } , y _ { m } )$ 是长度分别为 $n$ 和 $m$ 的两个时间序列，为了对齐这两个序列，构建一个距离矩阵 $M ^ { \times m }$ ，矩阵元素 $( i , j )$ 表明 $x$ 中第 $i$ 个点 $\boldsymbol { \mathscr { x } } _ { i }$ 与 $\textbf { y }$ 中第 $j$ 个点 $y _ { j }$ 的距离， $x _ { i }$ 和$y _ { j }$ 两点间的距离定义为欧氏距离 $w _ { s } = ( x _ { i } - y _ { j } ) ^ { 2 }$ ,在矩阵中找到一条通过若干格点的路径，路径通过的格点即为两个序列进行计算的对齐的点。DTW算法就是找到一条最优路径，使得路径上所有匹配点对的距离和最小，该最小距离对应DTW距离为

$$
\mathrm { D T W } ( x , y ) = \mathrm { m i n } \sqrt { \sum _ { s = 1 } ^ { k } w _ { s } }
$$

矩阵 $M$ 上的最优路径可以使用如下递归函数计算：

$$
\gamma ( i , j ) = d ( x _ { i } , y _ { j } ) +
$$

$$
\operatorname* { m i n } \{ \gamma ( i - 1 , j - 1 ) , \gamma ( i - 1 , j ) , \gamma ( i , j - 1 ) \}
$$

# 1.3 $\pmb { K }$ means-DTW算法

将距离算法修改为DTW后， $K$ -means的目标函数 $E$ 如式(4)所示。给定数据集 $D = \{ x _ { 1 } , x _ { 2 } , \cdots ,$ ,$\boldsymbol { x } _ { m } \mathrm { ~ \rangle ~ }$ ，将其聚为 $k$ 个簇（ $k$ 根据需要设定）； $C = \left\{ c _ { 1 } \right.$ ,$c _ { 2 } \ldots , \ldots , c _ { m } \} , C$ 为 $k$ 个簇的中心。过程是先计算每个序列到当前对应簇的中心序列的DTW距离的累积。

$$
E = \sum _ { i = 1 } ^ { k } \sum _ { x \in c _ { i } } \mathrm { D T W } ( x , \mu _ { i } )
$$

式中： $\mu _ { i } = { \frac { 1 } { \left| c _ { i } \right| } } \sum _ { x \in c _ { i } } x$ 为 $c _ { i }$ 的中心。

具体步骤： $\textcircled{1}$ 随机选择选择 $k$ 个样本作为初始聚类中心 $\mu _ { i }$ ; $\textcircled{2}$ 计算其余样本到聚类中心 $\mu _ { i }$ 的DTW，将样本划到距离最近的聚类中心所属的类别中； $\textcircled{3}$ 重新计算每个簇的新中心 $\mu _ { i }$ ; $\textcircled{4}$ 重复 $\textcircled{2}$ (cid:) $\textcircled{3}$ 步骤直到每个簇的中心不再变。

# 1.4长短期记忆网络(LSTM)

LSTM是由 Hochreiter和 Schmidhuber[9]首次提出的一种神经网络模型。LSTM模型通过引入门控机制，确保梯度能长期维持传递，有效缓解了标准RNN中梯度消失、梯度爆炸问题，近年来，被广泛应用于时间序列数据的处理与分析。LSTM模型存储单元由3部分组成，分别是遗忘门、输入门和输出门，如图1所示。

![](images/5d0e332ebd6431fab676f0d7aad1662fbdc8a9d61703df73375274beda8a5e6d.jpg)  
图1LSTM模型结构

$C _ { t - 1 }$ 为上一个单元状态； $C _ { t }$ 为当前单元状态； $\widetilde { C } _ { t }$ 为临时单元状态； $\boldsymbol { \mathcal { X } } _ { t }$ 为当前时刻的输入值； $h _ { t - 1 }$ 为上一时刻的输出值； $h _ { t }$ 为当前时刻的输出值；sig为sigmoid 函数；tanh为tanh函数； $f _ { t }$ 为遗忘门的输出值； $i _ { t }$ 为输入门的输出值； $O _ { t }$ 为输出门的输出值；$\otimes$ 为乘法运算符； $\textcircled{+}$ 为加法运算符

LSTM模型的计算过程如下。

将上一时刻的输出值和当前时刻的输入值输入到遗忘门中，经过计算得到遗忘门的输出值：

$$
f _ { t } = \sigma ( W _ { f } [ h _ { t - 1 } , x _ { t } ] + b _ { f } ) f _ { t } = \sigma
$$

式中： $\boldsymbol { f } _ { t }$ 的取值范围为 $( 0 , 1 ) \colon W _ { f }$ 为遗忘门的权重；  
$b _ { f }$ 为遗忘门的偏置； $\boldsymbol { \mathcal { X } } _ { t }$ 为当前时刻的输入值； $h _ { t - 1 }$   
为上一时刻的单元输出； $\sigma$ 为sigmoid 函数。

将上一次的输出值和当前的输入值输入到输入门，经过计算得到输出值和输入门的候选细胞状态：

$$
i _ { t } = \sigma ( W _ { i } [ h _ { t - 1 } , x _ { t } ] + b _ { i } )
$$

$$
\widetilde { C } _ { t } = \operatorname { t a n h } ( W _ { c } [ h _ { t - 1 } , x _ { t } ] + b _ { c } )
$$

式中： $i _ { t }$ 取值范围为 $( 0 , 1 ) \colon W _ { i }$ 为输入门的权重； $b _ { i }$ 为输入门的偏置； $W _ { C }$ 为候选输入门的权重； $b _ { C }$ 为候选输入的偏置门。

更新当前单元状态：

$$
C _ { t } = f _ { t } C _ { t - 1 } + i _ { t } \widetilde { C } _ { t }
$$

式中： $C _ { t }$ 取值范围为(0，1)。

在 $t$ 时刻接收输出值 $h _ { t - 1 }$ 和输入值 $\boldsymbol { \mathcal { X } } _ { t }$ 作为输出门的输入，得到输出门的输出 $O _ { t }$ :

$$
O _ { t } = \sigma ( W _ { o } [ h _ { t - 1 } , x _ { t } ] + b _ { O } )
$$

式中： $O _ { t }$ 取值范围为 $( 0 , 1 ) \colon W _ { o }$ 为输出门的权重； $b _ { o }$   
为输出门的偏置。

通过计算输出门的输出和细胞状态得到LSTM的输出值：

$$
h _ { t } = O _ { t } { \mathrm { t a n h } } C _ { t }
$$

# 2数据描述

# 2.1数据

以中国主要的上市证券公司为研究对象，涵盖了共计15支不同的证券公司。这些公司包括海通证券（HT）、东北证券（DB)、西南证券（XN）、太平洋证券（TPY）、长江证券（CJ）、国金证券（GJ）、国元证券（GY）、东吴证券（DW）、兴业证券（XY）、广发证券（GF）、光大证券（GD)、山西证券（SX）、吉林敖东证券(JLAD)、中信证券（ZX)和招商证券（ZS）。以天为单位收集信息，研究数据的时间跨度为2011年12月12日至2021年12月17日，总计涵盖了2346个交易日的数据。实验的数据集均来自国泰安（CSMAR)数据库。

考虑不同股票的价格波动区间存在差异，所以在进行聚类和股票价格预测之前需要消除量纲不同带来的影响。常见的无量纲化处理方法有标准化和归一化，本文在聚类之前对数据进行标准化处理，公式为

$$
X ^ { \ast } = \frac { X - \mu } { S } 
$$

式中： $X$ 为样本数据； $\mu$ 和 $S$ 分别为样本数据的均值和标准差； $X ^ { \ast }$ 为标准化后的数据。

通过LSTM模型进行股票价格预测时，为了方便计算，对每个证券的收盘价进行归一化处理，使最终的输出值在0和1之间，计算公式为

$$
X ^ { \star } = \frac { X - X _ { \operatorname* { m i n } } } { X _ { \operatorname* { m a x } } - X _ { \operatorname* { m i n } } }
$$

式中： $X$ 为样本数据； $X _ { \operatorname* { m i n } }$ 为样本数据中的最小值；  
$X _ { \mathrm { m a x } }$ 为样本数据中的最大值； $X ^ { \ast }$ 为归一化后的数据。

# 2.2相关性分析

计算15支证券的相关系数，如图2所示。太平洋证券和西南证券的相关系数最高，为0.95；东北证券和中信证券的相关系数最低，为一0.09。总体而言，不同证券公司之间的两两相关系数是不规则的，不存在一定的正负相关规律。其中，中信证券与绝大多数证券的相关性皆较低。

# 3实验结果

实验环境基于Python3.8，并使用Keras神经网络库实现神经网络模型。LSTM模型的层数为2，每层的维度均为64。引入Relu激活函数增强神经网络模型的非线性。使用均方误差作为损失函数。为了防止过拟合，并提高模型的泛化能力，使用Dropout正则化，其中概率设置在区间[0，1）。在训练LSTM模型时，采用随机梯度下降法，批量大小设置为32，迭代次数为100，输入数据的时间步长为10。优化器选择了Adam，学习率被设置为0.001。

![](images/cb66bc2a79ad5c7a9f89b276912d2edd61ad2b738ed9264698bf9f29148d10b9.jpg)  
图2相关系数图

# 3.1聚类分析

对15支证券股票的收盘价进行标准化处理后，将 $K$ -means模型的距离算法修改为DTW后进行聚类，结果见表1。其中， $K$ 代表聚类簇的数量，在实验中分别设置了 $2 \sim 5$ 个不同的聚类数目，以观察在不同簇数下的聚类情况。

表1显示，国元证券、东吴证券、山西证券和吉林敖东证券这4支的股票一直被聚在同一类别中，即国元证券、东吴证券、山西证券和吉林敖东证券之间的相似性较高，这表明它们的股价波动趋势存在显著的相似性。这种相似性意味着它们之间可能存在一些共同的市场因素或者投资者行为，导致它们的股价表现相近。这4支的股票价格走势也极为相似，如图3所示。此外，在 $K = 3 \sim 5$ 时的聚类结果中，中信证券已经成为一个独立的类别，再次验证了中信证券与其他多数证券之间的低相关性。

表1股票价格聚类结果  

<table><tr><td rowspan=1 colspan=1>聚类簇数K</td><td rowspan=1 colspan=2>证券</td></tr><tr><td rowspan=2 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>海通证券、东北证券、西南证券、太平洋证券、长江证券、国金证券、国元证券、东吴证券、兴业证券、广发证券、光大证券、山西证券、吉林敖东证券</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>中信证券、招商证券</td></tr><tr><td rowspan=3 colspan=1>3</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>东北证券、西南证券、太平洋证券、长江证券、国金证券、兴业证券</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>海通证券、国元证券、东吴证券、广发证券、光大证券、山西证券、招商证券、吉林敖东证券</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>中信证券</td></tr><tr><td rowspan=4 colspan=1>4</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>东北证券、西南证券、太平洋证券、长江证券、国金证券、兴业证券</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>海通证券、广发证券、光大证券、招商证券</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>中信证券</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>国元证券、东吴证券、山西证券、吉林敖东证券</td></tr><tr><td rowspan=5 colspan=1>5</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>西南证券、太平洋证券、长江证券、国金证券</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>海通证券、光大证券、招商证券</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>中信证券</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>国元证券、东吴证券、山西证券、吉林敖东证券</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>东北证券、兴业证券、广发证券</td></tr></table>

选取来自不同集群的5支股票公司，分别是太平洋证券、广大证券、中信证券、吉林敖东证券和广发证券，以进一步分析。其股价走势如图4所示，与图3所示的趋势相比，这5支证券的趋势呈现出显著的差异。该结果证明了 $K$ means-DTW聚类算法的有效性，进一步说明DTW方法在计算时间序列相似性方面的重要性。DTW方法能够有效考虑时间序列上的畸变，有助于识别和分析不同证券之间的相似性和差异。

选择聚类后的4支证券进行股价预测，将数据集划分为训练集和测试集，使用前 $80 \%$ 作为训练集，后 $20 \%$ 作为测试集，训练集用于训练LSTM模型并调整模型参数，测试集用于模型性能测试。证券股价预测流程如图5所示。

实验选用3个评价指标衡量预测效果，分别为均方误差(MSE)、平均绝对误差(MAE)以及决定系

![](images/355f223fdd8b4bd5b59819fd75ed57a2c6ec4933098e6acb4e055a96084bb884.jpg)  
图32012—2022 年同一簇的证券股价趋势

![](images/d44141f7811a557041af2da8d806c2687b184d4ebc2d9306133af257f0595c95.jpg)  
图42012—2022 年不同簇的证券股价趋势

![](images/6ef2776825adbd0d3e7fb0333cc40f474d1ead8b9807a7f926b5c7044ea37345.jpg)  
图5预测流程

数 $( R ^ { 2 }$ )，具体计算公式为

$$
\begin{array} { l } { { \displaystyle \mathrm { M S E } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } { ( \hat { y } _ { 1 } - y _ { i } ) ^ { 2 } } \ ~ } } \\ { { \displaystyle \mathrm { M A E } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } { \left| \hat { y } _ { 1 } - y _ { i } \right| } } } \\ { { \displaystyle R ^ { 2 } = 1 - \frac { \sum _ { i = 1 } ^ { N } { ( y _ { i } - \hat { y } _ { i } ) ^ { 2 } } } { \sum _ { i = 1 } ^ { N } { ( y _ { i } - \overline { { y } } _ { i } ) ^ { 2 } } } } } \end{array}
$$

式中： $N$ 为样本总数； $y _ { i }$ 为测试集的实际值； $\hat { y } _ { l }$ 为测试集的预测值； $\overline { { y } } _ { l }$ 为测试集的真实值的平均值。

# 3.2预测分析和性能比较

通过单一模型和混合模型对股票价格进行预测。其中单一模型是指将每支股票的历史数据独立作为输入，然后使用LSTM模型进行预测。这意味着每支股票都有其独立的预测模型，不考虑其与其他股票之间的相关性。而混合模型即 $K$ -meansLSTM，将经过聚类后的多支股票的数据一起输入到LSTM模型中进行预测。这意味着多支股票共享一个模型，模型能够有效考虑不同股票之间的相关性，从而更全面地捕捉市场的复杂性。最后，比较单一模型和混合模型的预测结果。

由表2可知，混合模型在不同证券的预测中展现出较为出色的综合表现。例如，在预测国元证券时，混合模型的 $R ^ { 2 }$ 高于单一模型，MSE和MAE低于单一模型。这意味着混合模型在一定程度上能更好地解释股价变动的方差，表现出更好的拟合效果，更接近实际股价趋势。混合模型在考虑不同股票之间相关性的情况下，具有更强的预测能力。

然而，混合模型在吉林敖东证券的预测中表现不如单一模型，这可能是因为吉林敖东证券受其他3支证券的影响较小，混合模型在这种情况下无法充分发挥其优势。这一观察结果强调了模型选择的重要性，应根据不同证券的特性来决定使用单一模型还是混合模型，以获得更准确的预测结果。

表2混合模型和单一模型评价指标  

<table><tr><td rowspan=1 colspan=1>模型</td><td rowspan=1 colspan=1>证券</td><td rowspan=1 colspan=1>MSE</td><td rowspan=1 colspan=1>MAE</td><td rowspan=1 colspan=1>$R^2}$</td></tr><tr><td rowspan=4 colspan=1>混合模型</td><td rowspan=1 colspan=1>东吴证券</td><td rowspan=1 colspan=1>0.057 22</td><td rowspan=1 colspan=1>0.155 88</td><td rowspan=1 colspan=1>0.93434</td></tr><tr><td rowspan=1 colspan=1>国元证券</td><td rowspan=1 colspan=1>0.047 44</td><td rowspan=1 colspan=1>0.13439</td><td rowspan=1 colspan=1>0.95875</td></tr><tr><td rowspan=1 colspan=1>山西证券</td><td rowspan=1 colspan=1>0.02334</td><td rowspan=1 colspan=1>0.09677</td><td rowspan=1 colspan=1>0.95379</td></tr><tr><td rowspan=1 colspan=1>吉林敖东证券</td><td rowspan=1 colspan=1>0.079 64</td><td rowspan=1 colspan=1>0.187 19</td><td rowspan=1 colspan=1>0.89252</td></tr><tr><td rowspan=4 colspan=1>单一模型</td><td rowspan=1 colspan=1>东吴证券</td><td rowspan=1 colspan=1>0.05942</td><td rowspan=1 colspan=1>0.15848</td><td rowspan=1 colspan=1>0.92840</td></tr><tr><td rowspan=1 colspan=1>国元证券</td><td rowspan=1 colspan=1>0.067 84</td><td rowspan=1 colspan=1>0.166 98</td><td rowspan=1 colspan=1>0.940 67</td></tr><tr><td rowspan=1 colspan=1>山西证券</td><td rowspan=1 colspan=1>0.02818</td><td rowspan=1 colspan=1>0.11051</td><td rowspan=1 colspan=1>0.946 78</td></tr><tr><td rowspan=1 colspan=1>吉林敖东证券</td><td rowspan=1 colspan=1>0.077 63</td><td rowspan=1 colspan=1>0.18339</td><td rowspan=1 colspan=1>0.89497</td></tr></table>

综合而言，混合模型在使用相似证券股价数据进行训练时通常表现优于在单一证券数据上进行训练的模型，同时也说明了聚类算法的有效性。混合模型的优势在于它能够更好地分析同类证券之间的相互影响，从而更准确地捕捉市场的复杂性。即通过考虑相似性高的证券，提高预测的准确性。

# 4结论

为了提高股票预测的准确性，提出了一种混合深度学习模型 $K$ means-LSTM。该模型基于证券股价之间的相关性，运用 $K$ -means-DTW聚类算法，筛选出具有相似价格波动的股票，接着利用聚类数据训练LSTM模型并进行预测。选取15支证券的股票数据进行实验，结果表明混合模型 $K$ meansLSTM表现出更好的预测性能，其预测精度和稳定性均优于单一模型LSTM。混合模型综合考虑了具有相似股票价格趋势的公司之间的关联性，不仅弥补了仅使用历史数据的不足，还为股票价格预测提供了更为全面的数据基础，由于其更高的预测精度和良好的泛化能力，该模型在金融时间序列分析中表现出科学性和可行性，对于构建大数据和人工智能背景下的金融时间序列数据预测模型具有参考价值，有望为投资者提供更可靠的决策支持，促进投资组合模型的优化建立。

然而， $K$ means-LSTM模型只考虑股票价格历史数据和相关企业的影响，但是影响股票价格的因素是多方面的，如新闻，市场情绪等。因此，后续的研究应该更全面地考虑外部因素，以更准确地分析和预测股票价格，以便投资者能够更好地规划其投资策略并降低风险。

# 参考文献

[1] GANDHMAL D P, KUMAR K. Systematic analysis andreview of stock market prediction techniques[J]. Compu-ter Science Review, 2019, 34: 100190.  
[2]吴玉霞，温欣．基于ARIMA模型的短期股票价格预测[J]．统计与决策，2016，32(23)：83-86.  
[3] 何清，李宁，罗文娟，等．大数据下的机器学习算法综述[J]．模式识别与人工智能，2014，27(4)：327-336.  
[4] 邓烜堃，万良，黄娜娜．基于DAE-BP神经网络的股票预测研究[J.计算机工程与应用，2019，55(3)：126-132.  
[5] 邬春学，赖靖文．基于SVM及股价趋势的股票预测方法研究[J].软件导刊，2018，17(4)：42-44.  
[6 崔文喆，李宝毅，于德胜．基于GARCH模型和BP神经网络模型的股票价格预测实证分析[J．天津师范大学学报（自然科学版），2019，39(5）：30-34.  
[7]胡越，罗东阳，花奎，等．关于深度学习的综述与讨论

[J].智能系统学报，2019，14(1)：1-19.[8] BERNDT D J, CLIFFORD J. Using dynamic time war-ping to find patterns in time series: WS-94-03[R]. Palo

# Securities Stock Price Prediction Based on $\pmb { K }$ -means-LSTM Model

XIAO Tiantian

(School of Mathematics and Science, Anhui Jianzhu University, Hefei 230601, China)

Ast the distance algorithm is modified to DTW (dynamic time warping) by expanding the $K$ —means clustering algorithm to $K$ -means-DTW,which is ui es through clustering data to predict the price of a single stock. Experimental results show that the hybrid model $K$ means-LSTM shows better prediction performance and its prediction accuracy and stability are better than the single LSTM model.

Keywords: stock price prediction; $K$ means;DTW;K-means-LSTM( $K$ means-long short-term memory) hybrid model