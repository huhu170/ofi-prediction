# 基于LSTM-CNN-CBAM模型的股票预测研究

赵红蕊，薛 雷

上海大学 通信与信息工程学院，上海 200444

摘 要：为了更好地对股票价格进行预测，进而为股民提供合理化的建议，提出了一种在结合长短期记忆网络（LSTM）和卷积神经网络（CNN）的基础上引入注意力机制的股票预测混合模型（LSTM-CNN-CBAM），该模型采用的是端到端的网络结构，使用LSTM来提取数据中的时序特征，利用CNN挖掘数据中的深层特征，通过在网络结构中加入注意力机制——Convolutional Attention Block Module（CBAM）卷积模块，可以有效地提升网络的特征提取能力。基于上证指数进行对比实验，通过对比实验预测结果和评价指标，验证了在LSTM与CNN结合的网络模型中加入CBAM模块的预测有效性和可行性。

关键词：长短期记忆网络（LSTM）；卷积神经网络（CNN）；注意力机制；股价预测文献标志码：A 中图分类号：TP29 doi：10.3778/j.issn.1002-8331.1912-0448

# Research on Stock Forecasting Based on LSTM-CNN-CBAM Model

ZHAO Hongrui, XUE Lei

College of Communication and Information Engineering, Shanghai University, Shanghai 200444, China

Abstract：In order to better predict stock prices and provide reasonable suggestions for stockholders, a hybrid stock prediction model（LSTM-CNN-CBAM）that incorporates attention mechanism based on Long Short and Term Memory （LSTM）network and Convolutional Neural Network（CNN）is proposed. The model uses an end-to-end network structure. LSTM is used to extract the time-series features in the data, and then CNN is used to mine the deep features in the data. By adding an attention mechanism to the network structure Convolutional Attention Block Module convolution module, which can effectively improve the feature extraction capability of the network. Based on the Shanghai Stock Exchange Index, a comparative experiment is performed. By comparing the experimental prediction results and evaluation indicators, the prediction effectiveness and feasibility of adding the CBAM module to the network model combining LSTM and CNN are verified.

Key words：Long Short and Term Memory（LSTM）network; Convolutional Neural Network（CNN）;attention mechanism; stock price forecasting

随着计算机科学和市场经济的快速发展，股票市场作为资本市场的重要组成部分，成为了政府、上市公司、投资机构以及一些个人投资者的关注热点。股票市场是国民经济发展变化的“晴雨表”和“报警器”，其行情的变化与国家的宏观经济发展、法律法规的制定、政治事件的发生、公司的财务状况和政策、投资者心理、舆论引导等等都有所关联，从而导致股票价格具有高度的波动性与不规律性，因此合理准确地预测股票价格的变化趋势成为许多业界学者的主要探索和研究方向。

随着大数据时代的发展，像支持向量机[1]、决策树[2]、随机森林[3]以及深度学习算法[4]等机器学习算法模型被广泛应用于股票等金融数据研究。由于股票数据具有数据量大，信息模糊，长记忆性，非线性和非平稳性等特征，传统的机器学习算法并不能取得较好的预测效果，而深度学习模型与传统的机器学习模型相比有着更为强大的学习能力和自适应能力[5]，对非线性系统可以更好地进行预测分析。

LSTM[6]神经网络作为深度学习算法模型中的一种新型递归神经网络模型受到了广泛的关注，由于其具备良好的选择性、记忆性以及时序内部影响的特性极适用于股票价格时间序列的预测，因此具有广阔的应用前景。

![](images/c774af410c477803290b54c06f513a7e65aa9c2e0c82c5ce193d644e3ef8062b.jpg)  
图1 LSTM网络结构展开图

本文基于深度学习方法对股票金融数据进行研究，以收盘价格为预测目标，提出了一种结合LSTM和CNN的时间序列预测模型，该模型主要采用端到端的网络结构，首先使用LSTM来提取数据中的特征，尤其是时间序列中的时序特征，然后利用CNN挖掘时间序列中的局部特征和深层特征[7]。同时结合了注意力机制[8]CBAM[9]，相比于SENet[10]只关注通道的注意力机制可以更好地提升网络模型的特征提取能力。

本实验中使用PyTorch作为神经网络的框架，使用Python语言进行了网络的代码实现。使用上海证券综合指数（简称上证指数，证券代码为000001）1991—2018年股票数据进行分析预测实验，将真实值和预测值进行对比，并且进行预测结果图形拟合[11]和误差评估，通过与LSTM和LSTM-CNN模型的对比实验，最后验证了在LSTM与CNN结合的网络模型中加入CBAM模块预测模型的有效性。

# 1 LSTM的结构和原理介绍

LSTM是为了解决循环神经网络[12（] RNN）模型由于输入序列过长而产生的梯度消失[13]问题而发展出来的一种机器学习神经网络，主要由记忆细胞、输入门、输出门、遗忘门组成，三个门的激活函数均为Sigmoid。输入门用来控制当前时刻神经单元的输入信息，遗忘门用来控制上一时刻神经单元中存储的历史信息，输出门用来控制当前时刻神经单元的输出信息[14]。

图1为LSTM的网络结构展开图，其中 $X _ { t }$ 表示当前$t$ 时刻的输入， $h _ { t }$ 表示当前 $t$ 时刻细胞的状态值，下面是LSTM的计算公式：

$$
\begin{array} { r l } & { i _ { t } = \sigma \big ( W _ { i } \^ { * } \big [ h _ { t - 1 } , X _ { t } \big ] + b _ { i } \big ) } \\ & { f _ { t } = \sigma \big ( W _ { f } { * \big [ h _ { t - 1 } , X _ { t } \big ] + b _ { f } } \big ) } \\ & { o _ { t } = \sigma \big ( W _ { o } { * \big [ h _ { t - 1 } , X _ { t } \big ] + b _ { o } } \big ) } \\ & { C _ { t } ^ { ' } = \mathrm { t a n h } \big ( W _ { c } { * \big [ h _ { t - 1 } , X _ { t } \big ] + b _ { c } } \big ) } \\ & { C _ { u } = f _ { t } { * C _ { t - 1 } + i _ { t } * C _ { t } ^ { ' } } } \\ & { h _ { t } = o _ { t } * \mathrm { t a n h } \big ( C _ { t } \big ) } \end{array}
$$

其中， $W _ { i }$ 、 $W _ { f }$ 、 $W _ { c }$ 、 $W _ { o }$ 分别为输入门、遗忘门、更新门

和输出门的权值矩阵， $b _ { i }$ 、 $b _ { f }$ 、 $b _ { c }$ 、 $b _ { o }$ 分别为输入门、遗忘门、更新门和输出门的偏置，以此计算得到当前 $t$ 时刻的输出 $h _ { t }$ 与当前 $t$ 时刻更新的细胞状态 $C _ { t }$ 。

# 2 注意力模型CBAM

最近几年注意力模型在深度学习的各个领域被广泛使用，深度学习中的注意力机制的核心目标是从众多信息中选择出对当前任务目标更关键的信息。

本文中，采用 Convolutional Block Attention Mod-ule（CBAM）去实现 attention 机制。CBAM 表示卷积模块的注意力机制模块，它是一种为卷积神经网络设计的，简单有效的注意力模块，结合了空间和通道的注意力模块，相对于SENet多了一个空间attention，可以取得更好的效果。CBAM使得模型拥有了重视关键特征忽视无用特征的能力。对于卷积神经网络生成的特征图，CBAM从通道和空间两个维度计算特征图的权重图，然后将权重图与输入的特征图相乘来进行特征的自适应学习。CBAM是一个轻量的通用模块，可以将其融入到各种卷积神经网络中进行端到端的训练。图 2 为CBAM 网络结构图，其中 Channel attention module 主要关注于输入数据中有意义的内容。它的表达式为：

$$
M _ { c } ( F ) = \sigma \big ( M L P \big ( A v g P o o l ( F ) \big ) + M L P \big ( M a x P o o l ( F ) \big ) \big ) =
$$

$$
\sigma \big ( W _ { 1 } \big ( W _ { 0 } \big ( F _ { \mathrm { a v g } } ^ { c } \big ) \big ) \big ) + W _ { 1 } \big ( W _ { 0 } \big ( F _ { \mathrm { m a x } } ^ { c } \big ) \big )
$$

其中， $\boldsymbol { W _ { 0 } } \in \boldsymbol { R } ^ { c / r \times c }$ ， $W _ { 1 } \in r ^ { c / r \times c }$ 。

![](images/53765ee1ef28c47fbb9b28e2ad84f77adea29a668740c3714198c85cf1cc58ac.jpg)  
图2 CBAM网络结构图

而 Spatial Attention Module 主要关注于哪个位置信息是有意义的，是对于通道注意力的补充。它的表达式为：

$$
\begin{array} { r l } & { M _ { c } ( F ) = \sigma \big ( f ^ { 7 \times 7 } \big ( \big [ A v g P o o l ( F ) \big ] \big ) \big ) } \\ & { M L P \big ( M a x P o o l ( F ) \big ) = \sigma \big ( f ^ { 7 \times 7 } \big [ F _ { \mathrm { a v g } } ^ { s } , F _ { \mathrm { m a x } } ^ { s } \big ] \big ) } \end{array}
$$

# 3 网络模型结构

基于 LSTM-CNN-CBAM 的股票预测网络模型是在LINUX操作系统下搭建的，使用的是GPU版本的PyTorch框架。通过在结合长短时记忆神经网络和卷积神经网络的长记忆性分析的时间序列分类模型中加入了CBAM注意力机制，使模型自动学习和提取时间序列中的局部特征和长记忆性特征，模型展开如图3所示。

![](images/fe227f746bb67abd5103e1e649e284ce9fe37d24ad672d95a5f8c8f1501f70b8.jpg)  
图3 网络模型结构图

首先是LSTM模块，使用了3层LSTM神经网络学习数据中的时序特征，每层LSTM有128个隐藏神经元，学习率为0.001，迭代次数（epochs）为200次，然后将学习到的特征通过卷积神经网络进行特征学习和提取，并且加入了注意力机制，最后通过5层反向传播神经网络[15]输出预测价格，每个全连接层的神经元个数依次为1 024、128、64、20、1，激活函数使用ReLu函数。

# 4 实验及其结果分析

# 4.1 实验流程

实验主要由数据下载、数据处理、模型训练、微调参数[9]这几个部分组成，具体流程图如图4所示。

![](images/6ae823fd2b149973f0f0d83ed45e4bf8e7ed8e224816127708f644dae206e746.jpg)  
图4 实验流程图

# 4.2 实验数据

# 4.2.1 数据来源

本文的实验数据是利用Tushare财经接口包下载的上证指数 1991 年 1 月 1 日至 2018 年 12 月 28 日（共 6 847组数据），主要包含收盘价（close）、开盘价（open）、最高价（high）、最低价（low）、昨日收盘价（pre_close）、涨跌额（change）、涨跌幅（pct_chg）成交量（vol）、成交额（amount）等时序数据。

![](images/bbd8cfbeb3b17096ab0d86c588dc2775ab412945149c4761964c3b56c466c9b1.jpg)  
图5 时间步长 $= 5$

# 4.2.2 数据处理

数据预处理：由于获取到的原始数据集存在缺值和乱序等情况，所以要先对下载的数据集进行插值和按日期进行排序等操作，获得一个无乱序的完整数据集。

数据标准化[16]：由于数据集的数据之间量级不一样，例如开盘价、收盘价与成交量、成交额等数据量级之间存在着巨大的差异，为了消除数据之间不同量级的影响，将不同量级的数据统一转化为同一个量级，所以本模型对这些数据进行了 $\mathcal { Z }$ -score标准化处理，它是将观测值减去该组观测值的中值 $( \mu )$ ，再除以标准差 $( \sigma )$ 得到的，有利于提高模型的训练速度和预测精度。表达式如公式（10）所示：

$$
x ^ { \prime } { = } \frac { x - \mu } { \sigma }
$$

# 4.2.3 时间步数设置

因为LSTM神经网络具有时间序列的特性，本文将数据集的前 $8 5 \%$ 作为训练集数据，后 $1 5 \%$ 作为测试集数据。在LSTM-CNN-CBAM股票预测网络模型中，通过设置不同的时间步长进行实验对比，分别得到图 $5 { \sim } 1 0$ 的实验结果，通过实验可以发现，设置不同的步长时间，对预测结果的准确性具有影响。

通过观察图 $5 { \sim } 1 0$ 可以发现当时间步长为5时，因为考虑的时间步长较短，没有考虑到全局因素影响，预测结果有较大偏差，数据具有一定的波动。

当时间步长设置为30时，考虑的时间范围过大，容易忽略短时间内舆情等因素产生的影响，预测结果不准确。步长设置为20的时候，误差最小，准确率最高。所以最后将时间步长设置为20，用前20天的9个属性的数据作为神经单元的输入层，第21天的收盘价格作为标签进行训练模型。

![](images/19056b1c2b6080b0899ab47fac14dd38d74fa35d9a17913e0649852048b94b0e.jpg)  
图6 时间步长 ${ \boldsymbol { \cdot } } = 1 0$

![](images/33ebcf23f0350ec4ceaf06c7143e5e6e3b803e0911712c8156679d4739fa8801.jpg)  
图7 时间步长 $= 1 5$

![](images/1e3a9f96e6a3fc387efcf048ad6da599b2f3bdbe981f6aea39dca7fc2e7d4bfd.jpg)  
图8 时间步长 $= 2 0$

![](images/f7229fdf5bdfb3d673be68810fcb26b53d8a200948f2db97139bab57afa92f84.jpg)  
图9 时间步长 $= 2 5$

![](images/37788e47f764300abe17d320e8323f58643acaefce913732f71ac8b649162b0f.jpg)  
图10 时间步长 $= 3 0$

![](images/39f43a7f06c6b27822d260fd8283069ce7839f1dee2bcc5a922931870836a38f.jpg)  
图11 LSTM

![](images/ba10828acd14be2e2b971878dfd2b2f6efcdca3a432a1311deda8e5d94252b0a.jpg)  
图 12 LSTM-CNN

![](images/6ba916b3b632e363cdc1a2f59121a37de1fe2ea985caec3d6c68554383262da6.jpg)  
图13 LSTM-CNN-CBAM

# 4.3 实验结果

模型预测结果如图 11\~13所示，红色虚线为股票收盘价预测值，蓝色曲线为股票收盘价真实值，横坐标为时间，纵坐标为股票标准化处理后的价格。

通 过 观 察 对 比 实 验 拟 合 图 形 可 以 发 现 单 一 的LSTM网络对于股票价格的波动不敏感，而LSTM与卷积神经网络的结合模型有能力学习到股票价格波动的特征。然而，从图12可知，尽管LSTM-CNN模型可以拟合股票价格的大致波动，但拟合的精度较低，LSTM-CNN-CBAM的预测效果明显优于单纯的LSTM网络模型和LSTM-CNN网络模型，因为CBAM模块能够通过通道注意力机制从卷积神经网络产生的大量特征图中选择对预测结果有重要影响的特征图。同时，通过空间注意力机制能够从特征图的空间信息中选择有效的特征信息。该模型可以合理准确地预测到股票的价格。图13中，在150天左右，该模型对于股票的价格的预测值与实际值差别较大，这可能是由于股市受到当时的政府政策或者网络舆情的影响所造成的而非本文所提出的网络的缺陷所致。因此，本次实验验证了本文提出的网络具有有效性和可行性。

# 4.4 实验评价指标

本文主要目标是预测股指未来收盘价，采用均方根误差[17（] RMSE）对预测结果进行评价。均方根误差也称之为标准误差，是观测值与真实值之间的偏差，常用来作为机器学习模型预测结果衡量的标准。

$$
R M S E = \sqrt { \frac { 1 } { N } \sum _ { t = 1 } ^ { N } \bigl ( X _ { \mathrm { p r e d } , t } - X _ { \mathrm { r e a l } , t } \bigr ) ^ { 2 } }
$$

由表1可知，表中各模型之间的误差走势与图 $1 1 \sim$ 13中的预测价格曲线与实际价格曲线的之间的误差走势具有一致性。单一的LSTM模型的预测价格能力最差，LSTM-CNN 模型的预测价格能力次之。LSTM-CNN-CBAM模型的性能较其他模型的预测RMSE小，其预测值与真实值拟合图形的分散程度较小，预测精度最高。因此，表1定量地证明了本文中设计的网络模型的有效性。

表1 网络模型预测误差  

<table><tr><td>模型</td><td>RMSE</td></tr><tr><td>LSTM</td><td>0.1901</td></tr><tr><td>LSTM+CNN</td><td>0.0817</td></tr><tr><td>LSTM+CNN+CBAM</td><td>0.027 8</td></tr></table>

# 4.5 预测模型的时效性

时间复杂度决定了模型的预测时间。如果复杂度过高，则会导致模型预测耗费大量时间，既无法快速地验证想法和改善模型，也无法做到快速地预测。针对这一问题，对三种股票预测模型的时效性做了一个对比分析，运用训练集的数据进行预测，得到的三种模型的预测完成所需时间如表2。

表2 网络模型的预测时效性  

<table><tr><td>模型</td><td>用时/ms</td></tr><tr><td>LSTM</td><td>4.912</td></tr><tr><td>LSTM+CNN</td><td>7.890</td></tr><tr><td>LSTM+CNN+CBAM</td><td>8.702</td></tr></table>

由表2可知，三种模型的预测完成所需时间相差不大，只有几毫秒。在预测过程中不会耗费大量时间，可以做到快速的预测。因此，表2定量地证明了本文中设计的网络模型具有一定的时效性。

# 5 结束语

本文通过对在结合长短时记忆网络和卷积神经网络中引入CBAM进行理论研究与对比，并利用Python语言和PyTorch框架对模型进行代码实现，采用上证指数数据进行价格预测，通过与LSTM和LSTM-CNN的对比实验可以看出预测的准确率得到了一定的提升。表明了此模型对股票信息的预测是具有可行性和有效性的。通过对三种预测模型的时效性对比分析，证明了此模型具有良好的时效性。考虑到本次实验中在峰值处出现的预测值与真实值之间的误差，将会在未来的工作中考虑结合舆情分析等因素，进行文本挖掘[18]，希望能提高模型对股价预测的精准度，给股民的选择带来更加有价值的参考。

# 参考文献：

[1] 黄秋萍，周霞，甘宇健，等.SVM与神经网络模型在股票预测中的应用研究[J].微型机与应用，2015，34（5）：88-90.  
[2] 沈金榕.基于决策树的逐步回归算法及在股票预测上的应用[D].广州：广东工业大学，2017.  
[3] 方昕，李旭东，曹海燕，等.基于改进随机森林算法的股票趋势预测[J].杭州电子科技大学学报（自然科学版），2019，39（2）：25-30.  
[4] KROLLNER B，VANSTONE B，FINNIE G.Financial timeseries forecasting with machine learning techniques：asurvey[C]//European Symposium on ESANN，2010.  
[5] 陈祥一.基于卷积神经网络的沪深300指数预测[D].北京：北京邮电大学，2018.  
[6] 王锦涛. 基于混合模型的金融时间序列预测研究[D]. 郑州：郑州大学，2019.  
[7] 李新娟.基于长短期记忆与卷积结合的深度学习模型及试应用[D].上海：上海大学，2018.  
[8] 乔若羽. 基于神经网络的股票预测模型[J]运筹与管理，2019，28（10）：132-140.  
[9] WOO S，PARK J，LEE J Y，et al.Cbam：convolutionalblock attention module[C]//Proceedings of the EuropeanConference on Computer Vision（ECCV），2018：3-19.  
[10] HU J，SHEN L，SUN G.Squeeze-and-excitation networks[C]//Proceedings of the IEEE Conference on ComputerVision and Pattern Recognition，2018：7132-7141.  
[11] 孙瑞奇.基于LSTM神经网络的美股股指价格趋势预测模型的研究[D].北京：首都经济贸易大学，2015.  
[12] 李洁，林永峰.基于多时间尺度 RNN的时序数据预测[J].计算机应用与软件，2018，35（7）：33-37.  
[13] 任君，王建华，王传美，等.基于正则化LSTM模型的股票指数预测[J].计算机应用与软件，2018，35（4）：44-48.  
[14] 刘长坤.基于深度学习的股票预测研究[D].北京：北京工业大学，2018.  
[15] 肖琪.人工神经网络在股票预测中的应用研究[D].广州：华南理工大学，2017.  
[16] 冯宇旭，李裕梅.基于LSTM神经网络的沪深300指数预测模型研究[J]. 数学的实践与认识，2019，49（7）：308-315.  
[17] 邓凤欣，王洪良.LSTM神经网络在股票价格趋势预测中的应用-基于美港股票市场个股数据的研究[J].金融经济，2018（14）：98-100.  
[18] 郑国杰.基于互联网投资者情绪的股票时间序列分析与预测浙江[D].杭州：浙江工业大学，2019.