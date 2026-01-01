# 基于LSTM的股票价格预测建模与分析

彭 燕，刘宇红，张荣芬贵州大学 大数据与信息工程学院，贵阳 550025

摘 要：股价波动是一个高度复杂的非线性系统，其股票的调整不是按照均匀的时间过程推进，具有自身的推进过程。结合LSTM（Long Short-Term Memory）递归神经网络的特性和股票市场的特点，对数据进行插值、小波降噪、归一化等预处理操作后，推送到搭建的不同LSTM层数与相同层数下不同隐藏神经元个数的LSTM网络模型中进行训练与测试。对比评价指标与预测效果找到适宜的LSTM层数与隐藏神经元个数，提高了预测准确率约 $30 \%$ 。测试结果表明，该模型计算复杂度小，预测准确率有所提高，不仅能在股票投资前对预测股票走势提供有益的参考，还能帮助投资者在对实际股价有了进一步的认知后构建合适的股票投资策略。

关键词：小波降噪；长短期记忆网络（LSTM）层数；隐藏神经元；股价预测文献标志码：A 中图分类号：TP29 doi：10.3778/j.issn.1002-8331.1811-0239彭燕，刘宇红，张荣芬.基于LSTM的股票价格预测建模与分析.计算机工程与应用，2019，55（11）：209-212.PENG Yan, LIU Yuhong, ZHANG Rongfen. Modeling and analysis of stock price forecast based on LSTM. ComputerEngineering and Applications, 2019, 55（11）：209-212.

# Modeling and Analysis of Stock Price Forecast Based on LSTM

PENG Yan, LIU Yuhong, ZHANG Rongfen College of Big Data and Information Engineering, Guizhou University, Guiyang 550025, China

Abstract：Stock price volatility is a highly complex nonlinear system. The adjustment of stocks is not based on a uniform time process and has its own process of advancement. Combining the characteristics of LSTM（Long Short-Term Memory） recurrent neural network and the characteristics of stock market, and after preprocessing operations such as interpolation, wavelet noise reduction, and normalization of data, all of this data will be inputted into the LSTM network model of different LSTM layers and the number of different hidden neurons in the same layer for training and testing. Comparing the evaluation indicators with the prediction results, it finds the appropriate number of LSTM layers and hidden neurons, and improves the prediction accuracy by about $30 \%$ . The test results show that the computational complexity of this model is small and the prediction accuracy is improved. It not only provides a useful reference for predicting stock trend before stock investment, but also helps investors to build a suitable stock investment strategy after further understanding of the actual stock price.

Key words：wavelet noise reduction; number of Long Short-Term Memory（LSTM）layer; hidden neurons; stock price forecast

# 1 引言

随着人工智能技术与大数据技术的不断应用和发展，伴随着金融市场的进一步完善和金融服务业的强烈需求，股票市场预测引起了业界、学术界的广泛关注[1]。

决策树[2]、遗传算法[3-4]、支持向量机[5]、逻辑回归[6] 等机器学习算法[7]以及深度学习网络模型都被应用于股票预测研究中，文献[8]对比了几种机器学习算法与卷积神经网络在股票预测中的表现，验证了卷积神经网络模型的预测准确率优于传统机器学习算法。

文献[9]在 tensorflow 下搭建了 MLP 神经网络来对股票价格进行预测，通过与传统的BP神经网络方法对比，说明了合适的神经网络结构有助于提高模型的预测能力，使得预测耗时少且预测效果较好。

RNN（循环神经网络）被频繁用于分析预测序列数据[10]，但研究表明，随着时间的推移RNN会忘记之前的状态信息，故引入了LSTM（长短时循环神经网络）。LSTM时间递归神经网络具有适合处理和预测时间序列中间隔和延迟较长的重要事件的特性，近年来在很多领域表现突出。LSTM网络除了应用于图像分析、文档摘要、语音识别、手写识别等领域，在时序数据预测方面也具有较好的表现。它主要用于刻画当前数据与之前的输入数据之间的关系，利用其记忆能力，保存输入网络之前的状态信息[11]，利用之前的状态信息影响后续数据的确切值与发展趋势。基于LSTM网络特性，文献[12]将LSTM模型应用于股票波动率的预测，将其预测结果与18种经典模型预测结果对比，验证了LSTM的预测效果最好，且随着历史股票数据的增加，LSTM网络模型的预测效果趋于稳定。另外，文献[13]将LSTM网络模型应用于股票收益的预测中，对比了不同的输入特征对模型预测准确率的影响。

基于神经网络在股票预测中的优良表现与LSTM网络在数据预测中的特殊性，本文进一步将LSTM网络应用于股票预测中，重点探索LSTM网络适宜的层数和其前馈网络层的隐藏神经元个数，研究有效的LSTM股票预测网络模型。实验中采用封装及其简便的keras框架下搭建LSTM网络模型，利用不同层数的LSTM网络模型对苹果公司2000—2018年股票数据进行分析预测，将真实值和预测值进行对比，最后验证不同的LSTM网络深度对股价预测的影响。

# 2 LSTM的结构和原理介绍

LSTM网络结构采用控制门的机制，由记忆细胞、输入门、输出门、遗忘门组成[12]，具体结构如图1所示。结构图中 $X _ { t }$ 表示 $t$ 时刻的输入， $h _ { t }$ 表示 $t$ 时刻细胞的状态值。其中图中的三个不同的大框表示细胞在不同时序的状态，细胞中带有 $\sigma$ 的小框表示激活函数为sig-moid的前馈网络层[14]，同理，带有tanh的表示激活函数为tanh的前馈网络层。其中前馈网络层中的隐藏神经元个数经过不断的训练调试，对比和衡量各个模型的预测准确率后确定一个最佳值。下面介绍各个控制门的计算原理。

首先计算输入门 $i _ { t }$ 的值和在 $t$ 时刻输入细胞的候选状态值 $\tilde { C } _ { t }$ ，公式如下：

$$
i _ { t } = \delta ( W _ { i } { * ( X _ { t } , h _ { t - 1 } ) } + b _ { i } )
$$

$$
\tilde { C } _ { t } = \operatorname { t a n h } ( W _ { c } { * } ( X _ { t } , h _ { t - 1 } ) + b _ { c } )
$$

其次，计算在 $t$ 时刻遗忘门的激活值 $f _ { t }$ ，公式如下：

$$
f _ { t } = \delta ( W _ { f } \ast ( X _ { t } , h _ { t - 1 } ) + b _ { f } )
$$

由以上两步的计算，就可以计算出 $t$ 时刻的细胞状态更新值 $C _ { t }$ ，公式如下：

$$
C _ { t } = i _ { t } { * \tilde { C } _ { t } } + f _ { t } { * C _ { t - 1 } }
$$

在计算得到细胞状态更新值后，最后就可以计算输出门的值，其计算公式如下：

$$
O _ { t } = \delta ( W _ { o } \ast ( X _ { t } , h _ { t - 1 } ) + b _ { o } )
$$

$$
h _ { t } = O _ { t } { * } \operatorname { t a n h } ( C _ { t } )
$$

通过以上的计算，LSTM就可以有效地利用输入来使其具有长时期的记忆功能。

# 3 LSTM网络模型的搭建

首先，在linux操作系统下搭建GPU版本的keras框架。keras框架高度封装，具有模块化、简单、易扩展，且fine-tuning步骤简单等优点，其核心数据结构是模型。其包含两种模型，一种是 Sequential 模型，另一种叫Model模型，Sequential模型是一系列网络层按顺序构成的栈，是单输入和单输出的，层与层之间只有相邻关系，是最简单的一种模型。Model模型是用来建立更复杂的模型，本实验采用Model模型，分别搭建单层、两层、三层的LSTM网络模型分别对同一支股票进行分析预测。对于股票这种数据规模不是很大的情况，实验采用LSTM网络与全连接层连接方式，对股票价格进行短期及长期价格预测。其模型的预测性能评价指标采用均方根误差（RMSE）、平均绝对误差（MAE）、模型预测准确率（accurancy）来对实验结果进行对比。其中RMSE、MAE的计算公式如下：

![](images/1116786d499cb1ef3bc1c36aed0576c1f058ba375bf4e8473b2c750f24af9231.jpg)  
图1 LSTM网络结构图

$$
R M S E = \sqrt { \frac { 1 } { N } \sum _ { t = 1 } ^ { N } ( X _ { \mathrm { p r e d i c t i o n } , t } - X _ { \mathrm { r e a l } , t } ) ^ { 2 } }
$$

$$
M A E { = } \frac { 1 } { N } { \sum _ { i = 1 } ^ { N } } \Bigl | ( X _ { \mathrm { p r e d i c t i o n } , i } - X _ { \mathrm { r e a l } , i } ) \Bigr |
$$

图2展示了搭建的两层LSTM和全连接层网络模型的部分实验代码，经实验得出第一个LSTM层的前馈网络层的最佳隐藏神经元个数为10个，为避免过拟合现象采用L2正则化项和dropout机制，来提高模型的泛化能力。

![](images/3512835a859ece0d2a41cd14fd9e63f8f879153e33971807c616ed62278e77c6.jpg)  
图2 搭建的部分网络框架代码

# 4 实验及其结果分析

实验的训练流程按照下载数据、数据预处理、数据降噪、数据归一化、模型的训练、微调参数、价格预测几步完成。

数据下载：利用python自带的Tushare包，下载苹果公司 2000—2018 年包含开盘价、最高价、最低价、交易量、调整后的收盘价、收盘价等时序数据。

数据预处理：获取到的原始数据可能存在乱序及缺值的情况，需要进行插值、排序等操作[15]，从而得到规整的股票时序数据，进一步构建了完整有效的数据集。数据集总共包含苹果公司从 2000-01-26 到 2018-09-28 的数据共4 718条。

数据降噪：由于复杂的市场动态，这些数据含有不经常的噪声，所以采用 python 中自带的 pywt 库来进行小波变换去除数据噪声。

数据归一化：由于股票的价格和交易量等参数同时作为特征值输入，交易量的数值一般达到上亿，而价格大多在几十上百，不能因为交易量的数值过大就表现出对价格的影响比例大，所以需要将特征序列进行归一化处理。

微调参数：在模型的训练过程中不断地调整网络框架结构以及LSTM层中正则化项参数值，直至效果模型的预测效果最佳。整体实验流程如图3所示。

![](images/050e2e2d5694e28384164332c4c39c0fb7aa69ae88dc433762bd7fc57185dde1.jpg)  
图3 实验流程图

（1）首先搭建单层的LSTM网络和全连接层模型，对股票价格进行长期的预测，对于苹果公司的股票数据，将2000—2014年的数据作为训练集，2015—2018年的数据作为测试集，即为图4中700天的测试数据，在实验中经过多次调试与测试，在权衡计算量与模型的预测准确度后将前馈网络层的隐藏神经元个数设置为10个，在实验一中主要探索一层LSTM网络对股价的预测性能，其实验结果如图4和表1所示。

![](images/9f55540813a741618854b865675c6c03455e9d81e165b91f13d9f066d19c73ff.jpg)  
图4 单层LSTM网络模型预测结果图

表1 单层LSTM网络模型预测结果  

<table><tr><td>性能</td><td>RMSE</td><td>MAE</td><td>accurancy</td></tr><tr><td>数值</td><td>2.58</td><td>1.74</td><td>0.44</td></tr></table>

从表1和图4直观地看出单层的LSTM的预测效果不理性，不论是前200天还是整个700天的预测价格虽然与实际股价走势一致，但只是达到对股票的趋势预测，与实际的价格存在一定的差距，预测值普遍高于实际值，短期预测与长期预测的性能都需要进一步的提高。

（2）实验二搭建了两层的LSTM网络和全连接层模型，其中第一个LSTM层的隐藏神经元个数与实验一相同，并采用与实验一相同的输入值与测试值，推送到模型进行训练与测试后，与实验一进行预测性能的对比分析，其实验二结果如图5和表2所示。

![](images/534bb7826cb956fb1d8f6eded557f4f6c880ba9c42dfc4a24219b7625bbdb9c7.jpg)  
图5 两层LSTM网络模型预测结果图

表2 两层LSTM网络模型预测结果  

<table><tr><td>性能</td><td>RMSE</td><td>MAE</td><td>accurancy</td></tr><tr><td>数值</td><td>0.46</td><td>0.108</td><td>0.78</td></tr></table>

从实验二中的结果可知，两层的LSTM网络模型的预测性能大大提升，图5显示出前200天的股价预测值与实际值拟合度较高，后期虽然预测值与实际值出现了偏差，但是对于短期预测不造成股价参照影响。表2中RMSE与MAE值分别减少了2.12、1.632，预测准确率比实验一提升了约 $30 \%$ ，对比文献[9]搭建的多层感知器MLP（Multi-Layer Perceptron）神经网络模型对苹果公司股价的预测评价指标RMSE的值减少0.164，说明本文实验二中搭建的模型对股票价格预测的准确度更高。从计算复杂度方面比较，对于 LSTM 神经网络而言，每输入一步，每一层各自共享参数，而MLP神经网络层参数是不共享的，使得对比搭建相同层数的两种网络，LSTM神经网络能大大减少网络中需要学习的参数，从而降低计算复杂度。证明了LSTM时间递归神经网络用于处理股票序列数据的优越性。其次，本文对下载的股票数据进行的小波降噪、归一化等预处理操作，利于模型提取更多有效的特征，使得预测准确率得到进一步提升。

另外，由文献[13]可知适当增加LSTM网络层数可以增强输入序列的特征提取，把握数据走势，提升模型预测的准确率，本文在前两个实验基础上增加到三层LSTM网络，可是预测准确率只提升了 $0 . 0 0 2 \%$ ，说明不断地增加网络的深度并不能使得预测性能一定改善，反而增加了计算冗余。同时也表明了对于股票这样的时间序列数据，在适宜的网络层数下继续增加网络的层数对预测准确率的提升效果并不明显，只是表现于数据预测的微小抖动，故综合对比计算量和预测性能，在适宜的网络层数下不再增加网络层数。进一步突出了本文搭建的两层LSTM网络模型在股票预测中的准确度和计算消耗上的优异表现。

# 5 结束语

对于股票市场来说，只要知道近期股票的走势以及价格，就对股民的选择带来有价值的参考，本文通过对股票数据进行小波去噪、归一化等预处理操作后，在LSTM网络模型中引入L2正则化项和dropout机制，对比实验探索适宜的LSTM网络层数和前馈网络层的隐藏神经元个数，来对苹果公司股票价格进行分析预测，使得短期预测值和实际价格拟合度较高，股价预测准确率提高约 $30 \%$ 。但是总体的预测准确度需要进一步的提升，后期考虑加入对股票影响大的爆炸性新闻、市场指数等特征来训练模型，希望能提高模型对股价预测的精准度，给股民的选择带来更加有价值的参考。

# 参考文献：

[1] Agrawal J G，Chourasia D V S，Mittra D A K.State-of-the-art in stock prediction techniques[J].International Journalof Advanced Research in Electrical Electronics & Instru-mentation Engineering，2013，2（4）.  
[2] 沈金榕.基于决策树的逐步回归算法及在股票预测上的应用[D].广州：广东工业大学，2017.  
[3] 张炜.基于遗传算法的属性约简方法在股票预测中的应用研究[D].长沙：湖南大学，2013.  
[4] 李忍东，饶佳艺，严亚宁.基于智能计算的股票价格预测[J].科技通报，2013，29（4）：152-154.  
[5] 黄秋萍，周霞，甘宇健，等.SVM与神经网络模型在股票预测中的应用研究[J].微型机与应用，2015，34（5）：88-90.  
[6] Huo J，Zheng Y，Chen X.Implementation of transactiontrend prediction model based on regression analysis[J].Journal of Baoshan Teachers’College，2009（1）：19-23.  
[7] Krollner B，Vanstone B，Finnie G .Financial time seriesforecasting with machine learning techniques：a survey[C]//European Symposium on ESANN，2010.  
[8] 陈祥一.基于卷积神经网络的沪深300指数预测[D].北京：北京邮电大学，2018.  
[9] 韩山杰，谈世哲.基于TensorFlow进行股票预测的深度学习模型的设计与实现[J].计算机应用与软件，2018，35（6）：267-271.  
[10] 李洁，林永峰.基于多时间尺度 RNN 的时序数据预测[J].计算机应用与软件，2018，35（7）：33-37.  
[11] 陈卫华，徐国祥.基于深度学习和股票论坛数据的股市波动率预测精度研究[J]. 管理世界，2018，34（1）：180-181.  
[12] 陈卫华.基于深度学习的上证综指波动率预测效果比较研究[J]. 统计与信息论坛，2018，33（5）：99-106.  
[13] Chen K，Zhou Y，Dai F.A LSTM- based method forstock returns prediction：a case study of China stockmarket[C]//IEEE International Conference on Big Data，2015：2823-2824.  
[14] 於雯，周武能. 基于 LSTM 的商品评论情感分析[J]. 计算机系统应用，2018，27（8）：159-163.  
[15] 周恺越.基于深度学习的股票预测方法的研究与实现[D].北京：北京邮电大学，2018.