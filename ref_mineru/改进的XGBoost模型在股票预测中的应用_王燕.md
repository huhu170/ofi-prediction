# 改进的XGBoost模型在股票预测中的应用

王 燕，郭元凯兰州理工大学 计算机与通信学院，兰州 730050

摘 要：随着时代的不断进步，人民生活水平日益提高。在解决温饱问题之余，有了可供投资的余财。越来越多的人将目光转向股市投资，为股市发展提供了资金条件。然而在纷繁复杂的股票市场，如何寻找最优股成为亟待解决的问题。这不仅是投资者单方面的困惑，也是股票预测领域中学者们所关心的重点。通过网格搜索算法对XGBoost模型进行参数优化构建GS-XGBoost的金融预测模型，并将该模型运用于股票短期预测中。分别以中国平安、中国建筑、中国中车、科大讯飞和三一重工2005年4月至2018年12月28日的每日收盘价作为实验数据。通过实验对比，相较于XGBoost原模型、GBDT模型以及SVM模型，GS-XGBoost模型在MSE、RMSE与MAE三个评价指标上都表现出较好的预测结果。从而验证，GS-XGBoost金融预测模型在股票短期预测中具有更好的拟合性能。

关键词：XGBoost；网格搜索；梯度增强决策树（GBDT）；支持向量机（SVM）；股价预测文献标志码：A 中图分类号：TP181 doi：10.3778/j.issn.1002-8331.1904-0007王燕，郭元凯.改进的XGBoost模型在股票预测中的应用.计算机工程与应用，2019，55（20）：202-207.WANG Yan, GUO Yuankai. Application of improved XGBoost model in stock forecasting. Computer Engineering andApplications, 2019, 55（20）：202-207.

# Application of Improved XGBoost Model in Stock Forecasting

WANG Yan, GUO Yuankai

College of Computer and Communication, Lanzhou University of Technology, Lanzhou 730050, China

Abstract：With the continuous advancement of the times, people’s living standards have been increasing. In addition to solving the problem of food and clothing, there is surplus money available for investment. More and more people are turning their attention to stock market investment, which provides financial conditions for the development of the stock market. However, in the complicated stock market, how to find the optimal stock has become an urgent problem to be solved. This is not only a unilateral confusion for investors, but also a focus of scholars in the field of stock forecasting. In this paper, the grid prediction algorithm is used to optimize the XGBoost model to construct the financial forecasting model of GS-XGBoost, and the model is applied to short-term stock forecasting. The daily closing prices of China Ping An, China State Construction Engineering Corporation, CRRC Corporation Limited, IFLYTEK and SANY HEAVY INDUSTRY from April 2005 to December 28, 2018 are used as experimental data. Through experimental comparison, compared with the original XGBoost model, GBDT model and SVM model, the GS-XGBoost model shows good prediction results on the three evaluation indexes of MSE, RMSE and MAE. It is verified that the GS-XGBoost financial forecasting model has better fitting performance in short-term stock forecasting.

Key words：XGBoost; grid search; Gradient Boosting Decision Tree（GBDT）; Support Vector Machine（SVM）; stock price forecast

# 1 引言

随着时代的不断进步，人民生活水平日益提高。在解决温饱问题之余，有了可供投资的余财。越来越多的人将目光转向股市投资，为股市发展提供了资金条件。然而在纷繁复杂的股票市场，如何寻找最优股成为亟待解决的问题。这不仅是投资者单方面的困惑，也是股票价格预测领域中学者们所关心的重点。

股票价格是一种非常不稳定的时间序列，受多种因素的影响。影响股市的外部因素很多，主要有经济因素、政治因素和公司自身因素三个方面的情况。自股票市场出现以来，研究人员采用各种方法研究股票价格的波动。从经济角度来看，投资者普遍使用传统的基本面分析、技术分析和演化分析来预测。而这些传统的分析方法过于理论化，不能充分反映数据之间的相关性。随着数理统计的深入和机器学习的广泛应用，越来越多的人将现代预测方法应用于股票预测中，如神经网络预测[1]、决策树预测[2]、支持向量机预测[3]、逻辑回顾预测[4]、深度学习预测[5]等。文献[6]采用一种基于卷积神经网络（CNNs）的深度学习集成算法，通过与多层神经网络和支持向量机进行比较，说明合适的神经网络模型对股票价格的预测精度有一定的提高效果。文献[7]采用支持向量机（SVM）对股票买卖点进行预测，说明了SVM在股票预测中具有较好的表现。文献[8]采用 Adaboost集成算法对股票收益进行预测，体现了机器学习算法在股票预测中具有很好的预测性能。文献[9]采用贡献度与相关分析相结合的方法，利用梯度增强决策树（GBDT）对股票走势进行预测[10]。结果表明，GBDT组合模型在预测精度上优于线性回归组合模型和随机森林组合模 型[11-12] 。

XGBoost 是由 Tianqi Chen 在 2016 年提出来，并在文献[13]中证明了其模型的计算复杂度低、运行速度快、准确度高等特点。XGBoost 是 GBDT 的高效实现。在分析时间序列数据时，GBDT虽然能有效提高股票预测结果，但由于检测速率相对较慢，为寻求快速且精确度较高的预测方法，采用XGBoost模型进行股票预测，在提高预测精度同时也提高预测速率。本文将XGBoost 模型与网格搜索相结合[14]，发挥各自的优势，提高股票价格预测的准确性。实验中采用封装及其简便的sklearn框架下搭建XGBoost网络模型，然后利用网格搜索对其进行优化，构建网格搜索优化的XGBoost模型（本文称为 GS-XGBoost），利用改进的 XGBoost网络模型对中国平安、中国建筑、中国中车、科大讯飞和三一重工股票历史数据的收盘价进行分析预测，将真实值和预测值进行对比，最后通过中国平安股票的预测结果来评判GS-XGBoost模型对股价预测的效果。

# 2 XGBoost模型结构和原理介绍

XGBoost是一种基于梯度提升决策树的改进算法，它可以有效地构建增强树并且并行运行。XGBoost中有两种增强树，回归树和分类树[15-16]。优化目标函数的价值是XGBoost的核心。这里以目标函数为例进行理论介绍。

目标函数如公式（1）所示：

$$
O b j ^ { ( t ) } = \sum _ { i = 1 } ^ { n } l \big ( ( y _ { i } , \hat { y } ^ { ( t - 1 ) } ) + f _ { t } ( x _ { i } ) \big ) + \varOmega ( f _ { t } ) + C
$$

其中， $\hat { y } ^ { ( t - 1 ) }$ 表示保留前面 t - 1 轮的模型预测， $f _ { t } ( x _ { i } )$ 为一个新的函数， $C$ 为常数项。

将目标函数进行泰勒二阶展开，针对原来的目标函数为了方便进行计算，定义两个变量。如公式（2）所示：

$$
g _ { i } = \partial _ { \hat { y } ^ { ( t - 1 ) } } l ( y _ { i } , \hat { y } ^ { ( t - 1 ) } ) h _ { i } = \partial _ { \hat { y } ^ { ( t - 1 ) } } ^ { 2 } l ( y _ { i } , \hat { y } ^ { ( t - 1 ) } )
$$

可以看到这时候的目标函数可以改成公式（3）的形式：

$$
O b j ^ { ( t ) } \approx \sum _ { i = 1 } ^ { n } [ l ( y _ { i } , \hat { y } ^ { ( t - 1 ) } ) + g _ { i } f _ { t } ( x _ { i } ) +
$$

$$
\frac { 1 } { 2 } h _ { i } f _ { t } ^ { 2 } ( x _ { i } ) ] + \varOmega ( f _ { t } ) + C
$$

模型训练时，目标函数可以用公式（4）表示：

$$
O b j ^ { ( t ) } = \sum _ { j = 1 } ^ { t } [ ( \sum _ { i \in I _ { j } } g _ { i } ) w _ { j } + \frac { 1 } { 2 } ( \sum _ { i \in I _ { j } } h _ { i } + \lambda ) w _ { j } ^ { 2 } ] + \gamma T
$$

定义公式（5）：

$$
G _ { j } = \sum _ { i \in I _ { j } } g _ { i } , H _ { j } = \sum _ { i \in I _ { j } } h _ { j }
$$

将公式（5）带入公式（4）中，得到公式（6）：

$$
\begin{array} { c } { { ) b j ^ { ( t ) } = \displaystyle \sum _ { j = 1 } ^ { t } \bigl [ G _ { j } w _ { j } + \frac { 1 } { 2 } ( H _ { j } + \lambda ) w _ { j } ^ { 2 } \bigr ] + \gamma T = } } \\ { { - \displaystyle \frac { 1 } { 2 } \sum _ { j = 1 } ^ { T } \frac { G _ { j } ^ { 2 } } { H _ { j } + \lambda } + \gamma T } } \end{array}
$$

最后用图1说明目标函数评价决策树性能的用法。

![](images/b8b92c7958d919858c1e5b133592a7217f6b54bf5cddce1add08e2a479c7fd09.jpg)  
图1 目标函数评价决策树性能的示例

# 3 GS-XGBoost网络模型的搭建

网格搜索是指定参数值的一种穷举搜索方法，通过将估计函数的参数通过交叉验证的方法进行优化来得到最优的学习算法。将各个参数的可能取值进行排列组合，列出所有的组合结果生成“网格”。然后将各组合参数用于XGBoost训练，并使用交叉验证对表现进行评估。在拟合函数尝试了所有的参数组合后，返回一个合适的分类器，自动调整至最佳参数组合。为解决参数值随机选取的不确定性，本文构建了GS-XGBoost金融预测模型。首先，根据网格搜索算法的思想，先设定将要选择的参数组合区间，基于Xgboost算法，在参数寻优的过程中，结合网格搜索算法的思想，不断地训练模型，通过评价函数对每个参数组合得到的分类结果进行评价，最终得到最优参数组合，最后将最优参数组合代入Xgboost算法，从而使预测性能得到提升。在构建好GS-XGBoost模型后，进行多步预测，将该模型应用于股票连续30天的收盘价的预测中，然后将预测结果分别与原始 XGBoost 模型、GBDT 模型和 SVM 模型进行比较，最后根据模型评价指标进行验证。

具体步骤如下：

（1）获取股票的历史数据，进行缺失值处理，并将数  
据集分为训练集和测试集。（2）构建XGBoost预测模型。（3）将网格搜索算法应用于XGBoost模型中，构建  
GS-XGBoost预测模型，并使用训练集对GS-XGBoost模  
型训练进行训练。（4）然后使用测试集对GS-XGBoost预测模型进行  
测试。（5）对比真实结果与预测结果之间的差异。具体实验流程图如图2所示。

![](images/008335aaa2d75305aadde271b29e8693cc25ed5af3b9f6b8ffcddaca9e1c4c32.jpg)  
图2 实验流程图

# 4 评价指标

股票预测模型的预测性能评价指标采用均方误差（MSE）、均方根误差（RMSE）和平均绝对误差（MAE）三个评价指标对实验结果进行对比。

（1）均方误差是线性回归模型拟合过程中，最小化误差平方和（SSE）代价函数的平均值。预测效果越好，值越接近于0，反之，值越远离0，其计算公式如公式（7）所示：

$$
M S E = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( y ^ { ( i ) } - \hat { y } ^ { ( i ) } ) ^ { 2 }
$$

式中， $y$ 为预测的真实值， $\hat { y }$ 为预测值。

（2）均方根误差计算公式如公式（8）所示：

$$
R M S E = \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( y ^ { ( i ) } - \hat { y } ^ { ( i ) } ) ^ { 2 } }
$$

（3）平均绝对误差计算公式如公式（9）所示：

$$
M A E = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Bigr | y ^ { ( i ) } - \hat { y } ^ { ( i ) } \Bigr |
$$

# 5 实验结果及分析

本文实验在英特尔 $1 7 \ 3 . 1 \ \mathrm { G H z }$ 双核四线程CPU，4 GB RAM，Windows8操作系统的计算机进行，仿真平台为 pycharm，使用 python 语言进行编程，分别用到了python 中的 sklearn、pandas、numpy、Tushare 等包。

本文选取中国平安作为预测对象，利用python自带的Tushare包，下载中国平安2005年1月4日至2018年12月28日中的开盘价、收盘价、最高价、最低价、交易量等时序数据，共3 253条。中国平安的收盘价涨跌图如图3所示。

![](images/e2b167aac41fa914ed77ba2e59fc2f64e587a53fdfeb1d1ed3aa4cc4c5a9ce29.jpg)  
图3 中国平安收盘价涨跌图

实验1 测试XGBoost模型的预测性能，使用中国平安数据集的训练集训练XGBoost模型，将训练过后的XGBoost模型测试中国平安数据集的测试集，计算中国平安收盘价预测的均方误差（MSE）、均方根误差（RMSE）和平均绝对误差（MAE）。

XGBoost模型在构建之前，为了防止原始数据可能存在乱序及缺值的情况，首先对数据集进行插值、排序等操作[17]，从而得到规整的股票时序数据，进一步构建完整有效的数据集。在搭建完成模型后，对于3 253条中国平安的股票数据进行数据集拆分，将前3 223条数据作为训练集，最后30条数据作为测试集，在实验中经过多次调试与测试，在权衡计算量与模型的综合得分后将 XGBoost 模型参数学习率 learning_rate 设置为 0.1，树的深度 max_depth 设置为 2，树的棵树 n_estimators 设置为 45，最小叶子权重 min_child_weight 设置为 4，其余参数都设置为默认参数。在实验1中主要探索XGBoost模型对短期股价的预测性能。其实验结果如表1和图4所示。

表1 XGBoost模型的预测结果  

<table><tr><td></td><td>MSE</td><td>RMSE</td><td>MAE</td></tr><tr><td>XGBoost模型</td><td>0.0170</td><td>0.1305</td><td>0.115 7</td></tr></table>

![](images/e34c5890ef80bd2f31eeaed0a16e05d492b5b74148d31d288d599d6ce397f42d.jpg)  
图4 XGBoost收盘价预测结果图

由表1和图4可以看出XGBoost模型在预测中的均方误差为0.017 0，均方根误差为0.130 5，平均绝对误差为0.115 7。发现该模型的预测效果不太理想，图4中可以清楚地看到预测效果虽然在趋势上有所接近实际股价趋势，但是总体上股价预测值普遍低于实际值，因此在股价短期预测中该模型还需要改进。

实验2 结合网格搜索算法与XGBoost模型，通过采用网格搜索算法寻得 XGBoost 模型的最优解，构建GS-XGBoost模型。采用与实验1一致的测试集和训练集，在模型进行训练和测试后，与实验1的结果进行对比分析，实验2预测结果如图5和表2所示。

![](images/c014091a3541564ce5317ea3f87c535c9fa795f30253086d2ee50afe1f836b0a.jpg)  
图5 GS-XGBoost收盘价预测结果图

表2 GS-XGBoost模型的预测结果  

<table><tr><td></td><td>MSE</td><td>RMSE</td><td>MAE</td></tr><tr><td>GS-XGB模型</td><td>0.000 7</td><td>0.0268</td><td>0.0212</td></tr></table>

从实验2中的结果可知，GS-XGBoost模型的预测性能要明显高于XGBoost模型的预测性能，图5显示出中国平安30天收盘价的预测值与实际值的拟合度较高。对比表1和表2可知MSE、RMSE与MAE的值分别减少了 $0 . 0 1 6 \ 3 \ . 0 . 1 0 3 \ 7 \ . 0 . 0 9 4 \ 5 $ 。对比发现本文提出的GS-XGBoost模型在短时股价预测中比XGBoost模型具有更高的拟合度，模型预测效果更好。

为了更进一步验证GS-XGBoost模型在短时股价预测中的有效性，分别与同类型的 GBDT 模型比较，同时与在股价预测中表现良好的支持向量机模型进行对比。实验数据采用与实验1一致的测试集和训练集，同时为了验证模型的泛化能力和预测性能，分别将该模型应用于中国建筑、中国中车、科大讯飞和三一重工连续30天的收盘价预测中，其中4支个股的数据集为2005年1月4日至2018年12月28日中的开盘价、收盘价、最高价、最低价、交易量等时序数据，在模型进行训练和测试后，对实验结果进行对比分析。其实验对比结果如图 6\~10 和表 3\~7 所示。

![](images/0795ce6363c370a5059caf75e41636361e085cfa0628fd64e228c7538987ee6f.jpg)  
图6 GS-XGBoost模型与GB、SVM的对比（中国平安）

![](images/4e1f52b63918323b9079c1ed4fd7e1bd0b34026d75327583c811222e6080dc95.jpg)  
图7 GS-XGBoost模型与GB、SVM的对比（中国建筑）

从图 6\~10中可知，GS-XGBoost模型在预测值与实际值的拟合度上表现较好。其预测性能明显高于GBDT模型的预测性能，与 SVM 模型进行比较也具有相对优势。表3\~7显示GS-XGBoost模型在MSE、RMSE、MAE都具有出色的表现。

经过以上对比，说明 GS-XGBoost模型在短期股价预测中比同类型的 GBDT 模型和股价预测中表现良好的支持向量机模型的预测性能都要高。从而验证本文提出的 GS-XGBoost在短期股价预测中的可行性，以及其出色的预测性能。

![](images/549c304c209cde5741d3fd3818a8374c22d01623414adbcfba7a8a30ca4d4b4d.jpg)  
图8 GS-XGBoost模型与GB、SVM的对比（中国中车）

![](images/9f620d521df6fe22c28e098e2e1c9db3b247d44e27d30308ebdf0e90def736c1.jpg)  
图9 GS-XGBoost模型与GB、SVM的对比（科大讯飞）

![](images/e65aac9ae10169d706f82001b8af6c9d22ea8465a0643a343b95d5d646224e10.jpg)  
图10 GS-XGBoost模型与GB、SVM的对比（三一重工）

表3 中国平安收盘价的模型预测结果对比  

<table><tr><td>模型</td><td>MSE</td><td>RMSE</td><td>MAE</td></tr><tr><td>GS-XGBoost</td><td>0.0007</td><td>0.0268</td><td>0.0212</td></tr><tr><td>XGBoost</td><td>0.0170</td><td>0.1305</td><td>0.1157</td></tr><tr><td>SVM</td><td>0.0099</td><td>0.9984</td><td>0.9984</td></tr><tr><td>GB</td><td>0.3978</td><td>0.6307</td><td>0.5396</td></tr></table>

表4 中国建筑收盘价的模型预测结果对比  

<table><tr><td>模型</td><td>MSE</td><td>RMSE</td><td>MAE</td></tr><tr><td>GS-XGBoost</td><td>0.0013</td><td>0.0368</td><td>0.0145</td></tr><tr><td>XGBoost</td><td>0.025 9</td><td>0.1612</td><td>0.1546</td></tr><tr><td>SVM</td><td>0.0088</td><td>0.0938</td><td>0.0928</td></tr><tr><td>GB</td><td>0.3537</td><td>0.5947</td><td>0.5101</td></tr></table>

表5 中国中车收盘价的模型预测结果对比  

<table><tr><td>模型</td><td>MSE</td><td>RMSE</td><td>MAE</td></tr><tr><td>GS-XGBoost</td><td>0.0001</td><td>0.0102</td><td>0.0069</td></tr><tr><td>XGBoost</td><td>0.0135</td><td>0.1164</td><td>0.1063</td></tr><tr><td>SVM</td><td>0.0099</td><td>0.0997</td><td>0.099 7</td></tr><tr><td>GB</td><td>0.2958</td><td>0.5438</td><td>0.4687</td></tr></table>

表6 科大讯飞收盘价的模型预测结果对比  

<table><tr><td>模型</td><td>MSE</td><td>RMSE</td><td>MAE</td></tr><tr><td>GS-XGBoost</td><td>0.0013</td><td>0.0361</td><td>0.0238</td></tr><tr><td>XGBoost</td><td>0.0829</td><td>0.2879</td><td>0.2664</td></tr><tr><td>SVM</td><td>0.0099</td><td>0.099 9</td><td>0.0999</td></tr><tr><td>GB</td><td>0.2970</td><td>0.5450</td><td>0.4841</td></tr></table>

表7 三一重工收盘价的模型预测结果对比  

<table><tr><td>模型</td><td>MSE</td><td>RMSE</td><td>MAE</td></tr><tr><td>GS-XGBoost</td><td>0.0001</td><td>0.0124</td><td>0.0087</td></tr><tr><td>XGBoost</td><td>0.0107</td><td>0.1036</td><td>0.088 7</td></tr><tr><td>SVM</td><td>0.004 3</td><td>0.066 2</td><td>0.059 4</td></tr><tr><td>GB</td><td>0.2915</td><td>0.5399</td><td>0.4564</td></tr></table>

# 6 结束语

本文提出了一种基于网格搜索算法改进的XGBoost的金融时间序列模型，即GS-XGBoost模型。利用网格搜索算法对XGBoost模型进行参数优化来提高XGBoost模型在股票短期预测中的拟合度，采用中国平安、中国建筑、中国中车、科大讯飞和三一重工的收盘价对该模型进行验证。通过评价指标均方误差MSE、均方根误差 RMSE 和平均绝对误差 MAE 的对比，发现本文提出的模型在短期股价多步预测上具有较高的拟合度。但是总体的预测性能还需要进一步的提升，后期将考虑网络舆情、市场指数和国家政策等因素对股价涨跌情况的影响，来提高模型对股价预测的精准度。进一步给股民如何掌握股票的总体趋势带来更有价值的参考。

# 参考文献：

[1] Ticknor J L.A Bayesian regularized artificial neural network for stock market forecasting[J].Expert Systems with Applications，2013，40（14）：5501-5506.   
[2] Panigrahi S S，Mantri J K.A text based decision tree model for stock market forecasting[C]//International Conference on Green Computing & Internet of Things，2015.   
[3] Kong F，Song G P.Stock price combination forecast model based on regression analysis and SVM[J].Applied Mechanics & Materials，2010，39：14-18.   
[4] Gong J，Sun S.A new approach of stock price prediction based on logistic regression model[C]//International Conference on New Trends in Information & Service Science，2009.   
[5] Gao T，Xiu L，Chai Y，et al.Deep learning with stock indicators and two- dimensional principal component analysis for closing price prediction system[C]//IEEE International Conference on Software Engineering & Service Science，2017.   
[6] Tsantekidis A，Passalis N，Tefas A，et al.Forecasting stock prices from the limit order book using convolutional neural networks[C]//IEEE Conference on Business Informatics，2017.   
[7] Jaiwang G，Jeatrakul P.A forecast model for stock trading using support vector machine[C]//Computer Science & Engineering Conference，2017.   
[8] Zhang Guoying，Chen Ping.Forecast of yearly stock returns based on Adaboost integration algorithm[C]//2017 IEEE International Conference on Smart Cloud（SmartCloud）， 2017：263-267.

[9] Zhang Xiao，Wei Zengxin，Yang Tianshan.The applica-

Journal of Hainan Normal University（Natural Science）， 2018.   
[10] Wen Z，He B，Kotagiri R，et al.Efficient gradient boosted decision tree training on GPUs[C]//IEEE International Parallel & Distributed Processing Symposium，2018.   
[11] Naseem I，Togneri R，Bennamoun M.Linear regression for face recognition[J].IEEE Transactions on Pattern Analysis & Machine Intelligence，2010，32（11）：2106-2112.   
[12] Belgiu M，Drăguţ L.Random forest in remote sensing： a review of applications and future directions[J].ISPRS Journal of Photogrammetry & Remote Sensing，2016， 114：24-31.   
[13] Chen T，Guestrin C.XGBoost：a scalable tree boosting system[C]//ACM SIGKDD International Conference on Knowledge Discovery & Data Mining，2016.   
[14] Hokamp C，Liu Q.Lexically constrained decoding for sequence generation using grid beam search[J].Association for Computational Linguistics，2017，55：1535-1546.   
[15] Andersson J O.The new foundations of evolution：on the tree of life[J].Quarterly Review of Biology，2011，60（3）： 114-115.   
[16] Hendrikx J，Murphy M，Onslow T.Classification trees as a tool for operational avalanche forecasting on the Seward Highway，Alaska[J].Cold Regions Science & Technology，2014，97：113-120.   
[17] Chen Y Y，Chen T C，Chen L G.Accuracy and power tradeoff in spike sorting microsystems with cubic spline interpolation[C]//IEEE International Symposium on Circuits & Systems，2010.

# （上接第151页）

[7] 沈占锋，夏列钢，李均力，等.采用高斯归一化水体指数实现遥感影像河流的精确提取[J].中国图象图形学报，2013，18（4）：421-428.  
[8] 周艺，谢光磊，王世新，等.利用伪归一化差异水体指数提取城镇周边细小河流信息[J].地球信息科学学报，2014，16（1）：102-107.  
[9] Haralick R M，Shanmugam K，Dinstein I.Textural fea-tures for image classification[J].IEEE Transactions on Sys-tems Man & Cybernetics，2010，3（6）：610-621.  
[10] 崔佳玲.基于纹理的高分辨率遥感图像水陆分离算法[D].武汉：华中科技大学，2016.  
[11] 易焱，蒋加伏.基于LBP和栈式自动编码器的人脸识别算法研究[J]. 计算机工程与应用，2018，54（2）：163-167.  
[12] Wang Xin，Shen Siqiu，Ning Chen.Visual saliency detec-tion based on in-depth analysis of sparse representation[J].Optical Engineering，2018，57（3）.  
[13] Wang Xin，Xu Lingling，Zhang Yuzhen，et al.A novelhybrid method for robust infrared target detection[J].KSII Transactions on Internet and Information Sys-tems，2017，11（10）：4986-5002.  
[14] Ojala T，Pietikäinen M，Harwood D.A comparative studyof texture measures with classification based on fea-ture distributions[J].Pattern Recognition，1996，29：51-59.  
[15] Ning Chen，Liu Wenbo，Wang Xin.Infrared object recog-nition based on monogenic features and multiple ker-nel learning[C]//The 2018 IEEE International Confer-ence on Image，Vision and Computing，2018.  
[16] 邓滢，张红，王超，等.结合纹理与极化分解的面向对象极化SAR水体提取方法[J].遥感技术与应用，2016，31（4）：714-723.  
[17] 范雪婷，史照良，刘波.基于SVM的资源三号测绘卫星影像多特征分类[J]. 地理空间信息，2015，13（4）：23-26.