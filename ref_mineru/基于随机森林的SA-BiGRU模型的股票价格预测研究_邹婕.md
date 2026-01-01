# 基於随机森林的SA-BiGRU模型的股票价格预测研究

邹婕李路①

摘要：为获取准确可靠的股价预测结果，针对股票数据非线性、多因子和时序性等特性，提出一种基于随机森林的SA-BiGRU（RF-SA-BiGRU）模型来预测股票收盘价。该方法在融合自注意力机制（SA）和双向门控循环单元（BiGRU)网络，构建SA-BiGRU模型的基础上，引入降维处理技术随机森林（RF）。采用 2008-2022 年顺鑫农业（00860）共260个因子数据，通过RF 对260 个因子进行降维筛选，将经过降维的股票数据作为 SA- BiGRU 模型的输入，先通过 BiGRU 充分提取股票数据本身的时空特征，再利用 SA 自动为 BiGRU 隐藏层状态分配相应的权重，减少历史信息的丢失和加强对重要因子的关注，然后传递给后续的神经网络层进行股价预测，实验显示RF-SA-BiGRU 模型的预测精度和稳定性均优于其他模型。

关键词：股价预测随机森林（RF）自注意力机制（SA）双向门控循环单元(BiGRU)

# 一、引言

股票市场受多种因素影响，其波动的幅度和不稳定性都极大。因此，股价预测一直是投资者在股票市场中关注的焦点以及大多学者研究的重难点。股价预测的方法主要分为线性预测模型和非线性预测模型，由于股票数据具有非线性、多因子和时序性等特性，非线性预测模型预测的效果并不理想。随着人工智能与大数据的发展，机器学习算法引起了越来越多的重视，逐渐取代了传统线性预测模型在金融预测方面的应用。机器学习中支持向量机、决策树和深度学习等技术广泛应用于金融数据的研究，主流的深度学习模型有卷积神经网络（CNN）、递归神经网络（RNN）和堆栈自编码网络（SAE）模型等。

对于股价预测这种序列预测问题，GRU和RNN都能进行序列建模，但是 RNN 在时间维度上有个先后顺序，输入的顺序会影响输出，CNN更多的是从局部信息聚合得到整体信息，对输入进行层次信息提取。此外，CNN仅考虑当前输入，而RNN考虑当前输入以及先前接收的输入，所以学者们更加倾向于利用RNN模型来预测股价，尤其是其衍生版本，如LSTM，GRU 等模型。其中GRU 不仅解决了RNN在模型训练过程中出现的长期记忆和反向传播中的梯度问题，还将输入门和遗忘门合成单一的更新门，混合了细胞状态和隐藏状态，最终的模型比LSTM模型更简单，参数量更少，减少了过拟合的风险。

Hinton etal.．(2006）阐述了神经网络的逐层训练方法，使得应用深度神经网络成为了可能。近年来，一些与GRU 相关的模型在股票价格预测问题中取得了较好的结果。谷丽琼（2020）提出基于Attention机制的GRU预测模型，模型更加关注重要时间点的股票特征并进行捕捉，从而解决股价预测中由于对时间特征不敏感而导致的预测精度不高的问题。

有时预测可能需要由前面若干输入和后面若干输入共同决定，这样会更加准确，因此提出双向门控循环单元网络（GRU）。曲衍旭（2022）提出利用BiGRU模型预测变频海水冷却系统状态参数，发现基于参数调优的BiGRU模型比GRU与RNN模型可以更为准确地预测系统预期状态参数，拥有优良的预测精度与稳定性，对船舶变频海水冷却系统参数预测具有参考价值。季威志（2020）提出BiGRU-CNN－Attention 模型来分析股市评论情感，BiGRU能更好地捕捉长文本前后的语义依赖和文本特征，结果表明 BiGRU－CNN－Attention模型在股票评论的情感分析上，具有较好的分类效果。

针对高维股票数据预测问题，陈亚丽(2021)建立基于RF-XGBoost 算法的汽油辛烷值损失预测模型，结果表明利用RF对高维数据降维后得到的特征更加全面，且可以有效提高预测的精度。

综上，针对高维的股票数据，直接使用预测模型进行股价预测，不仅训练时间成本较大，预测的精度和稳定性也有待提高。因此，本文提出一种基于随机森林的SA-BiGRU模型来预测高维股票数据的收盘价。

# 二、模型建立

本文的目标是基于随机森林的 SA- BiGRU（RF-SA-BiGRU）模型来预测股票的次日收盘价，使用过去 $T$ 天数据预测第 $T + 1$ 天数据，本文的时间窗口大小 $T$ 为10。

RF－SA-BiGRU模型由随机森林、双向GRU层、自注意力层，Flatten层以及全连接层组成。首先，对原数据进行预处理，以保证后续工作的可用性。利用RF从输入的 $M$ 个因子中选择 $Z$ 个因子构成最优因子集， $Z$ 个因子和1个标签共 $Z + 1$ 个因子构成降维后的股票因子数据。其次，构建SA-BiGRU模型，将 $N$ 天的股票因子数据按时间窗口大小 $T$ 逐步输入到BiGRU层提取特征的内部动态变化规律，将BiGRU层产生的特征作为SA的输入，利用SA自动对BiGRU隐藏层提取到的特征信息通过加权的方式进行重要程度的区分，挖掘更深层次的特征相关性。自注意力层加强了对输入数据中重要因子的关注以及因子内部的相关性的捕捉，突出输入中重要因子数据的影响作用。然后将自注意力层的输出作为Flatten层的输入，Flatten层用来将多维的输入转成一维，用于从自注意力层到全连接层的过渡。最后添加全连接层输出对次日收盘价的预测值。本文使用Adam优化算法更新模型的网络参数，损失函数采用均方误差（Mean Squared Error，MSE）。保存训练好的SA-BiGRU模型，并将模型应用于测试集上，通过分析模型评估结果来验证模型的有效性，如图1所示。

# （一）基于RF的特征选择

利用RF 进行特征选择，需要量化每个特征对构建的决策树分类性能的贡献度，贡献度用基尼指数GI(Gini index）来衡量，特征重要性评分（FeatureImportance Measures)用 $F I M$ 来表示。假设有 $M$ 个特征， $P$ 棵决策树， $C$ 个类别，则特征 $m$ 的重要性评分记为 $F I M _ { m } ^ { ( G i n i ) }$ O

![](images/099175d166e061ef93420adce15a943213820be0a999573788cd01a75e0d4953.jpg)  
图1 RF − SA − BiGRU 模型

计算 $M$ 个特征的重要性评分，按降序进行排序，选择重要性评分排名前 $Z$ 个特征，即经过RF，高维股票数据从 $M$ 维降到包含标签的 $Z + 1$ 维，降低预测模型工作量和复杂度的同时也提高了预测模型的预测效果。

# （二）基于BiGRU的特征提取

GRU是循环神经网络的一种。和LSTM一样，也是为了解决长期记忆和反向传播中的梯度等问题而提出来的（LinS，2022）。GRU网络中有两个门，分别是更新门和重置门，更新门控制前边记忆信息能够继续保留到当前时刻的数据量，重置门控制要遗忘多少过去的信息，GRU单元结构如图2。

![](images/91cad2bf76b72a738eb02439cef037fe719ad938b511721bb90ea7257499dff6.jpg)  
图2GRU网络内部结构

在单向的神经网络结构中，状态总是从前往后输出的，比如GRU对时序数据进行建模时无法编码从后到前的信息，而BiGRU则可以更好的捕捉双向的信息依赖，BiGRU结合一个前向GRU层和一个后向GRU层，结构如图3。可以看到前向GRU和后向GRU 共同连接输出层，其中 $x _ { t - 1 } , x _ { t } , x _ { t + 1 }$ 为输入数据，$R _ { t - 1 } , R _ { t } , R _ { t + 1 }$ 为向前迭代的 GRU隐藏状态， $Q _ { t - 1 } , Q _ { t }$ ,$Q _ { t + 1 }$ 为向后迭代的 GRU 隐藏状态， $Y _ { t - 1 } , Y _ { t } , Y _ { t + 1 }$ 为输出数据， $w _ { 1 } , w _ { 2 } , \cdots , w _ { 6 }$ 为共享权重。

![](images/438a8e5cddd46022bb7f7d2d42953884b7e4a2d49ae5eefb4367cb991e2216fb.jpg)  
图3 BiGRU网络结构

前向GRU、后向GRU的隐藏层更新状态以及BiGRU最终输出过程如式（1）-（3）所示。

$$
R _ { t } \ = s i g m o i d ( w _ { 1 } x _ { t } \ + w _ { 2 } R _ { t - 1 } )
$$

$$
Q _ { t } \ = s i g m o i d ( w _ { 3 } x _ { t } \ + w _ { 5 } Q _ { t + 1 } )
$$

$$
Y _ { \scriptscriptstyle t } \ = s i g m o i d ( w _ { \scriptscriptstyle t } R _ { \scriptscriptstyle t } \ + w _ { \scriptscriptstyle 6 } Q _ { \scriptscriptstyle t } )
$$

# （三）基于SA的特征融合

本文利用 SA 将股票特征进行融合，关注整个输入中不同特征之间的相关性。对于每个向量 $Y$ ，分别乘上三个系数 $w ^ { q } , w ^ { k } , w ^ { v }$ ，得到查询（Query， $Q$ ),键值对 (Key − Value, $K \ - \ V$ )，计算 $Q$ 和每个 $K$ 的相似性，再与 $\mathrm { V }$ 进行加权求和，得到最终的注意力数值。图4所示为 $Y _ { \mathrm { { t } } }$ 向量经过 SA 的输出过程，其他向量同理。

![](images/ee562ae003503c836f34d87282e1bcd0d79c4ed64d9904ec443aba95b5c9603f.jpg)  
图4SA原理

SA 的的计算公式如下：

（1）计算 $Q \setminus K \setminus V$   
$\boldsymbol { q } ^ { t } \ : = \boldsymbol { w } ^ { q } \cdot \boldsymbol { Y } ^ { t }$ 写成向量形式： $Q \ = \ \mathbb { W } ^ { q } \cdot I$   
$\boldsymbol { k } ^ { t } \ : = \boldsymbol { w } ^ { k } \cdot \boldsymbol { Y } ^ { t }$ 写成向量形式： $\boldsymbol { K } = \boldsymbol { \mathrm { W } } ^ { k } \cdot \boldsymbol { I }$   
$v ^ { t } = w ^ { v } \cdot Y ^ { t }$ 写成向量形式： $V = W ^ { v } \cdot I$   
其中 $I$ 为输入矩阵， $\boldsymbol { W } ^ { q } , \boldsymbol { W } ^ { k }$ (c: $\mathbb { W } ^ { v }$ 为变换矩阵。（2）计算权值矩阵 $A$   
$\hat { a } _ { i , j } \ = \ q ^ { i } \cdot k ^ { j }$ 写成向量形式： $\boldsymbol { A } = \boldsymbol { K } ^ { T } \cdot \boldsymbol { Q }$   
（3）计算注意力分数矩阵 $A ^ { \prime }$   
对矩阵 $A$ 进行softmax变换得到 $A ^ { \prime }$   
(4）计算输出矩阵 $o$   
$Y _ { \textit { t } } ^ { \prime } = \sum _ { j = 1 } ^ { T } v _ { t } \cdot a _ { \textit { t } , j } ^ { \prime }$ 写成向量形式： $O \ = \ V \cdot \ A ^ { \prime }$

# 三、实证分析

本节通过实验分析基于RF-SA-BiGRU模型预测股票次日收盘价的性能。

# （一）实验数据

本文的实验数据来源于优矿平台，选取顺鑫农业（000860）2008年3月31日至2022年7月29日的股票数据进行股价预测，共有 3396个样本260个因子，如表1。

表1股票因子  

<table><tr><td rowspan=1 colspan=1>因子类型</td><td rowspan=1 colspan=1>因子名称</td><td rowspan=1 colspan=1>因子数量</td></tr><tr><td rowspan=1 colspan=1>基本技术指标因子</td><td rowspan=1 colspan=1>preClosePrice、openPrice、highestPrice、lowestPrice 等</td><td rowspan=1 colspan=1>14</td></tr><tr><td rowspan=1 colspan=1>质量类因子</td><td rowspan=1 colspan=1>ARTDays、BLEV、CashRateOfSales 等</td><td rowspan=1 colspan=1>53</td></tr><tr><td rowspan=1 colspan=1>价值类因子</td><td rowspan=1 colspan=1>PB、PCF、PE、PS、FY12P、SFY12P等</td><td rowspan=1 colspan=1>15</td></tr><tr><td rowspan=1 colspan=1>情绪类因子</td><td rowspan=1 colspan=1>MAWVAD、PSY、VOL10、WVAD、RSI等</td><td rowspan=1 colspan=1>46</td></tr><tr><td rowspan=1 colspan=1>收益和风险类因子</td><td rowspan=1 colspan=1>CMRA、DDNBT、DDNCR、DDNSR等</td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=1 colspan=1>常用技术指标类因子</td><td rowspan=1 colspan=1>DHILO、EMA10、MFI、KDJ_K、RVI、TEMA10等</td><td rowspan=1 colspan=1>46</td></tr><tr><td rowspan=1 colspan=1>动量类因子</td><td rowspan=1 colspan=1>BIAS10、CCI10、ROC6、ChandeSD、REVS10等</td><td rowspan=1 colspan=1>56</td></tr><tr><td rowspan=1 colspan=1>成长类因子</td><td rowspan=1 colspan=1>EGRO、SUE、FinancingCashGrowRate、NetAssetGrowRate 等</td><td rowspan=1 colspan=1>15</td></tr><tr><td rowspan=1 colspan=1>其他因子</td><td rowspan=1 colspan=1>REC、ASSI、DilutedEPS、EPS、MACD</td><td rowspan=1 colspan=1>5</td></tr></table>

# （二）数据预处理

数据预处理包括数据整理、数据清洗、归一化处理。

第一步，数据整理。搜集与目标股票相关的因子，并将所选择的因子数据的时间跨度对齐，整理成目标股票的因子数据。

第二步，数据清洗。将重复、多余、有缺失、有错误的数据筛选清除，整理成完整的、可以进一步加工、使用的数据。

第三步，归一化处理。为了消除因子之间的量纲影响，需要进行数据标准化处理，让数据具有可比性。本文将数据归一化到[0，1]之间，表达式如式(4)。

$$
x _ { j } ^ { * } \ = \frac { x _ { \ j } ^ { \prime } - m i n ( x _ { j } ) } { m a x ( x _ { j } ) - m i n ( x _ { j } ) }
$$

式(4）中， $m i n ( x _ { j } )$ $m a x \big ( x _ { j } \big )$ 分别为第 $j$ 个特征数据中的最小值和最大值，其中 $j { \bf \Phi } = 1$ ,2, $\cdots 2 6 0$ 。

# （三）模型评价指标

本文选择均方根误差（RMSE）、平均绝对误差（MAE）和决定系数（ $\mathrm { R } ^ { 2 }$ ）三个评价指标来量化模型的预测性能，如表2。

表2模型预测评价指标  

<table><tr><td rowspan=1 colspan=1>指标名称</td><td rowspan=1 colspan=1>计算公式</td></tr><tr><td rowspan=1 colspan=1>RMSE</td><td rowspan=1 colspan=1>1$\frac$N(y i-y)2$MSE = </td></tr><tr><td rowspan=1 colspan=1>MAE</td><td rowspan=1 colspan=1>NMAE = 1N$\|y₁- y₁1</td></tr><tr><td rowspan=1 colspan=1>R</td><td rowspan=1 colspan=1>∑ ( -x)$R^ = 1$∑ (, − 2)2</td></tr></table>

RMSE 和 MAE 越小， $\mathrm { R } ^ { 2 }$ 越接近1，则模型预测的精确性越高，结果越好。其中， $y _ { i }$ 和 $\hat { y } _ { i }$ 分别为真实收盘价和预测收盘价， $N$ 为样本总数。

# （四）实验结果及分析

利用RF选择股票因子时，选取重要性评分大于等于0.01的因子，如图5 所示。

为验证本文提出的RF－SA-BiGRU 模型在股价预测方面具有较好的精确性和稳定性，选择GRU、BiGRU、RF – BiGRU、CNN –SA – BiGRU 和 RF –SA –GRU五个模型进行对比分析，为保证实验的可靠性和有效性，模型参数尽量保持一致。表3为6个模型的评估指标，为保证实验的客观性，6个模型分别运行20 次，取各项指标的平均值为最终结果。

![](images/e38013ea9c838ce5ccb1ffe3cfa8e4641a6f82e30fc3c8c5985281e5fa1dcfa1.jpg)  
图5最优因子集

表3模型评价指标  

<table><tr><td rowspan=1 colspan=1>模型</td><td rowspan=1 colspan=1>RMSE</td><td rowspan=1 colspan=1>MAE</td><td rowspan=1 colspan=1>R^$</td></tr><tr><td rowspan=1 colspan=1>GRU</td><td rowspan=1 colspan=1>6.083</td><td rowspan=1 colspan=1>5.003</td><td rowspan=1 colspan=1>0.828</td></tr><tr><td rowspan=1 colspan=1>BiGRURF − BiGRU</td><td rowspan=1 colspan=1>5.0913.571</td><td rowspan=1 colspan=1>4.1132.919</td><td rowspan=1 colspan=1>0.8790.941</td></tr><tr><td rowspan=1 colspan=1>CNN − SA – GRURF − SA − GRURF − SA – BiGRU</td><td rowspan=1 colspan=1>2.7862.5731.742</td><td rowspan=1 colspan=1>2.1982.0961.299</td><td rowspan=1 colspan=1>0.9640.9700.986</td></tr></table>

根据表3可知，利用RF-SA-BiGRU模型进行股价预测的RMSE、MAE、 $\mathrm { R } ^ { 2 }$ 三个评价指标均优于GRU、BiGRU、RF– BiGRU、CNN – SA– BiGRU 和RF-SA-GRU模型。此外，可以看出对于高维股票数据，直接利用GRU模型进行股价预测的效果较差，其评价指标与RF-SA-BiGRU模型的评价指标相比均具有较大差距，这也更加说明了本文提出的引入随机森林算法降维处理技术以及融入自注意力机制的 BiGRU 模型的有效性。

![](images/d836717e106be23140e95f6a7dc3bd3c3957537c96e8ceceb33c199a733755a8.jpg)  
图6顺鑫农业股价预测

为了更加直观的说明RF－SA-BiGRU模型在股价预测方面的优越性，图6表示6个模型分别在股票数据测试集上的预测曲线。

从图6中可以看出，RF-SA-BiGRU模型的预测曲线与真实股价曲线误差最小，相比于单一的

GRU 和 BiGRU预测模型，RF–SA- BiGRU模型的预测精度和稳定性提升效果非常明显。此外，RF-SA-BiGRU模型与RF–SA-GRU模型相比也具有更好的预测效果，这也说明了在上阶段工作研究的基础上优化RF-SA-GRU模型具有实验价值性。

# 四、结语

针对高维股票数据，构建RF-SA-BiGRU模型来解决股票预测问题。先利用降维处理技术RF 选择与股票收盘价高度相关的股票特征，达到降维的目的。再将降维的股票数据输入到SA-BiGRU预测模型中进行预测，通过BiGRU提取特征的内部变化规律和 SA挖掘更深层次的特征相关性，有效提高股价预测的精度和稳定性。本文将构建的RF－SA-BiGRU模型应用于顺鑫农业的股价预测，并与GRU、BiGRU、RF – BiGRU、CNN – SA – BiGRU 和 RF – SA –

GRU 五种模型对比。结果表明，RF-SA-BiGRU模型具有最高的预测精度和稳定，可以有效的应用于股价预测问题。但是，关于RF-SA-BiGRU模型是否可以通过调节参数或者改变层级结构进一步提高预测性能以及在一些特殊的股票上是否依旧具有较强的预测性能还有待进一步的研究。

# 参考文献：

陈亚丽，苟苗苗，邵露娟，等．基于RF-XGBoost算法的汽油辛烷值损失预测模型[J]．炼油技术与工程，2021，51(12)：49–53.

谷丽琼，吴运杰，逢金辉．基于Attention机制的GRU股票预测模型[J].系统工程，2020，38(05)：134–140.

曲衍旭，林叶锦，张均东，等．基于BiGRU的变频海水冷却系统状态参数预测[J]．大连海事大学学报，2022，48(01）：98-103.

Lin S, Shen S, Zhou A. Real – time analysis and prediction of shield cutterhead torque using optimized gated recurrent unitneural network [J]. Journal of Rock Mechanics and Geotechnical Engineering, 2022, 14 (04):1232 –1240.

（上接第51页)

# 四、结论与政策建议

本文从健康的角度研究了背景风险对家庭参与金融市场的影响，发现：在其他条件不变的情况下，当面临健康变差的冲击时，家庭投资风险资产的概率和份额都会出现显著性的下降。这种下降可以从风险厌恶水平、预期寿命（死亡风险）、医疗支出风险、金融财富四个渠道进行解释。其中，风险厌恶水平的解释力有限；后三者是主要渠道。本文的研究结论表明：家庭面临的背景风险增加（即健康变差导致家庭死亡风险、医疗支出风险上升）将使得家庭愿意承担的金融市场风险下降。因此，降低家庭健康风险的政策及其宣传在客观上有助于提升中国家庭的股市参与率。具体地，提出以下三点政策建议：

第一，加大“健康政策”和各种医疗保险的宣传，使更多人感知到自己的健康风险降低，从而更好地发挥健康政策促进家庭参与金融市场投资的作用。

第二，提升居民金融素养，尤其是农村居民，从而使健康政策促进家庭参与金融市场投资的传导链条更顺畅。对于低金融素养家庭，即使意识到家庭健康风险降低了，由于其对金融产品和金融知识的了解程度不够，可能也不会积极地调整家庭的风险资产投资决策。对此，应积极组织对金融产品和金融知识的普及活动，以更简单易懂的方式说明金融产品的运作过程和风险收益等，从而提升居民金融素养。

第三，加快“将心理治疗纳入医保支付”这一政策的全国推广。这一政策能显著提高居民的心理健康就医率，进而降低心理健康方面的患病风险和医疗支出风险。基于本文的结论，这些背景风险的降低有助于促进家庭参与金融市场投资。因此，未来应基于江苏、广东、深圳、北京等地的试点经验，进一步在全国范围内推广这一政策。

# 参考文献：

Atella V, Brunetti M, Maestas N. Household portfolio choices, health status and health care systems: a cross – country analysis based on SHARE [J]. Banking Finance, 2012 (5) : 1320 – 1355.

Ayyagari P, He D. The role of medical expenditure risk in portfolio allocation decisions [J]. Health Econ, 2017 (11) : 1447 – 1458.

Berkowitz M K, Qiu J. A further look at household portfolio choice and health status [J]. Journal of Banking & Finance, 2006 (4) : 1201 – 1217.

Love D A, Smith P A. Does health affect portfolio choice? [J]. Health Economics, 2010 (12) : 1441 – 1460.

Rosen H S, Wu S. Portfolio choice and health status [J]. Journal of Financial Economics, 2004 (3) : 457 –484.

Schurer S. Lifecycle patterns in the socioeconomic gradient of risk preferences [ J ]. Journal of Economic Behavior & Organization, 2015 (119):482 –495.