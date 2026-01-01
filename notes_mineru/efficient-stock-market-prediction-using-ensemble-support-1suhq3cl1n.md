# efficient-stock-market-prediction-using-ensemble-support-1suhq3cl1n

## 1）元数据卡（Metadata Card）
- 标题：Efficient Stock-Market Prediction Using Ensemble Support Vector Machine Enhanced with Genetic Algorithm for Feature Selection and Parameter Optimisation
- 作者：Isaac Kofi Nti*, Adebayo Felix Adekoya, Benjamin Asubam Weyori
- 年份：2020
- 期刊/会议：Computational Economics
- DOI/URL：https://doi.org/10.1515/comp-2020-0199
- 适配章节（映射到论文大纲，写1–3个）：Section2（Material and Methods）、Section3（Empirical Analysis & Discussion）、Section4（Conclusion）
- 一句话可用结论（必须含证据编号）：The proposed GASVM model outperforms DT, RF, NN, and ESVM in predicting 10-day-ahead stock price movement on the Ghana Stock Exchange with an accuracy of93.7%（依据证据E7、E8）
- 可复用证据（列出最关键3–5条E编号）：E1、E3、E5、E7、E9
- 市场/资产（指数/个股/期货/加密等）：Ghana Stock Exchange（GSE）、两家上市公司（银行与石油行业）
- 数据来源（交易所/数据库/公开数据集名称）：GSE官方网站（https://gse.com.gh）
- 频率（tick/quote/trade/分钟/日等）：日度
- 预测目标（方向/收益/价格变化/波动/冲击等）：10-day-ahead股票价格变动方向（上涨/下跌）
- 预测视角（点预测/区间/分类/回归）：分类（二元）
- 预测步长/窗口（horizon）：10-day
- 关键特征（尤其OFI/LOB/交易特征；列出原文术语）：Opening-Price、Closing-Price、Year-Lowest-Price、Year-Highest-Price、Total Stock-Traded-Volume、SMA、EMA、MACD、RSI、OBV、Stochastic %K、Stochastic %D、Accumulative Ratio（AR）、Volume Ratio（VR）
- 模型与训练（模型族/损失/训练方式/在线或离线）：Homogeneous Ensemble SVM（GASVM）；RBF kernel；GA特征选择与参数优化；多数投票融合；离线训练
- 评价指标（AUC/Accuracy/MAE/RMSE/收益等）：Accuracy、AUC、RMSE、MAE、Precision、Standard Deviation（SD）
- 主要结论（只写可证据支撑的，逐条列点）：1）GASVM预测准确率（93.7%）显著高于DT（75.3%）、NN（80.1%）、RF（92.3%）、ESVM（90.8%）（依据E7）；2）集成模型（RF、ESVM、GASVM）性能优于单一分类器（DT、NN）（依据E7）；3）GA特征选择与参数优化有效降低SVM过拟合风险并提升性能（依据E7、E9）；4）GASVM稳定性更高（SD=0.0484）（依据E7）
- 局限与适用条件（只写可证据支撑的）：1）GASVM训练时间长（18091.2秒），源于GA的迭代特征组合（依据E8）；2）仅采用GA优化，未对比其他技术（如PSO）（依据E9）；3）未纳入基本面数据（如客户满意度、新闻情绪）（依据E9）
- 与本论文题目“OFI + 美股指数/代表性个股 + 短期预测”的关联（用证据编号支撑）：本研究使用技术特征（含OBV等交易相关特征）进行短期（10-day）股价方向预测，与题目核心方向一致（依据E5、E7）


## 2）可追溯证据条目（Evidence Items）
### E1
- 证据类型：定义
- 定位信息：Section2.1 Ensemble GASVM
- 原文关键句："propose a 'homogeneous' ensemble classifier called (GASVM) based on an enhanced SVM with GA for feature selection and kernel parameter optimisation to predict stock market price movement."
- 我的转述：GASVM是基于SVM的同质集成分类器，通过GA进行特征选择和核参数优化，用于预测股票价格变动方向
- 证据等级：A

### E2
- 证据类型：方法
- 定位信息：Section2.1.1 SVM
- 原文关键句："The Gaussian Radial Basis Function was used as a kernelbased for our SVM for higher diversity achievement as reported in [20]."
- 我的转述：GASVM的基础SVM分类器采用高斯径向基函数（RBF）核，以提升集成模型的多样性
- 证据等级：A

### E3
- 证据类型：实验
- 定位信息：Section2.4 Study Dataset
- 原文关键句："We downloaded the stock-dataset for this study from the GSE official website (https://gse.com.gh), from June 25th,2007 to August27th,2019."
- 我的转述：研究数据集来自加纳证券交易所（GSE）官网，时间范围为2007年6月25日至2019年8月27日
- 证据等级：A

### E4
- 证据类型：实验
- 定位信息：Section2.4 Study Dataset
- 原文关键句："two companies from the banking and petroleum sectors, among the few companies listed before2005 that had fewer missing values in their dataset were selected for this study."
- 我的转述：数据集选取GSE中2005年前上市、缺失值较少的两家公司（银行与石油行业）
- 证据等级：A

### E5
- 证据类型：方法
- 定位信息：Section2.4 Study Dataset Table3
- 原文关键句："Nine (9) appropriate technical indicators were calculated and added to the opening-price, closing-price, year-lowestprice, year-highest-price, the total stock-traded-volume, as initial feature sets."
- 我的转述：初始特征集包含5个价格/成交量特征及9个技术指标（共14个），技术指标包括SMA、EMA、MACD、RSI、OBV等
- 证据等级：A

### E6
- 证据类型：方法
- 定位信息：Section2.1.2 Genetic Algorithm Table2
- 原文关键句："Population Size=250, Number of Generations=50, Genome length=100, Probability of Crossover=85%, Probability of Mutation=10%"
- 我的转述：GA优化参数为：种群大小250、迭代代数50、基因长度100、交叉概率85%、变异概率10%
- 证据等级：A

### E7
- 证据类型：结果
- 定位信息：Section3.2 Model Performance Table6
- 原文关键句："Accuracy values: GASVM=0.937, RF=0.923, ESVM=0.908, NN=0.801, DT=0.753"
- 我的转述：GASVM预测准确率（93.7%）在对比模型中最高，且稳定性优于其他模型（SD=0.0484）
- 证据等级：A

### E8
- 证据类型：结果
- 定位信息：Section3.2 Model Performance Table5
- 原文关键句："the GASVM training time was high (18091.2 sec), which can be attributed to the several different combinations (2^n, where n represent the number of input features) in the feature selection processes."
- 我的转述：GASVM训练时间长达18091.2秒，原因是GA的迭代特征组合过程
- 证据等级：A

### E9
- 证据类型：局限
- 定位信息：Section4 Conclusion
- 原文关键句："The current study adopted only a genetic algorithm for features and parameter optimisation based on findings in the previous studies without experimenting with other available optimisation techniques... stock price movement depends not only on historical stock data but also on fundamental data such as the satisfaction of the customers with the company’s market and web news."
- 我的转述：GASVM仅用GA优化且未纳入基本面数据，是其主要局限
- 证据等级：A


## 3）主题笔记（Topic Notes）
### GASVM模型的核心构成
依据E1、E2：GASVM是基于SVM的同质集成模型，通过GA实现特征选择与核参数优化，采用RBF核提升多样性，最终通过多数投票输出预测结果。

### 数据集与特征选择策略
依据E3、E4、E5：研究使用GSE 2007-2019年的日度数据，选取两家低缺失值公司；初始特征包含价格、成交量及9个技术指标，为后续GA优化提供基础。

### 模型性能对比分析
依据E7、E8：GASVM在准确率、稳定性上均优于其他模型，但训练时间显著更长；集成模型整体表现优于单一分类器，验证了集成学习的优势。

### GASVM的局限性与改进方向
依据E8、E9：GASVM的长训练时间、单一优化技术依赖及缺失基本面数据是主要不足，未来可引入PSO等优化技术及多源数据提升性能。


## 4）可直接写进论文的句子草稿（可选）
1. GASVM作为基于SVM的同质集成模型，通过GA优化特征与参数，在10-day-ahead股价方向预测中达到93.7%的准确率（依据E1、E7）。
2. 集成模型（如GASVM、RF）在股票预测任务中的表现显著优于决策树、神经网络等单一分类器（依据E7）。
3. GA的引入有效降低了SVM的过拟合风险，同时提升了模型的预测稳定性（依据E7、E9）。
4. 尽管GASVM准确率最高，但其训练时间长达18091.2秒，需在效率与性能间权衡（依据E8）。
5. GASVM未纳入基本面数据的局限，提示未来研究可结合新闻情绪等多源信息提升预测能力（依据E9）。
