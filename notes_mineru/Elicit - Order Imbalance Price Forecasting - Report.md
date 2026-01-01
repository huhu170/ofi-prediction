# Elicit - Order Imbalance Price Forecasting - Report

## 1）元数据卡（Metadata Card）
- 标题：Elicit - Order Imbalance Price Forecasting - Report
- 作者：[未核验]
- 年份：[未核验]
- 期刊/会议：[未核验]
- DOI/URL：[未核验]
- 适配章节（映射到论文大纲，写1–3个）：Abstract、Results - Short-term Price Prediction Accuracy Time Horizon Analysis、Limitations and Generalizability
- 一句话可用结论（必须含证据编号）：Order imbalance is an effective feature for short-term price prediction, with up to78% variance explained in US equities and AUC0.7-0.8 in Nasdaq large-tick stocks（依据E1, E2）
- 可复用证据（列出最关键3–5条E编号）：E1、E2、E3、E6、E8
- 市场/资产（指数/个股/期货/加密等）：US equities、Indian equities、Australian equities、CME agricultural futures、Bitcoin
- 数据来源（交易所/数据库/公开数据集名称）：Nasdaq、NYSE、CME、Australian Securities Exchange、Bitcoin exchanges
- 频率（tick/quote/trade/分钟/日等）：Seconds、minutes、intraday
- 预测目标（方向/收益/价格变化/波动/冲击等）：Price movement direction、price changes、returns/volatility
- 预测视角（点预测/区间/分类/回归）：Classification（binary）、regression、probabilistic
- 预测步长/窗口（horizon）：Seconds to1 hour
- 关键特征（尤其OFI/LOB/交易特征；列出原文术语）：Order flow imbalance、queue imbalance、limit order imbalance、trade imbalance、order book features
- 模型与训练（模型族/损失/训练方式/在线或离线）：Regression（linear/multivariate）、logistic regression、Markov chain-modulated pure jump model、generative temporal mixture model、ordered-probit-GARCH；训练方式[未核验]
- 评价指标（AUC/Accuracy/MAE/RMSE/收益等）：ROC AUC、variance explained（R²）、accuracy、profit boost（qualitative）
- 主要结论（只写可证据支撑的，逐条列点）：1）Order imbalance is effective for short-term price prediction across markets（依据E1, E4）；2）Performance varies by market and time scale，with up to78% variance explained in US equities（依据E2）；3）Key limitations are restricted market focus and short data periods（依据E6, E9）
- 局限与适用条件（只写可证据支撑的）：Restricted market focus（single exchanges/asset types）、short data periods（依据E6, E9）
- 与本论文题目“OFI + 美股指数/代表性个股 + 短期预测”的关联（用证据编号支撑）：The report covers US equities（relevant to 美股指数/个股）and short-term prediction（seconds to1 hour）with order imbalance as key feature（依据E1, E2, E4）

## 2）可追溯证据条目（Evidence Items）
### E1
- 证据类型：Result
- 定位信息：Abstract
- 原文关键句："Order imbalance measures appear to provide useful signals for short-term price prediction. In Nasdaq equities, queue imbalance applied in logistic regression yields receiver operating characteristic AUC values between0.7 and0.8 for large-tick stocks（and0.6–0.65 for small-tick stocks）, while linear models on New York Stock Exchange data explain65% of price variance from order flow imbalance."
- 我的转述：Order imbalance is a useful signal for short-term price prediction；logistic regression on queue imbalance gives AUC0.7-0.8 for large-tick Nasdaq stocks，and linear models on order flow imbalance explain 65% variance in NYSE data.
- 证据等级：A

### E2
- 证据类型：Result
- 定位信息：Results - Short-term Price Prediction Accuracy Time Horizon Analysis（Liu and Park,2015）
- 原文关键句："Liu and Park,2015:30 seconds to1 hour，Up to78% variance explained，US equities"
- 我的转述：Liu and Park（2015）report that multivariate linear models using order imbalance explain up to78% of price variance in US equities over30s-1h horizons.
- 证据等级：A

### E3
- 证据类型：Method
- 定位信息：Results - Prediction Method section
- 原文关键句："Regression-based approaches（linear, multivariate, or unspecified）:4 studies；Logistic regression:1 study；Markov chain-modulated pure jump model:1 study；Empirical observation:1 study；Generative temporal mixture（machine learning）model:1 study；Ordered-probit-GARCH:1 study；Order flow imbalance decomposition:1 study"
- 我的转述：Common models for order imbalance-based prediction include regression（linear/multivariate）、logistic regression、Markov chain models、and generative temporal mixture models.
- 证据等级：A

### E4
- 证据类型：Experiment
- 定位信息：Results - Market Type section
- 原文关键句："US equities:5 studies；Indian equities:1 study；Australian equities:1 study；CME agricultural futures:1 study；Bitcoin（cryptocurrency）:1 study"
- 我的转述：The review includes studies on US equities（5）、Indian（1）、Australian（1）、CME agricultural futures（1）、and Bitcoin（1）.
- 证据等级：A

### E5
- 证据类型：Result
- 定位信息：Results - Time Scale section
- 原文关键句："Seconds as primary time scale:2 studies；Minutes:3 studies；Seconds to minutes:1 study；Seconds to half-hour:1 study；30 seconds to1 hour:1 study；Intraday data:2 studies"
- 我的转述：Order imbalance prediction performance is evaluated across time scales from seconds to1 hour，with multiple studies focusing on minutes or intraday.
- 证据等级：A

### E6
- 证据类型：Limitation
- 定位信息：Limitations and Generalizability section
- 原文关键句："Restricted market or asset focus: For example, studies limited to a single exchange or specific asset types."
- 我的转述：A key limitation of existing studies is their restricted focus on single exchanges or specific asset types，reducing generalizability.
- 证据等级：A

### E7
- 证据类型：Result
- 定位信息：Results - Imbalance Type section
- 原文关键句："Order flow imbalance:3 studies；Order book imbalance/features:2 studies；Queue imbalance:1 study；Volume imbalance:1 study；Limit order imbalance:1 study；Trade imbalance:2 studies"
- 我的转述：Multiple imbalance metrics are used in prediction，including order flow、queue、limit order、and trade imbalance.
- 证据等级：A

### E8
- 证据类型：Result
- 定位信息：Results - Implementation Approaches table（Cont et al.,2010）
- 原文关键句："Cont et al.,2010: Linear regression；order flow imbalance；Variance explained（R²）=65%（order flow imbalance）,32%（trade imbalance）；95% significance"
- 我的转述：Cont et al.（2010）found linear regression on order flow imbalance explains65% of price variance in NYSE stocks（95% statistical significance）.
- 证据等级：A

### E9
- 证据类型：Limitation
- 定位信息：Limitations and Generalizability section
- 原文关键句："Short time frames or limited data periods: Some studies only cover brief periods，which may affect generalizability."
- 我的转述：Another limitation is that some studies use short data periods，which may impact the generalizability of their findings.
- 证据等级：A

## 3）主题笔记（Topic Notes）
### Order Imbalance Metrics and Their Use in Prediction
依据E7: Multiple imbalance metrics are used，including order flow、queue、limit order、and trade imbalance. Different metrics are applied in various models（e.g. queue imbalance in logistic regression for Nasdaq stocks，E1）.

### Performance Across Markets and Time Scales
依据E1, E2, E4, E5: Order imbalance is effective across markets（US equities、futures、Bitcoin）and time scales（seconds to1 hour）. For example，Nasdaq large-tick stocks have AUC0.7-0.8（E1），and US equities have up to 78% variance explained（E2）.

### Limitations of Existing Studies
依据E6, E9: Key limitations include restricted market focus（single exchanges/asset types）and short data periods，which affect generalizability.

### Model Types for Prediction
依据E3, E1, E8: Common models include regression（linear/multivariate）、logistic regression、and Markov chain models. Linear regression is used for variance explanation（E8），while logistic regression is for classification（E1）.

## 4）可直接写进论文的句子草稿（可选）
1. Order imbalance measures provide useful signals for short-term price prediction across various markets，including US equities、futures、and Bitcoin（依据E1, E4）.
2. Logistic regression on queue imbalance yields AUC values between0.7 and0.8 for large-tick Nasdaq stocks，indicating strong predictive power（依据E1）.
3. Multivariate linear models using order imbalance can explain up to78% of price variance in US equities over30-second to1-hour horizons（依据E2）.
4. Key limitations of existing studies include restricted market focus and short data periods，which need to be addressed in future research（依据E6, E9）.
5. Linear regression on order flow imbalance explains65% of price variance in NYSE stocks，with95% statistical significance（依据E8）.
