# file

## 1）元数据卡（Metadata Card）
- 标题：Mid-price prediction based on machine learning methods with technical and quantitative indicators
- 作者：Ntakaris A, Kanniainen J, Gabbouj M, Iosifidis A
- 年份：2020
- 期刊/会议：PLoS ONE
- DOI/URL：https://doi.org/10.1371/journal.pone.0234107
- 适配章节（映射到论文大纲，写 1–3 个）：Feature pool, Proposed adaptive logistic regression feature, Experimental results
- 一句话可用结论（必须含证据编号）：The proposed adaptive logistic regression feature was consistently selected as the top feature by multiple sorting methods (E1, E2), and combining ~5 advanced features from quantitative and technical sets achieved optimal mid-price prediction performance (E3, E6).
- 可复用证据（列出最关键 3–5 条 E 编号）：E1, E3, E5, E6, E7
- 市场/资产（指数/个股/期货/加密等）：Nordic TotalView-ITCH stocks
- 数据来源（交易所/数据库/公开数据集名称）：FI-2010 dataset
- 频率（tick/quote/trade/分钟/日等）：millisecond (ITCH feed)
- 预测目标（方向/收益/价格变化/波动/冲击等）：mid-price movement direction (up/down/stationary)
- 预测视角（点预测/区间/分类/回归）：classification
- 预测步长/窗口（horizon）：next 10th/20th/30th ITCH events
- 关键特征（尤其 OFI/LOB/交易特征；列出原文术语）：adaptive logistic regression feature, order book imbalance (VI), technical indicators (Bollinger Bands, MACD, ATR), quantitative indicators (autocorrelation, cointegration, partial autocorrelation)
- 模型与训练（模型族/损失/训练方式/在线或离线）：LMS/LDA/RBFN classifiers; anchored cross-validation; online (adaptive logistic regression) and offline (wrapper selection)
- 评价指标（AUC/Accuracy/MAE/RMSE/收益等）：Accuracy, Precision, Recall, F1 score (emphasized)
- 主要结论（只写可证据支撑的，逐条列点）：1. The adaptive logistic regression feature was top-ranked across multiple sorting criteria (E1, E2); 2. Combining ~5 advanced features achieved near-max F1 performance (E3);3. LMS classifier outperformed LDA/RBFN across all horizons (E6);4. Wrapper feature selection with entropy/LMS/LDA was effective (E7)
- 局限与适用条件（只写可证据支撑的）：1. Limited to Nordic stocks and short trading periods (E8);2. Results not tested on longer periods or other assets (E8)
- 与本论文题目“OFI + 美股指数/代表性个股 + 短期预测”的关联（用证据编号支撑）：1. Uses order book imbalance (OFI-related feature) as key input (E5);2. Focuses on short-term mid-price prediction (aligned with topic) (E3, E6);3. Wrapper selection methods can be adapted to US stocks (E7)

## 2）可追溯证据条目（Evidence Items）
### E1
- 证据类型：结果
- 定位信息：Feature sorting section, Table9
- 原文关键句："The proposed advanced quantitative feature based on logistic regression for online learning has most of the time been selected as the top feature by the sorting methods."
- 我的转述：The adaptive logistic regression feature was frequently ranked as the number one feature in multiple feature sorting lists (entropy, LMS1, LMS2, LDA1, LDA2).
- 证据等级：A

### E2
- 证据类型：方法
- 定位信息：Proposed adaptive logistic regression feature section
- 原文关键句："We introduce a novel logistic regression model that we use as a feature in our experimental protocol... The new feature operates under an online learning mechanism by taking into consideration the latest trading event of the 10-event message book block."
- 我的转述：The adaptive logistic regression feature uses online learning with Newton's update method (Hessian matrix for adaptive rate) and leverages the latest 10-event message block to predict mid-price moves.
- 证据等级：A

### E3
- 证据类型：结果
- 定位信息：Experimental results section, Fig3
- 原文关键句："These models were able to reach (close to) their maximum F1 score performance with approximately 5 top features."
- 我的转述：Several models (e.g., LMS2-LMS, LDA2-LMS) achieved near-maximum F1 score using only ~5 top-ranked features.
- 证据等级：A

### E4
- 证据类型：实验
- 定位信息：Experimental results section
- 原文关键句："We evaluated our experimental framework on five ITCH feed data stocks from the Nordic stock market. The dataset contained over 4.5 million events which were incorporated into the hand-crafted features. The suggested labels describe a binary classification problem since we consider two states, one for change in the best ask price and another one for no change in the best ask price."
- 我的转述：The experiment used five Nordic stocks from the FI-2010 dataset with over 4.5 million ITCH feed events, using anchored cross-validation for training/testing and binary labels for mid-price change.
- 证据等级：A

### E5
- 证据类型：定义
- 定位信息：Feature pool section, quantitative analysis
- 原文关键句："Order book imbalance. We calculate the order book imbalance [4] based on the volume depth of our LOB as follows: VI = (V_l^b - V_l^a)/(V_l^b + V_l^a) where V_l^a and V_l^b are the volume sizes for the ask and bid LOB sides at level l."
- 我的转述：Order book imbalance (VI) is defined as the ratio of the difference between bid and ask volumes to their sum at a given LOB level l.
- 证据等级：A

### E6
- 证据类型：结果
- 定位信息：Experimental results section, Table4
- 原文关键句："LMS classifier achieved the best F1 performance for every predicted horizon."
- 我的转述：The LMS classifier consistently outperformed LDA and RBFN in F1 score across all prediction horizons (10,20,30 events).
- 证据等级：A

### E7
- 证据类型：方法
- 定位信息：Wrapper method of feature selection section
- 原文关键句："Our wrapper approach consists of five different feature subset selection criteria based on two linear and one non-linear methods for evaluation... convert entropy, LMS, and LDA as feature selection criteria."
- 我的转述：The wrapper feature selection method uses five criteria: entropy, two LMS variants (classification rate and L2-norm), two LDA variants (within-class/between-class scatter ratio and classification rate).
- 证据等级：A

### E8
- 证据类型：局限
- 定位信息：Conclusion section
- 原文关键句："Our work opens avenues for other applications as well. For instance, the same type of analysis is suitable for exchange rates and bitcoin time series analysis. As part of our future work, we also intend to test our experimental protocol on a longer trading periods."
- 我的转述：The study's results are limited to short trading periods (current dataset) and Nordic stocks; future work includes testing on longer periods and other assets like exchange rates or bitcoin.
- 证据等级：B

## 3）主题笔记（Topic Notes）
### OFI/Order Book Imbalance Definition
依据证据 E5: The paper defines order book imbalance (VI) as the ratio of bid-ask volume difference to their sum at a given LOB level (E5). This is a key quantitative feature used in mid-price prediction models.

### Adaptive Logistic Regression Feature Design
依据证据 E1,E2: The proposed feature uses online learning with Newton's update method (Hessian matrix for adaptive rate) and considers the latest 10-event message block (E2). It was consistently ranked as the top feature across multiple sorting methods (E1).

### Feature Selection & Optimal Feature Set Size
依据证据 E3,E7: The wrapper feature selection method uses five criteria (entropy, LMS1/2, LDA1/2) to identify informative features (E7). Models like LMS2-LMS achieved near-maximum F1 score using only ~5 top-ranked features (E3), showing that a small set of advanced features is sufficient for optimal performance.

### Classifier Performance Comparison
依据证据 E6: The LMS classifier outperformed LDA and RBFN in F1 score across all prediction horizons (10,20,30 events) (E6). This suggests linear models are effective for short-term mid-price prediction in high-frequency data.

### Data & Evaluation Setup
依据证据 E4: The study uses five Nordic stocks from the FI-2010 dataset with over 4.5 million ITCH feed events (millisecond frequency) (E4). Anchored cross-validation is used for training/testing to avoid look-ahead bias, and F1 score is emphasized due to class imbalance.

## 4）可直接写进论文的句子草稿（可选）
1. The order book imbalance (VI) is calculated as the ratio of bid-ask volume difference to their sum at a given LOB level, serving as a key quantitative feature in mid-price prediction (E5).
2. The proposed adaptive logistic regression feature, which uses online learning with Newton's update method, was consistently selected as the top feature by multiple sorting criteria (E1,E2).
3. Combining approximately five advanced features from quantitative and technical sets achieved near-maximum F1 score performance for short-term mid-price prediction (E3).
4. The LMS classifier outperformed LDA and RBFN across all prediction horizons (10,20,30 events) in terms of F1 score (E6).
5. The wrapper feature selection method uses five criteria (entropy, two LMS variants, two LDA variants) to identify informative features for high-frequency mid-price prediction (E7).
6. The study's results are limited to Nordic stocks and short trading periods; future work should test the framework on longer periods and other assets like US stocks or exchange rates (E8).
