# stock-trend-prediction-using-candlestick-charting-and-4w2o4tspg2

## 1）元数据卡（Metadata Card）
- 标题：Stock Trend Prediction Using Candlestick Charting and Ensemble Machine Learning Techniques With a Novelty Feature Engineering Scheme
- 作者：Yaohu Lin, Shancun Liu, Haijun Yang, Harris Wu
- 年份：[未核验]
- 期刊/会议：[未核验]
- DOI/URL：https://digitalcommons.odu.edu/itds_facpubs
- 适配章节（映射到论文大纲，写1–3个）：2. Methodology, 3. Empirical Results,4. Conclusion
- 一句话可用结论（必须含证据编号）：The ensemble machine learning framework combining candlestick patterns and technical indicators (especially momentum indicators) achieves better short-term stock trend prediction performance than individual models, with momentum indicators improving F1 scores significantly for most patterns（依据证据E4, E9）.
- 可复用证据（列出最关键3–5条E编号）：E1, E2, E4, E6, E7
- 市场/资产（指数/个股/期货/加密等）：Chinese stock market（individual stocks, CSI300 Index, Shanghai Composite Index）
- 数据来源（交易所/数据库/公开数据集名称）：CCER（China Stock Market & Accounting Research Database）
- 频率（tick/quote/trade/分钟/日等）：Daily
- 预测目标（方向/收益/价格变化/波动/冲击等）：Next day's stock price direction（up/down）
- 预测视角（点预测/区间/分类/回归）：Classification
- 预测步长/窗口（horizon）：1 day
- 关键特征（尤其OFI/LOB/交易特征；列出原文术语）：13 one-day candlestick patterns, eight-trigram inter-day patterns（BullishHorn, BearHorn等）, 21 technical indicators（Overlap: MA/EMA; Momentum: ADX/RSI; Volume: OBV; Volatility: ATR）
- 模型与训练（模型族/损失/训练方式/在线或离线）：Ensemble of 6 models（RF/GBDT/LR/KNN/SVM/LSTM）, parameter tuning via GridSearchCV, offline training
- 评价指标（AUC/Accuracy/MAE/RMSE/收益等）：Accuracy, F1 score, Sharpe Ratio, Sortino Ratio, Maximum Drawdown
- 主要结论（只写可证据支撑的，逐条列点）：
  1. The candlestick-based feature engineering is effective（49/78 model-pattern combinations exceed random walk without indicators）（依据E3）；
  2. Momentum indicators are the most effective technical indicators（73/78 combinations exceed random walk）（依据E4,E9）；
  3. RF/GBDT perform best in most cases, while LSTM's advantage is not fully reflected（依据E6）；
  4. The investment strategy based on the framework yields better risk-adjusted returns than buy-and-hold（依据E7）.
- 局限与适用条件（只写可证据支撑的）：
  1. Transaction costs significantly reduce actual profits（依据E8）；
  2. Stop-trading rules limit profitability for certain patterns（依据E8）；
  3. Applicable only to Chinese stock market（依据E5）.
- 与本论文题目“OFI + 美股指数/代表性个股 + 短期预测”的关联（用证据编号支撑）：The paper uses daily data and technical indicators（including momentum）for short-term direction prediction, which provides a reference for short-term prediction of US stocks/indices（methodology applicable, though market differs）（依据E2,E4）.

## 2）可追溯证据条目（Evidence Items）
### E1
- 证据类型：定义
- 定位信息：Appendix I, Definition of Eight-Trigram
- 原文关键句："BullishHorn reflects that the oscillations of the day exceeded the previous cycle and reached a new high and a new low, reflecting the strong characteristics of an oscillation."
- 我的转述：Eight-trigram patterns are defined using two consecutive days' candlestick data（high, low, close prices）to capture inter-day price movements and market sentiment.
- 证据等级：A

### E2
- 证据类型：方法
- 定位信息：Section II.A, Table I
- 原文关键句："Four groups of technical indicators, including21 indicators, were introduced in our research: Overlap indicators（6）, Momentum indicators（9）, Volume indicators（3）, Volatility indicators（3）."
- 我的转述：The paper uses four groups of technical indicators（21 total）to enhance prediction: Overlap（MA/EMA等）, Momentum（ADX/RSI等）, Volume（OBV等）, Volatility（ATR等）.
- 证据等级：A

### E3
- 证据类型：结果
- 定位信息：Section III.B, Figure6（Fig.a）
- 原文关键句："First of all, we train the machine learning models without indicators.13 patterns and6 machine learning models resulting in a total of78 predictions shows the effectiveness of feature engineering...49 of the78 prediction models exceeded the random walk probability."
- 我的转述：Without technical indicators,49 out of78 model-pattern combinations（6 models ×13 patterns）achieved accuracy above random walk, validating the candlestick-based feature engineering.
- 证据等级：A

### E4
- 证据类型：结果
- 定位信息：Section III.B, Figure6（Fig.d）
- 原文关键句："After the introduction of Momentum indicators, the predicted maximum F1 score of each pattern are all improved...73 of the78 predictions exceeded the probability of random walk."
- 我的转述：Momentum indicators significantly improved prediction performance:73 out of78 combinations exceeded random walk, with most patterns' F1 scores increasing.
- 证据等级：A

### E5
- 证据类型：实验
- 定位信息：Section III.A
- 原文关键句："The daily data of the China Stock Market from the18-year period of2000 to2017 is used...All the3,455 stocks data is collected from CCER...We remove the daily data for a given stock if the trading volume is zero."
- 我的转述：The paper uses 18 years（2000-2017）of daily data from 3,455 Chinese stocks（CCER）, preprocessed by removing zero-volume days and balancing training data.
- 证据等级：A

### E6
- 证据类型：方法
- 定位信息：Section II.C, Prediction Models
- 原文关键句："The ensemble model includes six commonly-used effective prediction models（RF, GBDT, LR, KNN, SVM, LSTM）and optimizes the parameters of each model."
- 我的转述：The ensemble framework uses six models with parameter tuning（GridSearchCV）to select the best model for each candlestick pattern.
- 证据等级：A

### E7
- 证据类型：结果
- 定位信息：Section III.D, Figure8
- 原文关键句："The predicted maximum drawdown is71.4%, which is less than77.5% of the original stock. And the predicted Sharpe Ratio is0.31, which is bigger than0.25 of the original stock."
- 我的转述：The strategy based on the framework achieves better risk-adjusted returns: lower maximum drawdown and higher Sharpe ratio than buy-and-hold.
- 证据等级：A

### E8
- 证据类型：局限
- 定位信息：Section IV, Conclusion
- 原文关键句："However, the transaction costs have a significant impact on actual transactions...it is difficult to profit from certain patterns due to the stop-trading rules of the Chinese market."
- 我的转述：Practical profitability is limited by transaction costs and Chinese market-specific stop-trading rules.
- 证据等级：B

### E9
- 证据类型：结果
- 定位信息：Section III.B, Figure6（Fig.d）
- 原文关键句："The prediction effect of most prediction patterns has been improved after the introduction of these indicators, reflecting the obvious momentum characteristics in short-term prediction."
- 我的转述：Momentum indicators are the most effective for short-term prediction, reflecting strong momentum effects in the Chinese stock market.
- 证据等级：A

## 3）主题笔记（Topic Notes）
### Candlestick Pattern-Based Feature Engineering
依据E1,E2: The paper uses13 one-day candlestick patterns and eight-trigram inter-day patterns（based on two consecutive days' data）to capture price movements and sentiment. These are combined with 21 technical indicators（four groups）to form prediction features.

### Ensemble Machine Learning Framework
依据E6,E3: The framework employs six models（RF/GBDT/LR/KNN/SVM/LSTM）with parameter tuning. Without indicators, 49/78 combinations outperform random walk, validating base features.

### Technical Indicator Effectiveness
依据E4,E9: Momentum indicators are the most effective（73/78 combinations above random walk）, followed by overlap/volatility indicators. Volume indicators have minimal impact.

### Investment Strategy Performance & Limitations
依据E7,E8: The strategy yields better risk-adjusted returns than buy-and-hold, but transaction costs and stop-trading rules reduce actual profits.

## 4）可直接写进论文的句子草稿（可选）
1. The eight-trigram patterns, defined using two consecutive days' candlestick data, effectively capture inter-day price movements and market sentiment for short-term stock trend prediction（依据E1）.
2. Momentum indicators are the most effective among technical indicator groups, improving F1 scores for most candlestick patterns in short-term prediction（依据E4,E9）.
3. The ensemble framework using six machine learning models with parameter tuning selects the best model for each pattern, leading to superior prediction performance（依据E6）.
4. The investment strategy based on the ensemble framework achieves lower maximum drawdown and higher Sharpe ratio than buy-and-hold, indicating better risk-adjusted returns（依据E7）.
5. Transaction costs and market-specific stop-trading rules are key limitations that reduce the practical profitability of the prediction framework（依据E8）.
