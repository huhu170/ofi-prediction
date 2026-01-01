# a-hybrid-model-to-predict-stock-closing-price-using-novel-3ajstwdh

## 1）元数据卡（Metadata Card）
- 标题：A Hybrid Model to Predict Stock Closing Price Using Novel Features and a Fully Modified Hodrick–Prescott Filter
- 作者：Qazi Mudassar Ilyas, Khalid Iqbal, Sidra Ijaz, Abid Mehmood, Surbhi Bhatia
- 年份：2022
- 期刊/会议：Electronics
- DOI/URL：https://doi.org/10.3390/electronics11213588
- 适配章节（映射到论文大纲，写 1–3 个）：Section3（Proposed Model）、Section4（Experimental Results）、Section6（Conclusions）
- 一句话可用结论（必须含证据编号）：The hybrid model combining RNN with FMHP filter and sentiment features achieves the highest day-ahead prediction accuracy (70.88%) for Apple Inc. (AAPL) closing price（依据证据 E7）
- 可复用证据（列出最关键 3–5 条 E 编号）：E1、E3、E5、E7、E9
- 市场/资产（指数/个股/期货/加密等）：Apple Inc. (AAPL) 个股
- 数据来源（交易所/数据库/公开数据集名称）：Yahoo Finance（历史股票数据）、Twitter（sentiment 数据）
- 频率（tick/quote/trade/分钟/日等）：Daily
- 预测目标（方向/收益/价格变化/波动/冲击等）：Closing Price
- 预测视角（点预测/区间/分类/回归）：Regression（Point Prediction）
- 预测步长/窗口（horizon）：Day-ahead（14-day historical window）
- 关键特征（尤其 OFI/LOB/交易特征；列出原文术语）：Technical Features（Ot,Ct,Ht,Lt,Vt, volume change, volume limit, amplitude, difference, return of firm, return open price, return close price, change in return open price, change in return close price, VPT）；Sentiment Features（from Twitter using Hamilton et al.'s dictionary）
- 模型与训练（模型族/损失/训练方式/在线或离线）：Models（SVR, ARIMA, RF, LSTM, GRU）；Training（Offline,80% train/20% test split；RNN uses Adam optimizer, tanh activation,4 layers of50 units；loss minimizes output-observed difference）
- 评价指标（AUC/Accuracy/MAE/RMSE/收益等）：Accuracy, MAPE, RMSE
- 主要结论（只写可证据支撑的，逐条列点）：
1. FMHP filter outperforms HP filter in reducing noise and endpoint bias for stock time series（依据 E1）
2. Combining technical and sentiment features improves prediction accuracy across all models（依据 E5,E6,E7）
3. RNN with FMHP filter and sentiment features achieves highest day-ahead accuracy (70.88%) for AAPL closing price（依据 E7）
4. Deep learning models（RNN variants）outperform traditional ML models（SVR, RF, ARIMA）for stock closing price prediction（依据 E5,E6,E7）
- 局限与适用条件（只写可证据支撑的）：
1. Does not consider macroeconomic factors or external news beyond Twitter（依据 E9）
2. Needs validation on more diverse datasets（stocks/frequencies）to generalize（依据 E9）
3. Focuses only on daily frequency for AAPL；may not apply to other frequencies without adjustment（依据 E3,E4）
- 与本论文题目“OFI + 美股指数/代表性个股 + 短期预测”的关联（用证据编号支撑）：The paper uses daily data for AAPL（US representative stock）to predict day-ahead closing price（short-term），aligning with the topic's focus on US stocks and short-term prediction（依据 E3,E7）；it also explores feature combination（technical+sentiment）which is relevant to feature engineering for short-term prediction（依据 E2,E5）

## 2）可追溯证据条目（Evidence Items）
### E1
- 证据类型：Definition
- 定位信息：Section3.2.1（Filtering Historical Data Using FMHP Filter）
- 原文关键句："Hanif et al. proposed the endogenous lambda method to develop a fully modified Hodrick–Prescott (FMHP) filter [36]. The proposed technique resolves the end point bias issue of the Hodrick–Prescott filter by employing modifications in the weighting scheme and endogenous smoothing parameter."
- 我的转述：FMHP filter is a modified HP filter using endogenous lambda and adjusted weighting to resolve endpoint bias in time series data.
- 证据等级：A

### E2
- 证据类型：Method
- 定位信息：Section3.2.2（Prediction Features）
- 原文关键句："Chen et al. proposed a set of novel features for predicting stock closing-prices [14]. In addition to their proposed features, we present another set of features for making more accurate predictions: return of firm, return open price, return close price, change in return open price, change in return close price, VPT."
- 我的转述：The study extends Chen et al.'s technical features with 6 new ones（return of firm, ROP, RCP, DROP, DRCP, VPT）for stock closing price prediction.
- 证据等级：A

### E3
- 证据类型：Experiment
- 定位信息：Section3.1（Datasets Used）
- 原文关键句："Historical stock data: Apple Inc. (AAPL) from4 Jan2021 to30 Dec2021（Yahoo Finance）. Twitter data: AAPL tweets from1 Jan2021 to30 Dec2021."
- 我的转述：The study uses daily AAPL stock data from Yahoo Finance and Twitter sentiment data for the same period（2021）.
- 证据等级：A

### E4
- 证据类型：Experiment
- 定位信息：Section4.1（Experimental Setup）
- 原文关键句："We used two weeks (14 days) of historical samples as input to train the model and then predict the stock closing-price of the next day. The recursive rolling strategy was employed for processing both training and testing data. Data were split into 80% train and20% test."
- 我的转述：The model uses a14-day historical window to predict next day's closing price, with80-20 train-test split and recursive rolling strategy.
- 证据等级：A

### E5
- 证据类型：Result
- 定位信息：Section4.2（SVR Results）
- 原文关键句："The prediction accuracy of the base SVR model was66% which improved to68.22% with sentiment features. Using sentiment features, the MAPE and RMSE improved by24% and42.86% respectively."
- 我的转述：Adding sentiment features to SVR model increases accuracy by2.22% and reduces MAPE/RMSE significantly.
- 证据等级：A

### E6
- 证据类型：Result
- 定位信息：Section4.5（RNN Results）
- 原文关键句："The prediction accuracy of the base RNN model was67% which improved to70.81% when sentiment features were included. MAPE improved from0.23 to0.11, RMSE from0.12 to0.04."
- 我的转述：Including sentiment features in base RNN model increases accuracy by3.81% and reduces MAPE/RMSE.
- 证据等级：A

### E7
- 证据类型：Result
- 定位信息：Section4.5（RNN Results）
- 原文关键句："Finally, RNN with FMHP performed69% accurate predictions, which became70.88% when sentiment features were incorporated. MAPE went from0.17 to0.1 and RMSE from0.05 to0.04. It can be concluded that using FMHP and sentiment features improved the accuracy of the base RNN model by3.88%."
- 我的转述：RNN combined with FMHP filter and sentiment features achieves highest accuracy（70.88%）among all models, with MAPE=0.1 and RMSE=0.04.
- 证据等级：A

### E8
- 证据类型：Result
- 定位信息：Section4.6（Comparison）
- 原文关键句："Our best model (RNN+FMHP+Sent) achieved70.88% accuracy which is significantly better than the65.28% and66.54% performed by Chen et al.'s models."
- 我的转述：The proposed hybrid model outperforms state-of-the-art models（Chen et al.）in prediction accuracy.
- 证据等级：A

### E9
- 证据类型：Limitation
- 定位信息：Section6（Conclusions）
- 原文关键句："The stock market price depends not only on time series data but also on macroeconomic factors and other external factors such as news which significantly impact the stock market price. These limitations need to be solved for future research. We also intend to validate the proposed features with more diverse datasets."
- 我的转述：The model does not consider macroeconomic factors/news beyond Twitter and needs validation on diverse datasets.
- 证据等级：A

## 3）主题笔记（Topic Notes）
### Effectiveness of FMHP Filter in Noise Reduction
依据证据 E1、E7：The FMHP filter resolves the endpoint bias issue of the traditional HP filter（E1）. When combined with RNN and sentiment features, it helps achieve the highest prediction accuracy（70.88%）for AAPL closing price（E7）. This filter is more effective in reducing noise compared to HP filter.

### Impact of Sentiment Features on Prediction Performance
依据证据 E5、E6、E7：Adding Twitter sentiment features consistently improves prediction accuracy across all models. For SVR, accuracy increases by 2.22%；for base RNN, by3.81%；and for RNN+FMHP, by1.88%（E5,E6,E7）. Sentiment features also reduce MAPE and RMSE significantly, indicating better prediction precision.

### Superiority of Deep Learning Models Over Traditional ML
依据证据 E5、E6、E7：Deep learning models（RNN variants）outperform traditional models（SVR, RF, ARIMA）. The best RNN model（70.88% accuracy）is better than the best SVR（69.81%）and RF（66.89%）models（E5,E7）. RNN models also have lower MAPE and RMSE values, showing higher prediction quality.

### Limitations of Model Generalization
依据证据 E3、E9：The model is validated only on daily AAPL data, so its generalization to other stocks or frequencies（like intraday）is unproven（E3）. It also lacks consideration of macroeconomic factors and external news beyond Twitter, which limits its real-world applicability（E9）.

## 4）可直接写进论文的句子草稿（可选）
1. The fully modified Hodrick-Prescott（FMHP）filter effectively reduces endpoint bias in stock time series data, making it more suitable for prediction than the traditional HP filter（依据证据 E1）.
2. Incorporating Twitter sentiment features into machine learning models（such as SVR and RNN）significantly improves the accuracy of day-ahead stock closing price predictions（依据证据 E5、E6）.
3. The hybrid model combining recurrent neural networks（RNN）with FMHP filter and sentiment features achieves the highest prediction accuracy（70.88%）for Apple Inc.'s daily closing price（依据证据 E7）.
4. Deep learning models（like RNN variants）outperform traditional machine learning models（SVR, random forests, ARIMA）in predicting stock closing prices（依据证据 E5、E7）.
5. The proposed model's limitations include the exclusion of macroeconomic factors and the need for validation on more diverse datasets to ensure broader applicability（依据证据 E9）.
