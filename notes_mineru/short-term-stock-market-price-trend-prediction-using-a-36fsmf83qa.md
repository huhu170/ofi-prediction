# short-term-stock-market-price-trend-prediction-using-a-36fsmf83qa

## 1）元数据卡（Metadata Card）
- 标题：Short-term stock market price trend prediction using a comprehensive deep learning system
- 作者：Jingyi Shen, M. Omair Shafiq
- 年份：2020
- 期刊/会议：SpringerOpen Journal (implied by publisher's note)
- DOI/URL：https://curve.carleton.ca/52e9187a-7f71-48ce-bdfe-e3f6a420e31a
- 适配章节（映射到论文大纲，写 1–3 个）：Methods, Results, Discussion
- 一句话可用结论（必须含证据编号）：The proposed FE+RFE+PCA+LSTM system achieves binary accuracy of0.9325 for bi-weekly stock price trend prediction, outperforming Random Forest (Evidence E2, E3).
- 可复用证据（列出最关键 3–5 条 E 编号）：E1, E2, E3, E4, E5
- 市场/资产（指数/个股/期货/加密等）：Chinese stock market (3558 stocks)
- 数据来源（交易所/数据库/公开数据集名称）：Tushare API, Sina Finance, SWS Research website
- 频率（tick/quote/trade/分钟/日等）：Daily
- 预测目标（方向/收益/价格变化/波动/冲击等）：Short-term price trend direction (up/down)
- 预测视角（点预测/区间/分类/回归）：Binary classification
- 预测步长/窗口（horizon）：2-day, weekly (5-day), bi-weekly (10-day)
- 关键特征（尤其 OFI/LOB/交易特征；列出原文术语）：Technical indices (SMA10, MACD, CCI24, MTM10, ROC10, RSI5, WNR9, SLOWK, SLOWD, ADOSC, AR26, BR26, VR26, BIAS20); FE-extended via max-min scaling, polarizing, fluctuation percentage
- 模型与训练（模型族/损失/训练方式/在线或离线）：LSTM (2 layers); loss=MAE; optimizer=Adam; offline training
- 评价指标（AUC/Accuracy/MAE/RMSE/收益等）：Binary accuracy, F1 score, MSE, MAE, TPR, TNR, FPR, FNR
- 主要结论（只写可证据支撑的，逐条列点）：
  1. FE+RFE+PCA+LSTM outperforms traditional models (e.g., Random Forest) with higher binary accuracy (0.93 vs0.88) (Evidence E3)
  2. Feature extension improves true positive/negative rates by7%/10% (Evidence E5)
  3. PCA reduces training time by ~36.8% while maintaining accuracy (Evidence E4)
  4. Optimal prediction horizons:2-day, weekly, bi-weekly (Evidence E6)
- 局限与适用条件（只写可证据支撑的）：
  1. Validated only on Chinese stock market; generalization to US markets untested (Evidence E7)
  2. No sentiment analysis integration from text data (Evidence E8)
- 与本论文题目“OFI + 美股指数/代表性个股 + 短期预测”的关联（用证据编号支撑）：The paper focuses on short-term prediction (Evidence E6) but lacks OFI/LOB features and US market validation (Evidence E7)

## 2）可追溯证据条目（Evidence Items）
### E1
- 证据类型：方法
- 定位信息：Section4 (Proposed solution)
- 原文关键句："The proposed solution is comprehensive as it includes pre-processing of the stock market dataset, utilization of multiple feature engineering techniques, combined with a customized deep learning based system for stock market price trend prediction."
- 我的转述：The system integrates data preprocessing, feature engineering (FE+RFE+PCA), and a customized LSTM model for short-term stock price trend prediction.
- 证据等级：A

### E2
- 证据类型：结果
- 定位信息：Table9
- 原文关键句："LSTM trained on29 features: Binary accuracy=0.9325, F1 score=0.9323"
- 我的转述：The LSTM model with29 selected features achieves 0.9325 binary accuracy and0.9323 F1 score for bi-weekly prediction.
- 证据等级：A

### E3
- 证据类型：结果
- 定位信息：Table7
- 原文关键句："Proposed model: F1 score=0.93, Binary accuracy=0.93; RAF: F1 score=0.88, Binary accuracy=0.88"
- 我的转述：The proposed model outperforms Random Forest with higher F1 score (0.93 vs0.88) and binary accuracy (0.93 vs0.88).
- 证据等级：A

### E4
- 证据类型：结果
- 定位信息：Discussion section (PCA effectiveness)
- 原文关键句："PCA has significantly improved the training efficiency of the LSTM model by36.8%"
- 我的转述：PCA reduces LSTM training time by ~36.8% while maintaining binary accuracy (0.9193 vs original0.9325).
- 证据等级：A

### E5
- 证据类型：结果
- 定位信息：Section Results (Feature extension and RFE)
- 原文关键句："Both precisions of true positive and true negative have been improved by7% and10% respectively"
- 我的转述：Feature extension increases true positive rate by7% and true negative rate by10% compared to original features.
- 证据等级：A

### E6
- 证据类型：结果
- 定位信息：Section Results (Term length)
- 原文关键句："there are three-term lengths that are most sensitive to the indices we selected... they are n={2,5,10}"
- 我的转述：Optimal prediction horizons are2-day, weekly (5-day), and bi-weekly (10-day).
- 证据等级：A

### E7
- 证据类型：局限
- 定位信息：Section Conclusion
- 原文关键句："the policies of different countries might impact the model performance, which needs further research to validate."
- 我的转述：The model's effectiveness is limited to Chinese stock market; generalization to US markets requires testing.
- 证据等级：B

### E8
- 证据类型：局限
- 定位信息：Section Conclusion
- 原文关键句："by combining latest sentiment analysis techniques with feature engineering and deep learning model, there is also a high potential to develop a more comprehensive prediction system which is trained by diverse types of information such as tweets, news, and other text-based data."
- 我的转述：The model lacks sentiment analysis integration from text data (e.g., tweets, news).
- 证据等级：B

## 3）主题笔记（Topic Notes）
### Feature Engineering's Impact on Accuracy
依据证据E1、E5：The system uses FE to expand technical indices with investor techniques, improving true positive/negative rates by7%/10%. RFE selects effective features, and PCA reduces dimensionality while preserving accuracy.

### Model Performance Comparison
依据证据E2、E3：The LSTM model achieves0.9325 binary accuracy for bi-weekly prediction, outperforming Random Forest (0.88). Deep learning combined with feature engineering is more effective for short-term trends.

### Optimal Prediction Horizons
依据证据E6：The system performs best on2-day, weekly, and bi-weekly horizons, as these are most sensitive to selected indices. Longer/shorter horizons may have higher noise.

### PCA's Efficiency Gain
依据证据E4：PCA reduces training time by~36.8% without significant accuracy loss, making the system efficient for large datasets like 3558 Chinese stocks.

### Limitations of the System
依据证据E7、E8：The model is only validated on Chinese stocks; it also lacks sentiment analysis from text data.

## 4）可直接写进论文的句子草稿（可选）
1. The FE+RFE+PCA+LSTM system achieves binary accuracy of0.9325 for bi-weekly stock price trend prediction, outperforming Random Forest (Evidence E2, E3).
2. Feature extension using investor-inspired techniques improves true positive and negative rates by7% and10% respectively (Evidence E5).
3. PCA significantly reduces the LSTM model's training time by~36.8% while maintaining a high binary accuracy of0.9193 (Evidence E4).
4. The optimal prediction horizons for the system are2-day, weekly, and bi-weekly, as these lengths are most sensitive to the selected technical indices (Evidence E6).
5. The model's effectiveness is limited to the Chinese stock market, and further research is needed to validate its performance on US markets (Evidence E7).
