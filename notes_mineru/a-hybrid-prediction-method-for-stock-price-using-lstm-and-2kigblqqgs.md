# a-hybrid-prediction-method-for-stock-price-using-lstm-and-2kigblqqgs

## 1）元数据卡（Metadata Card）
- 标题：A Hybrid Prediction Method for Stock Price Using LSTM and Ensemble EMD
- 作者：Yang Yujun, Yang Yimei, Xiao Jianhua
- 年份：2020
- 期刊/会议：【未核验】
- DOI/URL：【未核验】
- 适配章节（映射到论文大纲，写 1–3 个）：Section4（Methodology of hybrid LSTM-EEMD）、Section7（Experimental analysis of real-world stock data）、Section3（Preliminaries on LSTM and EEMD）
- 一句话可用结论（必须含证据编号）：The hybrid LSTM-EEMD prediction method achieves better performance than traditional LSTM and LSTM-EMD methods on real-world stock index data（依据证据 E5、E7）
- 可复用证据（列出最关键 3–5 条 E 编号）：E1、E2、E3、E5、E7
- 市场/资产（指数/个股/期货/加密等）：Real-world stock indices（SP500、HSI、DAX、ASX）、artificial sin data
- 数据来源（交易所/数据库/公开数据集名称）：Yahoo Finance（real stock indices）、artificial data generated via sin function
- 频率（tick/quote/trade/分钟/日等）：Daily
- 预测目标（方向/收益/价格变化/波动/冲击等）：Future stock price
- 预测视角（点预测/区间/分类/回归）：Regression
- 预测步长/窗口（horizon）：Short-term（e.g., SP500 test length=2320）
- 关键特征（尤其 OFI/LOB/交易特征；列出原文术语）：Daily transaction volume、highest price、lowest price、closing price、opening price；daily logarithmic return；standardized daily return
- 模型与训练（模型族/损失/训练方式/在线或离线）：Hybrid model（LSTM + EEMD）；训练方式：Offline；损失函数：【未核验】
- 评价指标（AUC/Accuracy/MAE/RMSE/收益等）：RMSE、MAE、R²
- 主要结论（只写可证据支撑的，逐条列点）：1）The LSTM-EEMD method outperforms LSTM-EMD on real-world stock data（依据 E5）；2）BARDR and LSTM are more suitable for unstable stock sequences than SVR/KNR（依据 E6）；3）The LSTM-EMD method is effective for artificial sin data（依据 E4）
- 局限与适用条件（只写可证据支撑的）：The proposed methods may have unexpected results on highly orderly time series（依据 E8）
- 与本论文题目“OFI + 美股指数/代表性个股 + 短期预测”的关联（用证据编号支撑）：The paper provides a hybrid LSTM-based short-term prediction method for US stock indices（SP500）using daily features，which can inform model structure design for OFI+美股 index prediction tasks（依据 E1、E7）

## 2）可追溯证据条目（Evidence Items）
### E1
- 证据类型：方法
- 定位信息：Section4（Methodology: Steps of LSTM-EEMD）
- 原文关键句："Our proposed hybrid LSTM-EEMD prediction method first uses the EEMD to decompose the stock index sequences into a few simple stable subsequences. Then, the predict result of each subsequence is predicted by the LSTM method. Finally, the LSTM-EEMD obtains the final prediction result by fusing all LSTM prediction results of several stock index subsequences."
- 我的转述：The LSTM-EEMD hybrid method involves three core steps：decomposing original stock sequences into stable subsequences via EEMD，predicting each subsequence with LSTM，and fusing results to get final predictions.
- 证据等级：A

### E2
- 证据类型：实验
- 定位信息：Section5.2（Real-World Experimental Data）
- 原文关键句："we collected stock indices data in the real-world stock field as experiment data from Yahoo Finance. To obtain more objective experimental results，we choose 4 stock indices from different countries or regions: ASX, DAX, HSI, and SP500."
- 我的转述：Real-world stock index data（ASX、DAX、HSI、SP500）for experiments is collected from Yahoo Finance.
- 证据等级：A

### E3
- 证据类型：实验
- 定位信息：Section4（Methodology: Evaluation criteria）
- 原文关键句："We use three evaluation criteria of the RMSE, MAE, and R² to evaluate the LSTM-EEMD hybrid prediction method."
- 我的转述：The hybrid method is evaluated using RMSE（Root Mean Square Error）、MAE（Mean Absolute Error）、and R²（Coefficient of Determination）metrics.
- 证据等级：A

### E4
- 证据类型：结果
- 定位信息：Section7.1（Artificial Simulation Data Analysis）
- 原文关键句："The LSTM-EMD method has the best prediction effect，indicating that the method we proposed is effective and can improve the prediction effect of the experiment."
- 我的转述：For artificial sin data，the LSTM-EMD method achieves the best prediction performance among the proposed hybrid methods.
- 证据等级：B

### E5
- 证据类型：结果
- 定位信息：Section7.2（Real Data Analysis）
- 原文关键句："The prediction results of the LSTM-EEMD prediction method in the four sequence data are better than the LSTM-EMD prediction method."
- 我的转述：On real-world stock index data（SP500、HSI、DAX、ASX），the LSTM-EEMD method outperforms the LSTM-EMD method．
- 证据等级：A

### E6
- 证据类型：结果
- 定位信息：Section6（Experiment Results of Other Methods）
- 原文关键句："The BARDR and LSTM methods can predict stock time series，so the BARDR and LSTM are more suitable to predict sequence predictions with unstable and irregular changes."
- 我的转述：BARDR and LSTM methods are more suitable for predicting unstable and irregular stock time series compared to SVR and KNR.
- 证据等级：B

### E7
- 证据类型：实验
- 定位信息：Section5.2（Real-World Experimental Data: SP500）
- 原文关键句："The SP500 is a US stock market index. This stock index is a synthesis of the stock indexes of 500 companies listed on NASDAQ and NYSE. The datasets of every stock index include five daily properties: transaction volume，highest price，lowest price，closing price，and opening price."
- 我的转述：SP500 is a US stock index covering 500 NASDAQ/NYSE companies，with daily data including transaction volume，highest/lowest price，closing/opening price.
- 证据等级：A

### E8
- 证据类型：局限
- 定位信息：Section8（Conclusion）
- 原文关键句："However，there are some shortcomings. The proposed method has some unexpected effects on the experimental results of time series with very orderly changes."
- 我的转述：The proposed hybrid methods may yield unexpected results when predicting highly orderly time series（e.g., artificial sin data）.
- 证据等级：A

## 3）主题笔记（Topic Notes）
### Hybrid LSTM-EEMD Prediction Method（Steps and Principles）
The LSTM-EEMD hybrid method leverages EEMD's non-stationary data decomposition capability and LSTM's time series prediction strength. It follows three core steps: decomposing original stock sequences into stable subsequences using EEMD，predicting each subsequence with LSTM，and fusing the results to get the final prediction（依据证据 E1）. This approach addresses the complexity of stock price time series by breaking it into manageable parts.

### Data Sources and Preprocessing for Stock Prediction
Real-world stock index data（SP500、HSI、DAX、ASX）is collected from Yahoo Finance，covering daily features like transaction volume，highest/lowest price，closing/opening price（依据证据 E2、E7）. Artificial sin data（length=10000）is used to validate method correctness. Data preprocessing includes calculating daily logarithmic returns and standardizing them to stabilize the data（依据证据 E7）.

### Evaluation Metrics for Stock Price Prediction
The proposed methods are evaluated using three key metrics: RMSE（measures average squared error between predicted and actual values）、MAE（measures average absolute error）、and R²（indicates how well the model explains variance in the data）. Lower RMSE/MAE values and higher R² values indicate better performance（依据证据 E3）.

### Performance Comparison of Hybrid vs Traditional Methods
On real-world stock data，the LSTM-EEMD method outperforms the LSTM-EMD method（依据证据 E5）. Traditional methods like BARDR and LSTM are more effective than SVR and KNR for unstable/irregular stock sequences（依据证据 E6）. For artificial sin data，the LSTM-EMD method shows the best performance among hybrid methods（依据证据 E4）.

### Limitations of the Proposed Methods
A key limitation of the hybrid methods is their unexpected performance on highly orderly time series（e.g., artificial sin data），which may be due to over-decomposition of regular patterns（依据证据 E8）. Future work should address this issue to improve method robustness.

## 4）可直接写进论文的句子草稿（可选）
1. The hybrid LSTM-EEMD prediction method combines EEMD's non-stationary data decomposition capability with LSTM's time series prediction strength，following three core steps: decomposition，prediction，and fusion（依据证据 E1）.
2. Real-world stock index data（including SP500、HSI、DAX、ASX）for experiments is collected from Yahoo Finance，covering daily features like transaction volume and closing price（依据证据 E2、E7）.
3. The LSTM-EEMD method outperforms the LSTM-EMD method on real-world stock index data，as indicated by lower RMSE/MAE values and higher R² scores（依据证据 E5）.
4. A key limitation of the proposed hybrid methods is their unexpected performance on highly orderly time series（e.g., artificial sin data）（依据证据 E8）.
5. BARDR and LSTM methods are more effective than SVR and KNR for predicting unstable and irregular stock time series（依据证据 E6）.
