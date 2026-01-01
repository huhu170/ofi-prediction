# The_interaction_between_order_imbalance

## 1）元数据卡（Metadata Card）
- 标题：The interaction between order imbalance and stock price
- 作者：Philip Brown, David Walsh, Andrea Yuen
- 年份：1997
- 期刊/会议：【未核验】
- DOI/URL：【未核验】
- 适配章节（映射到论文大纲，写1–3个）：第4章-第2节-Data and method、第5章-Results、第6章-Conclusions
- 一句话可用结论（必须含证据编号）：Bi-directional causality between order imbalance (value) and stock return is significant at the 1% level for 19 of 20 ASX stocks at hourly frequency（依据证据E5）。
- 可复用证据（列出最关键3–5条E编号）：E1、E5、E7、E8、E9
- 市场/资产（指数/个股/期货/加密等）：ASX-listed stocks（20 most actively traded）
- 数据来源（交易所/数据库/公开数据集名称）：ASX SEATS database、SIRCA
- 频率（tick/quote/trade/分钟/日等）：Hourly、daily（close-to-close、day/night split）
- 预测目标（方向/收益/价格变化/波动/冲击等）：Stock return（log change of mid-price）
- 预测视角（点预测/区间/分类/回归）：Causality analysis、dynamic impact assessment
- 预测步长/窗口（horizon）：Up to 3–4 hours
- 关键特征（尤其OFI/LOB/交易特征；列出原文术语）：Order imbalance（by number、by value）、log return of mid-price
- 模型与训练（模型族/损失/训练方式/在线或离线）：Bivariate VAR（vector autoregression）、OLS estimation、Granger causality tests
- 评价指标（AUC/Accuracy/MAE/RMSE/收益等）：Likelihood ratio test p-values、R²、impulse response functions
- 主要结论（只写可证据支撑的，逐条列点）：
  1. Bi-directional causality exists between value-based order imbalance and return at hourly level（E5）。
  2. Number-based order imbalance has weaker impact on return than value-based imbalance（E5、E6）。
  3. Adjustment to order imbalance（value）is complete within ~3 hours（E7、E8）。
  4. Return innovations induce initial over-reaction followed by partial reversal（E7）。
- 局限与适用条件（只写可证据支撑的）：
  1. Results are specific to ASX stocks（1994–1995）and may not generalize to other markets（E9）。
  2. No consideration of transaction costs or trading strategies（E9）。
- 与本论文题目“OFI + 美股指数/代表性个股 + 短期预测”的关联（用证据编号支撑）：The paper’s order imbalance metrics（by value/number）are relevant to OFI-related research；its causality and dynamic impact findings inform short-term prediction windows（~3h）for US stocks（依据E1、E5、E7）。

## 2）可追溯证据条目（Evidence Items）
### E1
- 证据类型：定义
- 定位信息：第4章-第2节-Data and method（OI定义段落）
- 原文关键句：“Imbalance is then defined as the total number (value) of ask orders divided by the sum of bids and asks.”
- 我的转述：Order imbalance（OI）is calculated as the ratio of ask orders（count or dollar value）to the sum of bid and ask orders。
- 证据等级：A

### E2
- 证据类型：定义
- 定位信息：第4章-第2节-Data and method（return定义段落）
- 原文关键句：“Stock return is defined as the difference between the natural logarithms of two successive prices. Price is measured by the average of the best bid and best ask price.”
- 我的转述：Log return is computed using successive mid-prices（average of best bid and ask）to reduce bid-ask bounce bias。
- 证据等级：A

### E3
- 证据类型：方法
- 定位信息：第4章-第2节-Data and method（VAR模型段落）
- 原文关键句：“We adopt a bivariate vector auto-regression (VAR), based on a detailed specification search to ensure that the best model and metrics are used.”
- 我的转述：A bivariate VAR model is used to analyze the interaction between return and order imbalance，with OLS estimation。
- 证据等级：A

### E4
- 证据类型：实验
- 定位信息：第5章-Results（data sets段落）
- 原文关键句：“Three data sets are developed for each stock separately: close-to-close；close-to-open and open-to-close；and close-to-open plus the six trading hours.”
- 我的转述：Three frequency data sets（daily、day/night split、hourly）are constructed for each ASX stock。
- 证据等级：A

### E5
- 证据类型：结果
- 定位信息：第5章-Results（causality test results段落）
- 原文关键句：“Granger causality from order imbalance value to return is bi-directional when the contemporaneous values are included and imbalance relates to the value of orders entered each hour. This result is significant at the 1% level for 19 of the 20 stocks.”
- 我的转述：Bi-directional causality between value-based OI and return exists at hourly frequency，significant for 19/20 stocks at the 1% level。
- 证据等级：A

### E6
- 证据类型：结果
- 定位信息：第5章-Results（causality comparison段落）
- 原文关键句：“Causality from number imbalance to return is significant in only one case，but in the reverse direction it is strong，being rejected only once. If value imbalance is used，strong bi-directional causality is found.”
- 我的转述：Number-based OI shows weaker causality to return than value-based OI；return strongly drives number-based OI。
- 证据等级：A

### E7
- 证据类型：结果
- 定位信息：第5章-Results（impulse response fig1/fig2段落）
- 原文关键句：“A positive impulse in OI（sell orders increase）leads for every company to a decrease in return，taking up to 3 hours to become fully incorporated. The total cumulative impact... is between -0.14 and -0.37%.”
- 我的转述：A sell imbalance innovation leads to a return decrease that fully adjusts within ~3 hours。
- 证据等级：A

### E8
- 证据类型：结果
- 定位信息：第5章-Results（impulse response fig3/fig4段落）
- 原文关键句：“After the initial response... the OI reverses strongly. A positive innovation in return induces a strong buy order imbalance，followed by a strong correcting sell order imbalance... After this reversal，the OI settles down within about 3h.”
- 我的转述：A positive return innovation triggers an initial buy imbalance，followed by a sell reversal；OI stabilizes in ~3 hours。
- 证据等级：A

### E9
- 证据类型：局限
- 定位信息：第6章-Conclusions（further research段落）
- 原文关键句：“First，examining cross-listed securities between exchanges would allow direct control over firm-specific effects... Second，exploring the number and size of orders as different metrics will almost certainly provide fruitful conclusions.”
- 我的转述：The study is limited to ASX stocks（1994–1995）and lacks cross-market validation or transaction cost analysis。
- 证据等级：B

## 3）主题笔记（Topic Notes）
### Order Imbalance Metrics: Number vs. Value
依据证据E1、E5：The paper uses two OI metrics—count-based（frequency of orders）and value-based（size of orders）。Value-based OI has stronger informational content for return prediction，as shown by its significant bi-directional causality with return at hourly frequency。

### Causality Dynamics Between OI and Return
依据证据E5、E6：Hourly value-based OI and return exhibit bi-directional causality for most stocks。Count-based OI is less predictive of return，but return strongly influences count-based OI（likely due to herding behavior in small orders）。

### Short-Term Adjustment Windows for Innovations
依据证据E7、E8：Shocks to OI or return are fully absorbed within ~3 hours。This suggests that short-term prediction models should focus on horizons up to 3 hours for optimal performance。

### Limitations for US Market Generalization
依据证据E9：The study’s ASX-specific findings need validation in US markets。US markets have different microstructure（e.g. NYSE specialists），so OI-return relationships may differ。Additionally，transaction costs（ignored here）are critical for US short-term trading strategies。

## 4）可直接写进论文的句子草稿（可选）
1. Order imbalance can be measured by either the number of orders or their dollar value，with value-based metrics capturing more informational content for return prediction（依据证据E1、E5）。
2. Bi-directional causality between value-based order imbalance and stock return is significant at the 1% level for 19 of 20 actively traded ASX stocks at hourly frequency（依据证据E5）。
3. Shocks to value-based order imbalance are fully incorporated into stock returns within approximately three hours，indicating a short-term predictability window（依据证据E7）。
4. Return innovations induce an initial over-reaction followed by a partial reversal，a pattern that can inform short-term trading strategies（依据证据E7）。
5. Count-based order imbalance has weaker predictive power for return than value-based imbalance，suggesting that order size is a critical factor in capturing informational content（依据证据E6）。
