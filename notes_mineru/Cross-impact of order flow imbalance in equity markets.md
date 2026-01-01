# Cross-impact of order flow imbalance in equity markets

## 1）元数据卡（Metadata Card）
- 标题：Cross-impact of order flow imbalance in equity markets
- 作者：Rama Cont, Mihai Cucuringu & Chao Zhang
- 年份：2023
- 期刊/会议：Quantitative Finance
- DOI/URL：10.1080/14697688.2023.2236159
- 适配章节（映射到论文大纲，写 1–3 个）：Section2-Data and Variables, Section3-Contemporaneous Cross-impact, Section4-Forecasting Future Returns
- 一句话可用结论（必须含证据编号）：Integrated OFI improves contemporaneous return explanation over best-level OFI (E3), and lagged cross-asset OFIs enhance short-term future return forecasting (E7, E8)
- 可复用证据（列出最关键 3–5 条 E 编号）：E1, E3, E7, E8, E9
- 市场/资产（指数/个股/期货/加密等）：Top100 components of S&P500 index（equity）
- 数据来源（交易所/数据库/公开数据集名称）：Nasdaq ITCH from LOBSTER
- 频率（tick/quote/trade/分钟/日等）：minute
- 预测目标（方向/收益/价格变化/波动/冲击等）：future returns（price change）
- 预测视角（点预测/区间/分类/回归）：regression
- 预测步长/窗口（horizon）：1 minute to30 minutes
- 关键特征（尤其 OFI/LOB/交易特征；列出原文术语）：best-level OFI, multi-level OFI, integrated OFI, lagged OFIs（cross-asset and own）
- 模型与训练（模型族/损失/训练方式/在线或离线）：OLS（self impact）, LASSO（sparse cross-impact；offline, rolling window）
- 评价指标（AUC/Accuracy/MAE/RMSE/收益等）：adjusted R²（in/out sample）, annualized PnL
- 主要结论（只写可证据支撑的，逐条列点）：
  1. Integrated OFI has higher explanatory power for contemporaneous returns than best-level OFI（E3）
  2. Cross-asset best-level OFIs improve contemporaneous return explanation over own best-level OFI, but cross-asset integrated OFIs do not add to own integrated OFI（E3, E5）
  3. Lagged cross-asset OFIs enhance future return forecasting（statistical and economic）at short horizons, but predictability decays quickly（E7, E8, E9）
- 局限与适用条件（只写可证据支撑的）：
  1. Integrated OFIs do not explicitly account for level information（distance to mid-price）of multi-level OFIs（E10）
  2. The study excludes first/last30 minutes of trading day（E6）
  3. Economic gain analysis ignores trading costs（Table11 note）
- 与本论文题目“OFI + 美股指数/代表性个股 + 短期预测”的关联（用证据编号支撑）：The paper uses S&P500 top100 stocks（美股代表性个股）, focuses on OFI（best-level, multi-level, integrated）for short-term（1min-30min）return prediction, showing cross-asset OFIs improve short-term forecasts（E7, E8, E9）

## 2）可追溯证据条目（Evidence Items）
### E1
- 证据类型：定义
- 定位信息：Section2.1.3（Integrated OFI）
- 原文关键句："we propose an integrated version of OFIs via Principal Components Analysis (PCA) as shown in equation (4), which only preserves the first principal component. We further normalize the first principal component by dividing by its l₁ norm so that the weights of multi-level OFIs in constructing integrated OFIs sum to1"
- 我的转述：Integrated OFI is the first principal component of top10-level OFIs, normalized by l₁ norm to ensure weights sum to1.
- 证据等级：A

### E2
- 证据类型：定义
- 定位信息：Section2.1.1（Best-level OFI）
- 原文关键句："Best-level OFI calculates the accumulative OFIs at the best bid/ask side during a given time interval... defined as OFIᵢ,ₜ¹ʰ = sumₙ (OFᵢ,ₙ¹ᵇ - OFᵢ,ₙ¹ᵃ)"
- 我的转述：Best-level OFI is the cumulative difference between bid and ask order flows at the best LOB level over a time interval.
- 证据等级：A

### E3
- 证据类型：结果
- 定位信息：Table3（In-sample performance for contemporaneous returns）
- 原文关键句："PI^I displays higher and more consistent explanation power, with an average adjusted R² value of87.14% and a standard deviation of9.16%, indicating the effectiveness of our integrated OFIs. The increments of the in-sample R² are smaller when using integrated OFIs (0.71%) compared to best-level OFIs (2.71%)"
- 我的转述：Integrated OFI model（PI^I）has higher in-sample R²（87.14%）than best-level OFI model（PI^[1],71.16%）; cross-asset integrated OFIs add only0.71% to in-sample R² vs cross-asset best-level OFIs adding 2.71%.
- 证据等级：A

### E4
- 证据类型：结果
- 定位信息：Table4（Coefficients in cross-impact models）
- 原文关键句："The frequency of a cross-asset integrated OFI variable selected by CI^I is around1/2 of its counterpart in CI^[1]. The cross-impact coefficients in CI^I are about1/3 in scale of their counterparts in CI^[1]"
- 我的转述：Cross-asset integrated OFIs are selected less frequently（8.29% vs17.34%）and have smaller coefficients（1.6e-3 vs4.5e-3）than cross-asset best-level OFIs in cross-impact models.
- 证据等级：A

### E5
- 证据类型：结果
- 定位信息：Table5（Out-of-sample performance for contemporaneous returns）
- 原文关键句："When involving multi-level or integrated OFIs, the performance of CI^I is slightly worse than PI^I, indicating that the cross-impact model with integrated OFIs cannot provide extra explanatory power to the price impact model with integrated OFIs"
- 我的转述：Out-of-sample R² of CI^I（83.62%）is slightly lower than PI^I（83.83%）, so cross-asset integrated OFIs do not improve out-of-sample contemporaneous return explanation over own integrated OFI.
- 证据等级：A

### E6
- 证据类型：方法
- 定位信息：Section3.2（Empirical results）
- 原文关键句："We exclude the first and last30 minutes of the trading day due to the increased volatility near the opening and closing sessions"
- 我的转述：The study excludes first/last30 minutes of trading day to avoid opening/closing volatility.
- 证据等级：A

### E7
- 证据类型：结果
- 定位信息：Table8（Out-of-sample performance for one-minute-ahead returns）
- 原文关键句："The cross-impact models FCI^[1] (respectively, FCI^I, CAR) achieve higher out-of-sample R² statistics compared to the price impact models FPI^[1] (respectively, FPI^I, AR)"
- 我的转述：Cross-impact models（FCI^[1], FCI^I）have higher out-of-sample R² than price impact models（FPI^[1], FPI^I）for1-minute-ahead return forecasting.
- 证据等级：A

### E8
- 证据类型：结果
- 定位信息：Table11（Economic performance of forecast-implied strategy）
- 原文关键句："Portfolios based on forecasts of the forward-looking cross-impact model outperform those based on forecasts of the forward-looking price impact model. For example, FCI^[1] has annualized PnL of0.43 vs FPI^[1] of 0.21"
- 我的转述：Cross-impact models yield higher annualized PnL（FCI^[1]:0.43, FCI^I:0.39）than price impact models（FPI^[1]:0.21, FPI^I:0.23）for1-minute-ahead return forecasting.
- 证据等级：A

### E9
- 证据类型：结果
- 定位信息：Figure10（Annualized PnL as function of forecasting horizon）
- 原文关键句："Superior forecasting ability arises from cross-asset terms at short horizons. However, the PnL of cross-asset models declines more quickly over longer horizons"
- 我的转述：Cross-asset models' PnL peaks at short horizons（1min）and decays rapidly; by30min, their PnL is similar to price impact models.
- 证据等级：A

### E10
- 证据类型：局限
- 定位信息：Section4.4（Discussion about predictive cross-impact）
- 原文关键句："The integrated OFIs do not explicitly take into account the level information (distance of a given level to the best bid/ask) of multi-level OFIs, and are agnostic to different sizes resting at different levels"
- 我的转述：Integrated OFIs lack explicit consideration of level distance to mid-price or size differences across LOB levels, limiting their predictive power for future returns.
- 证据等级：A

## 3）主题笔记（Topic Notes）
### Integrated OFI vs Best-level OFI for Contemporaneous Returns
依据E1, E3：Integrated OFI, constructed via PCA on multi-level OFIs and normalized by l₁ norm, has significantly higher explanatory power for contemporaneous returns than best-level OFI. The in-sample R² of integrated OFI model is 87.14% vs best-level OFI model's71.16%. Cross-asset integrated OFIs add minimal value（0.71% R²）compared to cross-asset best-level OFIs（2.71% R²）, indicating integrated OFI captures most relevant information from multi-level and cross-asset sources.

### Cross-impact of OFIs on Contemporaneous Returns
依据E3, E4, E5：Cross-asset best-level OFIs improve both in-sample and out-of-sample contemporaneous return explanation over own best-level OFI. However, cross-asset integrated OFIs are selected less frequently（8.29% vs17.34%）and have smaller coefficients than cross-asset best-level OFIs, and do not improve out-of-sample performance over own integrated OFI. This suggests integrated OFI already incorporates cross-asset information indirectly.

### Lagged Cross-asset OFIs for Future Return Forecasting
依据E7, E8, E9：Lagged cross-asset OFIs enhance future return forecasting at short horizons（1min）. Cross-impact models achieve higher out-of-sample R² and annualized PnL than price impact models. However, this predictability decays quickly—by30min, cross-asset models' PnL is similar to price impact models, indicating short-term nature of cross-impact.

### Limitations of Integrated OFI
依据E10：Integrated OFIs do not explicitly account for level distance to mid-price or size differences across LOB levels, which may limit their predictive power for future returns. This is reflected in forward-looking models where integrated OFI models do not significantly outperform best-level OFI models.

## 4）可直接写进论文的句子草稿（可选）
1. Integrated OFI, constructed via PCA on multi-level OFIs and normalized by l₁ norm, exhibits higher explanatory power for contemporaneous returns than best-level OFI（E1, E3）.
2. Cross-asset best-level OFIs improve contemporaneous return explanation over own best-level OFI, but cross-asset integrated OFIs do not add to own integrated OFI's explanatory power（E3, E5）.
3. Lagged cross-asset OFIs enhance short-term future return forecasting, as evidenced by higher out-of-sample R² and annualized PnL of cross-impact models（E7, E8）.
4. The predictive power of lagged cross-asset OFIs decays rapidly over longer horizons, with their performance becoming similar to price impact models by30 minutes（E9）.
5. Integrated OFIs have a limitation: they do not explicitly consider level distance to mid-price or size differences across LOB levels（E10）.
