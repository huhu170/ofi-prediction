# impactandinformationvalueoftrades

## 1）元数据卡（Metadata Card）
- 标题：Is Market Impact a Measure of the Information Value of Trades? Market Response to Liquidity vs. Informed Trades
- 作者：C. Gomes, H. Waelbroeck
- 年份：2013
- 期刊/会议：SSRN Electronic Journal
- DOI/URL：10.2139/ssrn.2291720
- 适配章节（映射到论文大纲，写 1–3 个）：3. Breakeven, Efficiency and Fair Pricing；5. Shape and scale of the impact function for cash flows vs other metaorders；6. Discussion
- 一句话可用结论（必须含证据编号）：Cash flow trades revert almost completely in 2-5 days (无永久影响), while non-cash flow trades revert ~1/3 of peak impact (依据证据 E1, E2)
- 可复用证据（列出最关键 3–5 条 E 编号）：E1, E2, E3, E4, E5
- 市场/资产（指数/个股/期货/加密等）：Stocks（institutional metaorders, Nasdaq-listed, S&P500 beta adjustment）
- 数据来源（交易所/数据库/公开数据集名称）：Proprietary OMS data from institutional clients (Portware LLC)
- 频率（tick/quote/trade/分钟/日等）：Metaorder level（aggregated from trades over minutes to days）
- 预测目标（方向/收益/价格变化/波动/冲击等）：Price reversion, permanent vs temporary market impact
- 预测视角（点预测/区间/分类/回归）：Empirical analysis（非预测模型）
- 预测步长/窗口（horizon）：Reversion up to 10 days（2-5 days for cash flows）
- 关键特征（尤其 OFI/LOB/交易特征；列出原文术语）：Metaorder size (% ADV), participation rate, volatility (90-day annualized), spread (basis points), cash flow label, beta-adjusted returns, peak impact, shortfall, square root impact function
- 模型与训练（模型族/损失/训练方式/在线或离线）：Square root impact model（impact(Q)=2.8*volatility*sqrt(Q/ADV)）, regression analysis, bootstrapped standard errors
- 评价指标（AUC/Accuracy/MAE/RMSE/收益等）：Mean return (bps), beta-adjusted return (bps), bootstrapped standard error, significance (* for p<0.05)
- 主要结论（只写可证据支撑的，逐条列点）：
  1. Cash flow metaorders revert almost completely in 2-5 days, with no permanent impact（E1, E5）
  2. Non-cash flow metaorders revert ~1/3 of peak impact, leading to breakeven after trading costs（E2）
  3. Impact shape/scale during execution are similar for cash and non-cash flow metaorders（E3）
  4. Fair pricing holds for non-cash flow trades（permanent impact equals shortfall）（E4）
  5. Informed trades（multiple PMs, new trades, Nasdaq）deviate from fair pricing（E9）
- 局限与适用条件（只写可证据支撑的）：
  1. Cash flow label not available for all portfolio managers（unlabeled trades may be cash flows）（E7）
  2. Metaorder merging uses 60-minute gap rule（may miss some order flows）（E8）
  3. Data period: July 2009-March 2012（market regime dependency）
- 与本论文题目“OFI + 美股指数/代表性个股 + 短期预测”的关联（用证据编号支撑）：The paper’s analysis of US stock metaorders（Nasdaq-listed）and their short-term reversion patterns provides empirical context for short-term price prediction using order flow-related features（依据证据 E1, E2, E3）


## 2）可追溯证据条目（Evidence Items）
### E1
- 证据类型：结果
- 定位信息：Section 1 Abstract, Section3.1 Breakeven and Efficiency
- 原文关键句："For cash flows, the impact reverts almost completely on average in two to five days. For other metaorders we find that reversion erases about 1/3 of peak impact"
- 我的转述：Cash flow metaorders exhibit almost full price reversion within 2-5 days, while non-cash flow metaorders revert approximately one-third of their peak impact.
- 证据等级：A

### E2
- 证据类型：结果
- 定位信息：Section1 Abstract, Table2a/b
- 原文关键句："For other metaorders we find that reversion erases about 1/3 of peak impact: for each size, price reverts to the average execution price, leaving no immediate profits after accounting for trading costs"
- 我的转述：Non-cash flow metaorders revert ~1/3 of peak impact, leading to no net profits after trading costs (price returns to average execution price).
- 证据等级：A

### E3
- 证据类型：结果
- 定位信息：Section5 Shape and scale..., Figure5
- 原文关键句："We cannot reject the hypothesis that impact is the same for cash flows as for other metaorders, or the hypothesis that peak impact is 1.5 times the estimated average impact"
- 我的转述：During execution, the shape and scale of market impact are statistically indistinguishable between cash flow and non-cash flow metaorders.
- 证据等级：A

### E4
- 证据类型：结果
- 定位信息：Section3.2 Fair Pricing, Figure2
- 原文关键句："Fair pricing holds consistently across all ranges of trade difficulty since there are no significant differences between the average shortfall and the returns to the reversion price"
- 我的转述：For non-cash flow metaorders, fair pricing condition holds—permanent impact equals average execution shortfall.
- 证据等级：A

### E5
- 证据类型：结果
- 定位信息：Section1 Abstract, Table2a/b
- 原文关键句："For cash flows, the impact reverts almost completely on average in two to five days... there is no permanent impact, only information that causes trades"
- 我的转述：Cash flow metaorders have no permanent market impact, as their impact is fully reversed within 2-5 days.
- 证据等级：A

### E6
- 证据类型：方法
- 定位信息：Section2 Data..., Impact adjustment model
- 原文关键句："The square root impact model for the additional positions... impact(Q)=2.8*volatility*sqrt(Q/ADV)"
- 我的转述：Impact of additional metaorders is estimated using a square root function of size (% ADV) and volatility.
- 证据等级：A

### E7
- 证据类型：局限
- 定位信息：Section3.1 Breakeven..., Assumptions
- 原文关键句："The cash flow label is not available for all portfolio managers in our dataset; the subset of trades that are not labeled may include some cash flows"
- 我的转述：Some non-labeled metaorders may be cash flows, as the label isn't universal.
- 证据等级：B

### E8
- 证据类型：方法
- 定位信息：Section2 Data..., Metaorder merging rule
- 原文关键句："We merge orders in the same symbol and side if they were placed on the same day or consecutive open-market days by the same portfolio manager, or placed by another portfolio manager at the same firm within 60 minutes of each other counting only open-market time"
- 我的转述：Metaorders are aggregated from same-firm, same-symbol, same-side orders with a 60-minute gap (or consecutive days for same manager).
- 证据等级：A

### E9
- 证据类型：结果
- 定位信息：Section4 Deviations..., Table4
- 原文关键句："Metaorders aggregated across multiple portfolio managers, new trades and Nasdaq-listed stocks suggest that these trades are more informed than the average"
- 我的转述：Metaorders from multiple portfolio managers, new trades, and Nasdaq-listed stocks are more informed (associated with negative P&L for liquidity providers).
- 证据等级：A


## 3）主题笔记（Topic Notes）
### Metaorder Definition and Data Merging
依据证据 E8: Metaorders are aggregated from same-firm, same-symbol, same-side orders with a 60-minute gap (or consecutive days for same manager). Cash flow metaorders are labeled for inflow/outflow purposes, but not all trades have this label.

### Market Impact Shape and Scale
依据证据 E3, E6: Impact during execution follows a square root function (impact(Q)=2.8*volatility*sqrt(Q/ADV)). Peak impact is ~1.5x shortfall, consistent with square root shape, and this relationship holds for both cash and non-cash flow metaorders.

### Price Reversion and Permanent Impact
依据证据 E1, E2, E5: Cash flow trades revert fully in 2-5 days (no permanent impact). Non-cash flow trades revert ~1/3 of peak impact, leading to breakeven after trading costs (no net profit/loss).

### Fair Pricing Condition
依据证据 E4: For non-cash flow trades, permanent impact equals execution shortfall (fair pricing), meaning the market efficiently incorporates information from these trades into prices.

### Informed vs Uninformed Trades
依据证据 E9: Informed trades (multiple PMs, new trades, Nasdaq-listed) deviate from fair pricing (liquidity providers lose), while price improvement trades, large-cap stocks, and momentum trades are profitable for liquidity providers.

### Limitations of Empirical Analysis
依据证据 E7: Cash flow label not universal, so some non-labeled trades may be cash flows. The square root impact model may not capture all market conditions, and merged metaorders may miss edge cases.


## 4）可直接写进论文的句子草稿（可选）
1. Cash flow metaorders exhibit almost complete price reversion within 2-5 days, indicating no permanent market impact（依据证据 E1, E5）.
2. Non-cash flow metaorders revert approximately one-third of their peak impact, resulting in no immediate profits after accounting for trading costs（依据证据 E2）.
3. During execution, the shape and scale of market impact are statistically indistinguishable between cash flow and non-cash flow metaorders（依据证据 E3）.
4. For non-cash flow trades, the fair pricing condition holds—permanent impact equals the average execution shortfall（依据证据 E4）.
5. Informed trades, such as those from multiple portfolio managers or new trades in Nasdaq-listed stocks, deviate from fair pricing（依据证据 E9）.
6. The square root impact model (impact(Q)=2.8*volatility*sqrt(Q/ADV)) is widely used to estimate the impact of additional metaorders（依据证据 E6）.
