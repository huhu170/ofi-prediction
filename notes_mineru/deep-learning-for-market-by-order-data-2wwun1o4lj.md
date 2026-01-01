# deep-learning-for-market-by-order-data-2wwun1o4lj

## 1）元数据卡（Metadata Card）
- 标题：Deep Learning for Market by Order Data
- 作者：Zihao Zhang, Bryan Lim, Stefan Zohren
- 年份：2021
- 期刊/会议：【未核验】
- DOI/URL：【未核验】
- 适配章节（映射到论文大纲，写1–3个）：Section3（Market by Order Data）、Section4（Methodology）、Section5（Experiments）
- 一句话可用结论（必须含证据编号）：MBO数据可用于短期价格方向预测，且与LOB数据互补，其组合模型能提升预测性能（依据证据E5、E6）
- 可复用证据（列出最关键3–5条E编号）：E1、E3、E5、E6、E8
- 市场/资产（指数/个股/期货/加密等）：伦敦证券交易所（LSE）个股（LLOY、BARC、TSCO、BT、VOD）
- 数据来源（交易所/数据库/公开数据集名称）：伦敦证券交易所（2018年MBO数据）
- 频率（tick/quote/trade/分钟/日等）：Tick（订单级消息频率）
- 预测目标（方向/收益/价格变化/波动/冲击等）：Mid-price方向（上涨/下跌/平稳）
- 预测视角（点预测/区间/分类/回归）：分类（三分类）
- 预测步长/窗口（horizon）：k=20、50、100（Tick步长）
- 关键特征（尤其OFI/LOB/交易特征；列出原文术语）：Side、Action、Normalized price、Normalized size、Normalized change price、Normalized change size
- 模型与训练（模型族/损失/训练方式/在线或离线）：模型族（线性模型、MLP、LSTM、Attention）；损失（Categorical Cross-Entropy）；训练方式（离线，Adam优化器）
- 评价指标（AUC/Accuracy/MAE/RMSE/收益等）：Accuracy、Precision、Recall、F1-score
- 主要结论（只写可证据支撑的，逐条列点）：1）MBO模型与LOB模型预测性能相当（依据E5）；2）MBO与LOB信号相关性低，组合模型提升性能（依据E6）；3）循环模型（LSTM、Attention）优于线性/MLP模型（依据E5、E10）
- 局限与适用条件（只写可证据支撑的）：1）MBO原始消息比LOB快照更难建模（依据E7）；2）未考虑交易成本与实际执行约束（隐含于结论未来工作）
- 与本论文题目“OFI + 美股指数/个股 + 短期预测”的关联（用证据编号支撑）：论文验证了订单级数据（如MBO）可用于短期价格方向预测，与题目中订单流（OFI）和短期预测的核心方向一致（依据E1、E3、E5）

## 2）可追溯证据条目（Evidence Items）
### E1
- 证据类型：定义
- 定位信息：Section3.1（Descriptions of Market by Order Data）
- 原文关键句：“Level3 (L3): L3 is essentially the MBO data introduced in this work and it provides even more information than L2 as it shows non-aggregated bids and asks placed by individual traders.”
- 我的转述：MBO数据（Level3）是非聚合的订单级数据，展示个体交易者的买卖订单，粒度高于LOB（Level2）数据
- 证据等级：A

### E2
- 证据类型：方法
- 定位信息：Section3.2（Data Preprocessing and Normalisation）
- 原文关键句：“Normalised price: (price - mid-price)/(minimum tick size ×100). This calculation transforms price to tick change, representing how many ticks the price is away from the mid-price.”
- 我的转述：归一化价格通过（订单价格-中间价）除以（最小 tick 尺寸×100）计算，将价格转换为相对于中间价的tick变化
- 证据等级：A

### E3
- 证据类型：方法
- 定位信息：Section3.2（Data Preprocessing and Normalisation）
- 原文关键句：“At the end, we remove 'Time stamp' and 'ID', leading to6 features in our feature space: Side, Action, Normalized price, Normalized size, Normalized change price, Normalized change size.”
- 我的转述：MBO数据预处理后保留6个特征：Side、Action、归一化价格、归一化规模、归一化价格变化、归一化规模变化
- 证据等级：A

### E4
- 证据类型：实验
- 定位信息：Section5.1（Descriptions of Datasets）
- 原文关键句：“Our datasets consist of MBO data for five highly liquid stocks... for the entire year of2018 from the London Stock Exchange. We take first6 months as training data, next3 as validation, last3 as testing.”
- 我的转述：数据集包含伦敦证交所2018年5只高流动性股票的MBO数据，按时间分为6个月训练、3个月验证、3个月测试
- 证据等级：A

### E5
- 证据类型：结果
- 定位信息：Section5.3（Experimental Results）
- 原文关键句：“We observe that the models trained with LOB data are comparable, but slightly outperform the ones using MBO data... MBO-LSTM achieves61.94% accuracy at k=20, while LOB-LSTM achieves66.09%.”
- 我的转述：LOB模型性能略优于MBO模型，但MBO-LSTM在k=20时达到61.94%准确率，表现具有竞争力
- 证据等级：A

### E6
- 证据类型：结果
- 定位信息：Section5.3（Experimental Results）
- 原文关键句：“Ensemble-MBO-LOB delivers the best performance... at k=20, it achieves68.95% accuracy, which is higher than Ensemble-MBO (62.35%) and Ensemble-LOB (67.97%).”
- 我的转述：MBO与LOB组合模型性能最优，k=20时准确率达68.95%，高于单独MBO或LOB组合模型
- 证据等级：A

### E7
- 证据类型：局限
- 定位信息：Section5.3（Experimental Results）
- 原文关键句：“While a priori, MBO data contains more information (contents of level and trades), it is harder to model the raw messages rather than LOB snapshots which can be seen as derived or handcrafted features from the MBO data.”
- 我的转述：MBO数据信息更丰富，但原始消息比LOB快照更难建模（LOB是MBO的衍生手工特征）
- 证据等级：A

### E8
- 证据类型：方法
- 定位信息：Section3.3（Data Labelling）
- 原文关键句：“We define l_t=(m_+(t)-m_-(t))/m_-(t), where m_-(t) is average mid-price over past k steps, m_+(t) over future k steps; labels are up if l_t>α, down if < -α, stationary otherwise.”
- 我的转述：分类标签基于未来k步与过去k步中间价变化率，分为上涨（l_t>α）、下跌（l_t<-α）、平稳三类
- 证据等级：A

### E9
- 证据类型：实验
- 定位信息：Section5.1（Descriptions of Datasets）
- 原文关键句：“We test our models at three prediction horizons (k=20,50,100) and choose α for each instrument to have a balanced training set.”
- 我的转述：预测步长为k=20、50、100 Tick，α值根据工具调整以平衡训练集标签
- 证据等级：A

### E10
- 证据类型：结果
- 定位信息：Section5.3（Experimental Results）
- 原文关键句：“MBO-LSTM and MBO-Attention all have a recurrent structure... deliver better results than MLPs when modelling financial time-series.”
- 我的转述：循环模型（MBO-LSTM、MBO-Attention）在MBO数据建模中性能优于MLP模型
- 证据等级：A

## 3）主题笔记（Topic Notes）
### MBO数据的定义与粒度
MBO数据（Level3）是非聚合的订单级数据，展示个体交易者的买卖订单，粒度高于LOB（Level2）数据（依据E1）。它捕获订单添加、更新、取消等个体行为，包含更丰富的市场微观结构信息（依据E1）。

### MBO数据预处理与特征构建
论文通过归一化将MBO数据转化为6个特征：Side、Action、归一化价格、归一化规模、归一化价格变化、归一化规模变化（依据E2、E3）。归一化价格将订单价格转换为相对于中间价的Tick变化，支持多工具模型训练（依据E2）。

### 标签定义与预测步长
标签基于未来k步与过去k步中间价变化率，分为上涨、下跌、平稳三类（依据E8）。预测步长为k=20、50、100 Tick，α值调整以平衡训练集标签（依据E9）。

### MBO与LOB模型性能对比
LOB模型性能略优于MBO模型，但MBO-LSTM等循环模型表现具有竞争力（依据E5）。循环模型（LSTM、Attention）在MBO数据上的性能显著优于线性模型和MLP（依据E10）。

### MBO与LOB组合模型的优势
MBO与LOB信号相关性低，组合模型（Ensemble-MBO-LOB）性能最优，k=20时准确率达68.95%（依据E6）。这表明MBO数据提供了LOB未捕获的互补信息（依据E6）。

### MBO建模的局限
MBO原始消息比LOB快照更难建模，因为LOB是MBO的衍生手工特征（依据E7）。论文未考虑交易成本与实际执行约束，未来可扩展到市场-making或交易执行场景（隐含于结论）。

## 4）可直接写进论文的句子草稿（可选）
1. MBO数据（Level3）是非聚合的订单级数据，展示个体交易者的买卖订单，粒度高于LOB（Level2）数据（依据E1）。
2. 论文通过归一化将MBO数据转化为6个特征，支持多工具短期价格方向预测（依据E2、E3）。
3. 预测标签基于未来k步与过去k步中间价变化率，分为上涨、下跌、平稳三类（依据E8）。
4. MBO-LSTM等循环模型在k=20时达到61.94%准确率，与LOB模型性能相当（依据E5）。
5. MBO与LOB组合模型性能最优，k=20时准确率达68.95%，高于单独组合模型（依据E6）。
6. MBO数据信息更丰富但原始消息比LOB快照更难建模，LOB是MBO的衍生手工特征（依据E7）。
