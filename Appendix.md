# 附录A 实验代码说明

本研究的全部实验代码已开源，托管于GitHub仓库：

> **仓库地址：** https://github.com/huhu170/ofi-prediction

## 一、代码仓库结构

```
项目根目录/
├── scripts/                    # 数据处理与模型训练流水线
│   ├── 04_fetch_kline.py       # 从富途API拉取多周期K线数据
│   ├── 05_data_cleaner.py      # K线数据清洗（异常、时段、缺失处理）
│   ├── 06_feature_calculator.py # 22维特征计算（TI、动量、波动率等）
│   ├── 07_dataset_builder.py   # 滑动窗口数据集构建与时序划分
│   ├── 08_model_trainer.py     # 模型训练（含PV-CrossAttention与LSF）
│   ├── 09_backtest.py          # 策略回测引擎
│   └── 10_shap_analysis.py     # SHAP可解释性分析
│
├── experiments/                # 论文第四章实验脚本
│   ├── exp_config.py           # 实验全局配置
│   ├── run_all_experiments.py  # 实验批量调度器
│   ├── exp_4_1_*.py            # 4.1节：数据与特征统计分析
│   ├── exp_4_2_*.py            # 4.2节：模型性能评估
│   ├── exp_4_3_*.py            # 4.3节：策略回测与经济价值
│   ├── exp_4_4_*.py            # 4.4节：可解释性与稳健性检验
│   ├── exp_4_4_1b_shap_multi_stock.py  # 多股票SHAP跨标的鲁棒性检验
│   ├── exp_5_statistical_tests.py      # Wilcoxon符号秩检验
│   ├── exp_6_binary_f1.py              # 置信度分层F1分析
│   ├── backtest_single.py      # 单模型回测引擎（含标准夏普比率计算）
│   ├── backtest_batch.py       # 批量回测调度器
│   └── cost_sensitivity_real.py # 交易成本敏感性分析
│
├── models/                     # 已训练模型权重（.pt / .pkl）
├── outputs/ch4/                # 实验输出（表格CSV + 图表PNG）
│   ├── tables/                 # 各实验输出的CSV表格
│   └── figures/                # 各实验输出的可视化图表
│
└── data/                       # 原始与处理后数据
```

## 二、实验脚本命名规则

实验脚本采用统一命名格式：

```
exp_{章}_{节}_{序号}_{描述}.py
```

其中 `{章}` 对应论文章号（第四章为`4`），`{节}` 对应节号，`{序号}` 为节内实验编号，`{描述}` 为实验内容的英文简述。例如：`exp_4_1_1_sample_stats.py` 对应"第四章第一节第1个实验——样本描述性统计"。

辅助脚本（如回测引擎、成本分析等）不遵循此编号规则，以功能命名。

## 三、实验脚本与论文数据对应关系

下表列出每个实验脚本所生成的数据与论文中表格/图表的对应关系。所有实验均可通过 `python run_all_experiments.py` 一键复现。

### 3.1 第四章第一节——数据与特征的统计分析

| 脚本文件名 | 论文表格/图表 | 功能说明 | 输出文件 |
|-----------|-------------|---------|---------|
| `exp_4_1_1_sample_stats.py` | 表4.1-1a | 11只标的的样本量、数据完整性统计 | `table_4_1_sample_stats.csv` |
| `exp_4_1_2_feature_distribution.py` | 表4.1-1b、图4.1-1 | 22维特征的描述性统计与分布直方图 | `table_4_1_2_feature_stats.csv`、`fig_4_1_feature_distribution.png` |
| `exp_4_1_3_label_balance.py` | 表4.1-1c、表4.1-2 | 自适应阈值标签分布检验；固定阈值与自适应阈值的对比 | `table_4_1_3_label_distribution.csv` |
| `exp_4_1_4_correlation.py` | 表4.1-3a | 特征与同期收益率的Pearson/Spearman相关系数 | `table_4_1_4_correlation.csv` |
| `exp_4_1_5_ols_regression.py` | 表4.1-3b、表4.1-3c | 单变量OLS回归；多尺度解释力对比 | `table_4_1_5_ols_regression.csv` |

### 3.2 第四章第二节——模型性能评估与实证对比

| 脚本文件名 | 论文表格/图表 | 功能说明 | 输出文件 |
|-----------|-------------|---------|---------|
| `exp_4_2_2_deep_models.py` | 表4.2-1、表4.2-2a、表4.2-2b | 6种深度学习模型的测试集评估（Accuracy/F1/AUC）；PV-CrossAttention与LSF消融实验 | `table_4_2_2_model_comparison.csv` |
| `exp_4_2_eval_sklearn.py` | 表4.2-1（传统ML部分） | 3种传统机器学习模型（LogReg/RF/XGBoost）的测试集评估 | `table_4_2_2_model_summary.csv` |
| `exp_4_2_extract_metrics.py` | 图4.2-1 | 汇总全部9种模型的指标并生成对比可视化 | `fig_4_2_1_model_comparison.png` |
| `exp_5_statistical_tests.py` | 4.2节正文（Wilcoxon检验段落） | 逐股Wilcoxon符号秩检验（DL vs ML），逐股胜负统计 | `table_5_wilcoxon_tests.csv` |
| `exp_6_binary_f1.py` | 4.2节正文（置信度分层段落） | 10只股票的置信度分层F1分析（Top-10%/20%/30%），随机基线模拟 | `table_6_confidence_all_stocks.csv`、`table_6_random_baselines.csv`、`table_6_confidence_summary.csv` |

### 3.3 第四章第三节——策略回测与经济价值分析

| 脚本文件名 | 论文表格/图表 | 功能说明 | 输出文件 |
|-----------|-------------|---------|---------|
| `backtest_single.py` | 表4.3-1、图4.3-1 | 单模型回测引擎：从本地parquet文件加载K线数据，计算特征、加载模型、执行回测，输出收益率、标准夏普比率、最大回撤、胜率等指标。夏普比率通过`compute_sharpe()`函数从逐笔资金曲线聚合为日收益率后按标准公式计算 | `backtest_{model}_detail.csv`、`backtest_{model}.csv` |
| `exp_4_3_1_backtest_config.py` | 表4.3-1表注 | 回测参数配置输出 | `table_4_3_1_backtest_config.csv` |
| `cost_sensitivity_real.py` | 表4.3-2 | 4个代表性模型在5档交易成本下的收益变化 | `cost_sensitivity_*.csv` |
| `exp_4_3_real_backtest.py` | 表4.3-3 | PV-Transformer在不同概率阈值下的回测表现 | `table_4_3_3_threshold.csv` |

### 3.4 第四章第四节——可解释性与稳健性检验

| 脚本文件名 | 论文表格/图表 | 功能说明 | 输出文件 |
|-----------|-------------|---------|---------|
| `exp_4_4_1_shap_analysis.py` | 表4.4-1、图4.4-1 | XGBoost（TreeSHAP）与CNN-LSTM（GradientSHAP）的双模型特征归因对比（腾讯控股） | `table_4_4_1_feature_importance.csv`、`fig_4_4_1_feature_importance.png` |
| `exp_4_4_1b_shap_multi_stock.py` | 4.4节正文（跨标的鲁棒性段落） | 全部10只股票的双模型SHAP归因，跨标的Spearman秩相关与类别一致性分析 | `table_4_4_1b_shap_multi_stock.csv`、`table_4_4_1b_shap_category_summary.csv`、`table_4_4_1b_shap_cross_stock_consistency.csv` |
| `exp_4_4_5_granger_causality.py` | 表4.4-2 | 22维特征对未来收益率的Granger因果检验（滞后阶数=5） | `table_4_4_2_granger_causality.csv` |
| `exp_4_4_2_regime_split.py` | 表4.4-3 | 按波动率分位数划分市场状态，检验不同状态下各模型的F1-macro | `table_4_4_3_regime_heterogeneity.csv` |

## 四、数据处理流水线与论文章节的对应关系

数据处理脚本位于 `scripts/` 目录，按编号顺序执行：

| 脚本文件名 | 论文章节 | 功能说明 |
|-----------|---------|---------|
| `04_fetch_kline.py` | 第三章第一节 | 通过富途OpenD API拉取港股多周期K线数据并保存为本地parquet文件 |
| `05_data_cleaner.py` | 第三章第一节 | K线数据清洗：盘前/盘后过滤、异常值检测、缺失值处理 |
| `06_feature_calculator.py` | 第三章第二节 | 实现论文公式3.2-1至3.2-6定义的22维特征计算 |
| `07_dataset_builder.py` | 第三章第三节 | 滑动窗口采样、自适应标签生成、时序划分（70%/15%/15%）、防泄漏Gap |
| `08_model_trainer.py` | 第三章第四节 | 9种模型的统一训练框架（含PV-CrossAttention与LSF模块） |
| `09_backtest.py` | 第四章第三节 | 基于模型预测信号的策略回测引擎 |
| `10_shap_analysis.py` | 第四章第四节 | SHAP特征归因分析 |

## 五、环境与可复现性

**运行环境**：Python 3.10+，关键依赖见项目根目录 `requirements.txt`，主要包括：`torch`、`scikit-learn`、`pandas`、`numpy`、`pyarrow`、`statsmodels`、`matplotlib`、`shap`。

**可复现性保障**：

- 固定随机种子（seed=42），通过 `exp_config.set_seed()` 在每个实验脚本起始处设定
- 数据按时间顺序划分为训练集（70%）、验证集（15%）、测试集（15%），各集之间设30分钟Gap防止标签泄漏
- 标准化参数仅从训练集计算，验证集与测试集使用训练集参数进行变换
- 所有已训练模型权重保存在 `models/` 目录（`.pt` 格式为PyTorch深度学习模型，`.pkl` 格式为scikit-learn传统机器学习模型）

## 六、快速复现指南

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 数据预处理（需配合富途OpenD API拉取原始数据）
python scripts/05_data_cleaner.py --all --multi-scale
python scripts/06_feature_calculator.py --all --multi-scale
python scripts/07_dataset_builder.py --all --multi-scale

# 3. 一键运行全部第四章实验
python experiments/run_all_experiments.py

# 4. 或按章节选择性运行
python experiments/run_all_experiments.py --section 4.1    # 数据统计
python experiments/run_all_experiments.py --section 4.2    # 模型评估
python experiments/run_all_experiments.py --section 4.3    # 策略回测
python experiments/run_all_experiments.py --section 4.4    # 可解释性

# 5. 实验输出位于 outputs/ch4/tables/ 和 outputs/ch4/figures/
```
