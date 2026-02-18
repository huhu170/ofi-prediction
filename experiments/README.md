# 论文实验脚本

> 与论文第四章"分钟级价格预测实证分析"对齐

## 数据准备（实验前置步骤）

运行实验脚本前，需要先完成数据准备流程：

```bash
# Step 1: 数据清洗（从数据库读取原始K线，清洗异常）
python scripts/10b_kline_data_cleaner.py --all --multi-scale

# Step 2: 特征计算（计算22维特征，生成标签）
python scripts/11b_kline_feature_calculator.py --all --multi-scale

# Step 3: 数据集构建（滑动窗口、标准化、划分）
python scripts/12b_kline_dataset_builder.py --all --multi-scale
```

**数据配置说明（与论文第三章对齐）**：

| 配置项 | 取值 | 说明 |
|--------|------|------|
| 输入特征 | 22维 | 见论文表3.3-1 |
| 输入窗口 | 1M:60, 5M:24, 60M:12, 日K:20 | 见论文表3.3-2 |
| 标签 | 自适应α三分类 | 涨/跌/平 |
| 预测步长 k | 5分钟 | |
| Gap | 30分钟 | 避免标签泄漏 |
| 划分比例 | 70/15/15 | 训练/验证/测试 |

---

## 目录结构

```
experiments/
├── exp_config.py                      # 实验配置（路径、股票列表、特征定义、超参数）
├── run_all_experiments.py             # 主运行脚本（按节/实验ID调度）
├── README.md                          # 本文档
│
├── # ---- 4.1节 数据与特征的统计分析 ----
├── exp_4_1_1_sample_stats.py          # 样本描述性统计（真实parquet数据）
├── exp_4_1_2_feature_distribution.py  # 特征分布分析
├── exp_4_1_3_label_balance.py         # 标签分布检验
├── exp_4_1_4_correlation.py           # 特征-收益率相关性
├── exp_4_1_5_ols_regression.py        # OLS回归分析
│
├── # ---- 4.2节 模型性能评估 ----
├── exp_4_2_2_deep_models.py           # 深度学习模型评估（加载真实checkpoint）
├── exp_4_2_eval_sklearn.py            # sklearn模型评估（LogReg/RF/XGBoost）
├── exp_4_2_extract_metrics.py         # 模型指标汇总与对比图
│
├── # ---- 4.3节 策略回测 ----
├── exp_4_3_1_backtest_config.py       # 回测参数配置表
├── exp_4_3_real_backtest.py           # 真实数据策略回测（DB + 训练模型）
├── backtest_single.py                 # 单模型回测引擎（核心）
├── backtest_batch.py                  # 批量回测调度器
├── cost_sensitivity_real.py           # 交易成本敏感性分析（真实回测）
│
├── # ---- 4.4节 可解释性与稳健性 ----
├── exp_4_4_1_shap_analysis.py         # XGBoost特征重要性分析
├── exp_4_4_5_granger_causality.py     # Granger因果检验（真实DB数据）
├── exp_4_4_2_regime_split.py          # 市场状态异质性检验（真实模型预测）
│
└── # ---- 其他工具 ----
    └── retrain_multi_scale.py         # 多尺度模型批量重训练
```

## 使用方法

### 运行所有实验
```bash
python run_all_experiments.py
```

### 运行特定章节
```bash
python run_all_experiments.py --section 4.1    # 4.1节数据统计
python run_all_experiments.py --section 4.2    # 4.2节模型评估
python run_all_experiments.py --section 4.3    # 4.3节策略回测
python run_all_experiments.py --section 4.4    # 4.4节可解释性
```

### 运行单个实验
```bash
python run_all_experiments.py --exp 4.1.1      # 样本统计
python run_all_experiments.py --exp 4.2.2      # 深度模型评估
python run_all_experiments.py --exp 4.4.1      # 特征重要性分析
python run_all_experiments.py --exp 4.4.5      # Granger因果检验
python run_all_experiments.py --exp 4.4.2      # 市场状态异质性检验
```

### 独立回测脚本
```bash
python experiments/backtest_single.py lstm              # 单模型回测
python experiments/backtest_single.py pv_transformer    # PV-Transformer回测
python experiments/backtest_batch.py                    # 全模型批量回测
python experiments/cost_sensitivity_real.py             # 交易成本敏感性
```

### 列出所有实验
```bash
python run_all_experiments.py --list
```

## 输出说明

实验结果保存在 `outputs/ch4/` 目录：

```
outputs/ch4/
├── figures/               # 图表
│   ├── fig_4_1_*.png
│   ├── fig_4_2_*.png
│   └── fig_4_4_*.png
└── tables/                # 表格
    ├── table_4_1_*.csv
    ├── table_4_2_*.csv
    ├── table_4_3_*.csv    # 回测结果
    ├── table_4_4_*.csv    # 可解释性结果
    ├── backtest_*.csv     # 详细回测数据
    └── cost_sensitivity_*.csv  # 成本敏感性数据
```

## 实验与论文的对应关系

### 4.1节 数据与特征的统计分析

| 实验ID | 论文表/图 | 数据来源 |
|--------|-----------|----------|
| 4.1.1 | 表4.1-1 样本描述性统计 | 真实parquet数据 |
| 4.1.2 | 图4.1-1 特征分布 | 真实parquet数据 |
| 4.1.3 | 表4.1-3 标签分布 | 真实parquet数据 |
| 4.1.4 | 表4.1-4 相关性检验 | 真实parquet数据 |
| 4.1.5 | 表4.1-5 OLS回归 | 真实parquet数据 |

### 4.2节 模型性能评估

| 实验ID | 论文表/图 | 数据来源 |
|--------|-----------|----------|
| 4.2.2 | 表4.2-1 模型性能汇总 | 真实训练checkpoint |
| 4.2.2s | 表4.2-1 sklearn模型 | 真实DB数据 + pkl模型 |
| 4.2.2m | 图4.2-1 模型对比图 | 汇总checkpoint指标 |

### 4.3节 策略回测

| 实验ID | 论文表/图 | 数据来源 |
|--------|-----------|----------|
| 4.3.1 | 表4.3-1 回测参数 | 配置输出 |
| 4.3.2 | 表4.3-1 经济价值 | 真实DB数据 + 训练模型 |
| 4.3.4 | 表4.3-2 成本敏感性 | 真实回测 × 5档成本 |

### 4.4节 可解释性与稳健性

| 实验ID | 论文表/图 | 数据来源 | 研究线路 |
|--------|-----------|----------|----------|
| 4.4.1 | 表4.4-1 + 图4.4-1 特征重要性 | 真实XGBoost模型 | 线路一 |
| 4.4.5 | 表4.4-2 Granger因果检验 | 真实DB数据 + statsmodels | 线路二 |
| 4.4.2 | 表4.4-3 市场状态异质性 | 真实DB数据 + 训练模型 | 线路三 |

## 依赖

见项目根目录 `requirements.txt`

关键依赖：`torch`, `scikit-learn`, `pandas`, `psycopg2`, `statsmodels`, `matplotlib`

## 可复现性

- 固定随机种子: `seed=42`
- 所有实验使用 `exp_config.set_seed()` 确保可复现
- 数据库: PostgreSQL (127.0.0.1:5433, futu_ofi)