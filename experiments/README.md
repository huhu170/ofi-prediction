# 论文第四章实验脚本

> 版本: v1.0  
> 日期: 2026-01-14

---

## 概述

本目录包含论文第四章《订单流不平衡股价预测实证分析》的所有实验脚本，用于生成论文中的表格和图表。

## 目录结构

```
experiments/
├── exp_config.py              # 共享配置文件
├── run_all_experiments.py     # 一键运行脚本
├── README.md                  # 本文档
│
├── exp_4_1_*.py               # 4.1节: OFI与价格变动的统计分析
├── exp_4_2_*.py               # 4.2节: 模型性能评估与实证对比
├── exp_4_3_*.py               # 4.3节: 策略回测与经济价值评估
└── exp_4_4_*.py               # 4.4节: 模型可解释性与稳健性检验
```

## 快速开始

### 运行所有实验

```bash
cd experiments
python run_all_experiments.py
```

### 运行特定章节

```bash
# 只运行4.1节实验
python run_all_experiments.py --section 4.1

# 预览将要运行的脚本
python run_all_experiments.py --dry-run
```

### 运行单个实验

```bash
python exp_4_1_1_sample_stats.py
```

---

## 实验列表

### 4.1节: OFI与价格变动的统计分析

| 脚本 | 论文对应 | 输出 |
|------|----------|------|
| `exp_4_1_1_sample_stats.py` | 表4-1 | `table_4_1_sample_stats.csv` |
| `exp_4_1_2_ofi_distribution.py` | 图4-1, 表4-2 | `fig_4_1_*.png`, `table_4_2_*.csv` |
| `exp_4_1_3_label_balance.py` | 表4-3 | `table_4_3_label_distribution.csv` |
| `exp_4_1_4_correlation.py` | 表4-4 | `table_4_4_correlation.csv` |
| `exp_4_1_5_ols_regression.py` | 表4-5 | `table_4_5_ols_regression.csv` |
| `exp_4_1_6_intraday_impact.py` | 图4-2 | `fig_4_2_intraday_impact.png` |
| `exp_4_1_7_depth_comparison.py` | 表4-6, 图4-3 | `table_4_6_*.csv`, `fig_4_3_*.png` |

### 4.2节: 模型性能评估

| 脚本 | 论文对应 | 输出 |
|------|----------|------|
| `exp_4_2_1_baseline_models.py` | 表4-7 | `table_4_7_baseline_models.csv` |
| `exp_4_2_2_deep_models.py` | 表4-8 | `table_4_8_deep_models.csv` |
| `exp_4_2_3_model_comparison_plot.py` | 图4-4 | `fig_4_4_model_comparison.png` |
| `exp_4_2_4_ablation.py` | 表4-9 | `table_4_9_ablation.csv` |
| `exp_4_2_5_threshold_sensitivity.py` | 表4-10 | `table_4_10_threshold_sensitivity.csv` |

### 4.3节: 策略回测

| 脚本 | 论文对应 | 输出 |
|------|----------|------|
| `exp_4_3_1_backtest_config.py` | 表4-11 | `table_4_11_backtest_config.csv` |
| `exp_4_3_2_backtest.py` | 表4-12, 图4-5 | `table_4_12_*.csv`, `fig_4_5_*.png` |
| `exp_4_3_3_ofi_comparison.py` | 表4-13 | `table_4_13_ofi_economic_comparison.csv` |

### 4.4节: 可解释性与稳健性

| 脚本 | 论文对应 | 输出 |
|------|----------|------|
| `exp_4_4_1_shap_analysis.py` | 图4-6, 图4-7 | `fig_4_6_*.png`, `fig_4_7_*.png` |
| `exp_4_4_2_regime_split.py` | 表4-14 | `table_4_14_regime_comparison.csv` |
| `exp_4_4_3_asset_split.py` | 表4-15 | `table_4_15_asset_comparison.csv` |
| `exp_4_4_5_control_group.py` | 表4-16 | `table_4_16_control_group.csv` |

---

## 输出目录

运行实验后，结果保存在：

```
experiment_results/
├── tables/          # CSV格式表格
└── figures/         # PNG格式图表
```

---

## 依赖关系

实验脚本依赖以下数据流程脚本（位于 `scripts/` 目录）：

```
10_data_cleaner.py    → 数据清洗
11_feature_calculator.py → 特征计算
12_dataset_builder.py → 数据集构建
13_model_trainer.py   → 模型训练
14_backtest.py        → 策略回测
16_shap_analysis.py   → SHAP分析
```

确保先运行数据流程脚本生成必要的数据和模型。

---

## 注意事项

1. **数据依赖**: 部分脚本需要真实数据，若数据不存在会使用模拟数据演示
2. **计算资源**: 深度模型评估需要GPU支持（可选）
3. **运行时间**: 完整运行所有实验约需30-60分钟
4. **结果复现**: 已设置随机种子确保结果可复现
