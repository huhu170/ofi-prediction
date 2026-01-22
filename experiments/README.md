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
| 标签阈值 α | 0.002 | 三分类：涨/跌/平 |
| 预测步长 k | 5, 15, 30 分钟 | |
| Gap | 30分钟 | 避免标签泄漏 |
| 划分比例 | 70/15/15 | 训练/验证/测试 |

---

## 目录结构

```
experiments/
├── exp_config.py              # 实验配置文件
├── run_all_experiments.py     # 主运行脚本
├── README.md                  # 本文档
│
├── # 4.1节 数据与特征的统计分析
├── exp_4_1_1_sample_stats.py          # 表4.1-1 样本描述性统计
├── exp_4_1_2_feature_distribution.py  # 图4.1-1 特征分布 + 表4.1-2 统计量
├── exp_4_1_3_label_balance.py         # 表4.1-3 标签分布
├── exp_4_1_4_correlation.py           # 表4.1-4 相关性检验
├── exp_4_1_5_ols_regression.py        # 表4.1-5 OLS回归
├── exp_4_1_6_scale_comparison.py      # 表4.1-6 多尺度解释力对比
│
├── # 4.2节 模型性能评估
├── exp_4_2_1_baseline_models.py       # 表4.2-1 基准模型
├── exp_4_2_2_deep_models.py           # 表4.2-2 深度学习模型
├── exp_4_2_3a_pv_crossattn_ablation.py # 表4.2-3a PV-CrossAttention消融
├── exp_4_2_3b_lsf_ablation.py         # 表4.2-3b LSF消融
├── exp_4_2_4_feature_ablation.py      # 表4.2-4 特征消融
├── exp_4_2_5_threshold_sensitivity.py # 表4.2-5 阈值敏感性
│
├── # 4.3节 策略回测
├── exp_4_3_1_backtest_config.py       # 表4.3-1 回测参数
├── exp_4_3_2_backtest.py              # 表4.3-2 经济价值 + 图4.3-3 净值曲线
├── exp_4_3_3_scale_comparison.py      # 表4.3-4 多尺度回测对比
├── exp_4_3_4_cost_sensitivity.py      # 表4.3-5 交易成本敏感性
│
├── # 4.4节 可解释性与稳健性
├── exp_4_4_1_shap_analysis.py         # 图4.4-1/2 SHAP归因
├── exp_4_4_2_regime_split.py          # 表4.4-3 市场状态分组
├── exp_4_4_2a_event_study.py          # 表4.4-3a/b 金融事件案例
├── exp_4_4_3_asset_split.py           # 表4.4-4 资产类型分组
├── exp_4_4_5_granger_causality.py     # 表4.4-5 Granger因果
├── exp_4_4_6_causal_feature_comparison.py # 表4.4-6 因果特征对比
├── exp_4_4_7_counterfactual.py        # 图4.4-7 反事实分析
├── exp_4_4_8_decay_analysis.py        # 图4.4-8 预测衰减
├── exp_4_4_9_market_state.py          # 表4.4-10 市场状态对比
├── exp_4_4_10_rolling_training.py     # 图4.4-11 滚动训练
└── exp_4_4_11_shap_vs_causal.py       # 表4.4-13 SHAP vs Granger
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
python run_all_experiments.py --exp 4.4.1      # SHAP分析
```

### 列出所有实验
```bash
python run_all_experiments.py --list
```

## 输出说明

实验结果保存在 `experiment_results/` 目录：

```
experiment_results/
├── figures/               # 图表
│   ├── fig_4_1_*.png
│   ├── fig_4_2_*.png
│   ├── fig_4_3_*.png
│   └── fig_4_4_*.png
└── tables/                # 表格
    ├── table_4_1_*.csv
    ├── table_4_2_*.csv
    ├── table_4_3_*.csv
    └── table_4_4_*.csv
```

## 实验与论文的对应关系

### 4.1节 数据与特征的统计分析

| 实验ID | 论文表/图 | 说明 |
|--------|-----------|------|
| 4.1.1 | 表4.1-1 | 样本描述性统计 |
| 4.1.2 | 图4.1-1 + 表4.1-2 | 特征分布 |
| 4.1.3 | 表4.1-3 | 标签分布检验 |
| 4.1.4 | 表4.1-4 | 相关性检验 |
| 4.1.5 | 表4.1-5 | OLS回归 |
| 4.1.6 | 表4.1-6 + 图4.1-2 | 多尺度解释力 |

### 4.2节 模型性能评估

| 实验ID | 论文表/图 | 说明 |
|--------|-----------|------|
| 4.2.1 | 表4.2-1 | 基准模型（ARIMA/LR/RF/XGB） |
| 4.2.2 | 表4.2-2 | 深度模型（LSTM/GRU/Transformer） |
| 4.2.3a | 表4.2-3a + 图4.2-3a | PV-CrossAttention消融 |
| 4.2.3b | 表4.2-3b/c + 图4.2-3b | LSF消融 |
| 4.2.4 | 表4.2-4 | 特征消融 |
| 4.2.5 | 表4.2-5 | 阈值敏感性 |

### 4.3节 策略回测

| 实验ID | 论文表/图 | 说明 |
|--------|-----------|------|
| 4.3.1 | 表4.3-1 | 回测参数配置 |
| 4.3.2 | 表4.3-2 + 图4.3-3 | 经济价值指标 + 净值曲线 |
| 4.3.3 | 表4.3-4 | 多尺度回测对比 |
| 4.3.4 | 表4.3-5/6/7 + 图4.3-6 | 交易成本敏感性 |

### 4.4节 可解释性与稳健性

| 实验ID | 论文表/图 | 说明 | 研究线路 |
|--------|-----------|------|----------|
| 4.4.1 | 图4.4-1/2 | SHAP归因 | 线路一 |
| 4.4.5 | 表4.4-5 | Granger因果 | 线路二 |
| 4.4.6 | 表4.4-6 | 因果特征验证 | 线路二 |
| 4.4.7 | 图4.4-7 | 反事实分析 | 线路二 |
| 4.4.2 | 表4.4-3 | 市场状态分组 | 线路三 |
| 4.4.2a | 表4.4-3a/b + 图4.4-3c | 金融事件案例 | 线路三 |
| 4.4.3 | 表4.4-4 | 资产类型分组 | 线路三 |
| 4.4.8 | 图4.4-8 + 表4.4-9 | 预测衰减 | 线路三 |
| 4.4.9 | 表4.4-10 | 市场状态预测 | 线路三 |
| 4.4.10 | 图4.4-11 + 表4.4-12 | 滚动训练 | 线路三 |
| 4.4.11 | 表4.4-13 | SHAP vs Granger | 综合 |

## 依赖

见项目根目录 `requirements.txt`

## 可复现性

- 固定随机种子: `seed=42`
- 所有实验使用 `exp_config.set_seed()` 确保可复现
