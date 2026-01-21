# 第四章 实验任务清单

> 📅 更新时间：2026-01-21
> 📊 统计：共 **27** 个实验（表格 18 | 图片 9）

---

## 一、任务总览

| 节 | 任务数 | 脚本数 | 状态 |
|----|--------|--------|------|
| 4.1 数据与特征的统计分析 | 6 | 6 | ⬜ 0% |
| 4.2 模型性能评估 | 6 | 6 | ⬜ 0% |
| 4.3 策略回测与经济价值 | 4 | 4 | ⬜ 0% |
| 4.4 可解释性与稳健性 | 11 | 11 | ⬜ 0% |
| **合计** | **27** | **27** | **⬜ 0%** |

**状态图例**：⬜未开始 | 🔄进行中 | ✅已完成 | ❌阻塞

---

## 二、详细任务清单

### 第一节：数据与特征的统计分析 (4.1.x)

| 编号 | 任务名称 | 脚本 | 产出 | 状态 |
|------|----------|------|------|------|
| 4.1.1 | 样本描述性统计 | `exp_4_1_1_sample_stats.py` | 表4.1-1 | ⬜ |
| 4.1.2 | 特征分布分析 | `exp_4_1_2_feature_distribution.py` | 图4.1-1 + 表4.1-2 | ⬜ |
| 4.1.3 | 标签分布检验 | `exp_4_1_3_label_balance.py` | 表4.1-3 | ⬜ |
| 4.1.4 | 相关性检验 | `exp_4_1_4_correlation.py` | 表4.1-4 | ⬜ |
| 4.1.5 | OLS回归分析 | `exp_4_1_5_ols_regression.py` | 表4.1-5 | ⬜ |
| 4.1.6 | 多尺度解释力对比 | `exp_4_1_6_scale_comparison.py` | 表4.1-6 + 图4.1-2 | ⬜ |

### 第二节：模型性能评估 (4.2.x)

| 编号 | 任务名称 | 脚本 | 产出 | 状态 |
|------|----------|------|------|------|
| 4.2.1 | 基准模型评估 | `exp_4_2_1_baseline_models.py` | 表4.2-1 | ⬜ |
| 4.2.2 | 深度学习模型评估 | `exp_4_2_2_deep_models.py` | 表4.2-2 | ⬜ |
| 4.2.3a | PV-CrossAttention消融 | `exp_4_2_3a_pv_crossattn_ablation.py` | 表4.2-3a + 图4.2-3a | ⬜ |
| 4.2.3b | LSF消融实验 | `exp_4_2_3b_lsf_ablation.py` | 表4.2-3b/c + 图4.2-3b | ⬜ |
| 4.2.4 | 特征消融实验 | `exp_4_2_4_feature_ablation.py` | 表4.2-4 | ⬜ |
| 4.2.5 | 标签阈值敏感性 | `exp_4_2_5_threshold_sensitivity.py` | 表4.2-5 | ⬜ |

### 第三节：策略回测与经济价值 (4.3.x)

| 编号 | 任务名称 | 脚本 | 产出 | 状态 |
|------|----------|------|------|------|
| 4.3.1 | 回测参数配置 | `exp_4_3_1_backtest_config.py` | 表4.3-1 | ⬜ |
| 4.3.2 | 模型回测 | `exp_4_3_2_backtest.py` | 表4.3-2 + 图4.3-3 | ⬜ |
| 4.3.3 | 多尺度回测对比 | `exp_4_3_3_scale_comparison.py` | 表4.3-4 | ⬜ |
| 4.3.4 | 交易成本敏感性 | `exp_4_3_4_cost_sensitivity.py` | 表4.3-5/7 + 图4.3-6 | ⬜ |

### 第四节：可解释性与稳健性 (4.4.x)

| 编号 | 任务名称 | 脚本 | 产出 | 研究线路 | 状态 |
|------|----------|------|------|----------|------|
| 4.4.1 | SHAP归因分析 | `exp_4_4_1_shap_analysis.py` | 图4.4-1/2 | 线路一 | ⬜ |
| 4.4.2 | 市场状态分组 | `exp_4_4_2_regime_split.py` | 表4.4-3 | 线路三 | ⬜ |
| 4.4.2a | 金融事件案例 | `exp_4_4_2a_event_study.py` | 表4.4-3a/b + 图4.4-3c | 线路三 | ⬜ |
| 4.4.3 | 资产类型分组 | `exp_4_4_3_asset_split.py` | 表4.4-4 | 线路三 | ⬜ |
| 4.4.5 | Granger因果检验 | `exp_4_4_5_granger_causality.py` | 表4.4-5 | 线路二 | ⬜ |
| 4.4.6 | 因果特征子集验证 | `exp_4_4_6_causal_feature_comparison.py` | 表4.4-6 | 线路二 | ⬜ |
| 4.4.7 | 反事实分析 | `exp_4_4_7_counterfactual.py` | 图4.4-7 | 线路二 | ⬜ |
| 4.4.8 | 预测衰减分析 | `exp_4_4_8_decay_analysis.py` | 图4.4-8 + 表4.4-9 | 线路三 | ⬜ |
| 4.4.9 | 市场状态预测对比 | `exp_4_4_9_market_state.py` | 表4.4-10 | 线路三 | ⬜ |
| 4.4.10 | 滚动训练有效性 | `exp_4_4_10_rolling_training.py` | 图4.4-11 + 表4.4-12 | 线路三 | ⬜ |
| 4.4.11 | SHAP与Granger对比 | `exp_4_4_11_shap_vs_causal.py` | 表4.4-13 | 综合 | ⬜ |

---

## 三、研究线路说明

| 研究线路 | 核心问题 | 验证假设 | 关键实验 |
|----------|----------|----------|----------|
| **线路一** | 模型如何做出预测？ | H4a：量价特征是主要驱动因素 | 4.4.1 SHAP归因 |
| **线路二** | 特征重要性是真实因果吗？ | H4b：因果特征驱动预测 | 4.4.5-7 Granger+反事实 |
| **线路三** | 预测能力的边界条件？ | H4c：预测能力具有时变性 | 4.4.2/8/9/10 |

---

## 四、执行顺序建议

| 批次 | 任务编号 | 预计耗时 | 说明 |
|------|----------|----------|------|
| **Batch 1** | 4.1.1-4.1.6 | 30 min | 数据统计（可并行） |
| **Batch 2** | 4.2.1 | 30 min | 基准模型训练 |
| **Batch 3** | 4.2.2 | 2-4 h | 深度学习模型训练（最耗时） |
| **Batch 4** | 4.2.3a/b, 4.2.4-5 | 1 h | 消融实验 |
| **Batch 5** | 4.3.1-4.3.4 | 1 h | 回测与经济评估 |
| **Batch 6** | 4.4.1-4.4.11 | 2 h | 可解释性与稳健性 |

---

## 五、产出物核查清单

### 表格产出（18个）

- [ ] `tables/table_4.1-1_sample_stats.csv` 样本描述性统计
- [ ] `tables/table_4.1-2_feature_stats.csv` 特征描述性统计
- [ ] `tables/table_4.1-3_label_distribution.csv` 标签分布
- [ ] `tables/table_4.1-4_correlation.csv` 相关系数
- [ ] `tables/table_4.1-5_ols_regression.csv` OLS回归
- [ ] `tables/table_4.1-6_scale_comparison.csv` 多尺度解释力
- [ ] `tables/table_4.2-1_baseline_models.csv` 基准模型
- [ ] `tables/table_4.2-2_deep_models.csv` 深度模型
- [ ] `tables/table_4.2-3a_pv_crossattn.csv` PV-CrossAttention消融
- [ ] `tables/table_4.2-3b_lsf_ablation.csv` LSF消融
- [ ] `tables/table_4.2-4_feature_ablation.csv` 特征消融
- [ ] `tables/table_4.2-5_threshold_sensitivity.csv` 阈值敏感性
- [ ] `tables/table_4.3-1_backtest_config.csv` 回测参数
- [ ] `tables/table_4.3-2_backtest_results.csv` 经济价值
- [ ] `tables/table_4.3-4_scale_backtest.csv` 多尺度回测
- [ ] `tables/table_4.3-5_cost_sensitivity.csv` 成本敏感性
- [ ] `tables/table_4.4-3_regime_split.csv` 市场状态分组
- [ ] `tables/table_4.4-5_granger_causality.csv` Granger因果

### 图片产出（9个）

- [ ] `figures/fig_4.1-1_feature_distribution.png` 特征分布
- [ ] `figures/fig_4.1-2_scale_comparison.png` 尺度对比
- [ ] `figures/fig_4.2-3a_attention_heatmap.png` 注意力热力图
- [ ] `figures/fig_4.2-3b_scale_weights.png` 尺度权重时序
- [ ] `figures/fig_4.3-3_equity_curves.png` 净值曲线
- [ ] `figures/fig_4.3-6_cost_equity.png` 成本净值对比
- [ ] `figures/fig_4.4-1_shap_summary.png` SHAP归因
- [ ] `figures/fig_4.4-7_counterfactual.png` 反事实效应
- [ ] `figures/fig_4.4-8_decay_curve.png` 衰减曲线
- [ ] `figures/fig_4.4-11_rolling_vs_static.png` 滚动vs静态

---

## 六、执行日志

| 日期 | 完成任务 | 备注 |
|------|----------|------|
| 2026-01-21 | 清单更新 | 与论文大纲对齐，27个实验脚本已创建 |
| 2026-01-21 | 迁移位置 | 从todo_list/移动至outputs/ch4/ |

---

> **使用说明**：
> 1. 运行实验：`python run_all_experiments.py --exp 4.1.1`
> 2. 运行整节：`python run_all_experiments.py --section 4.1`
> 3. 运行全部：`python run_all_experiments.py`
