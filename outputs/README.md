# 论文输出资产管理

> 📅 更新时间：2026-01-21
> 📊 论文题目：**基于Transformer的港股分钟级价格预测：量价交叉注意力与多尺度融合方法**

---

## 一、整体进度概览

| 章节 | 内容 | 文本进度 | 图表进度 | TODO位置 |
|------|------|----------|----------|----------|
| 第一章 | 绪论 | ✅ 95% | ⬜ 0/4 | [ch1/TODO.md](ch1/TODO.md) |
| 第二章 | 文献综述 | ✅ 100% | N/A | [ch2/TODO.md](ch2/TODO.md) |
| 第三章 | 研究设计 | ✅ 85% | ⬜ 0/12 | [ch3/TODO.md](ch3/TODO.md) |
| 第四章 | 实证分析 | ⬜ 60% | ⬜ 0/27 | [ch4/TODO.md](ch4/TODO.md) |
| 第五章 | 结论与展望 | ⬜ 70% | N/A | [ch5/TODO.md](ch5/TODO.md) |
| **总计** | - | **~82%** | **0/43** | - |

---

## 二、目录结构

```
outputs/
├── README.md              # 本文件：项目总索引
│
├── ch1/                   # 第一章：绪论
│   ├── TODO.md            # 4项待办（图3 + 表1）
│   ├── figures/           # 图片输出
│   └── tables/            # 表格输出
│
├── ch2/                   # 第二章：文献综述
│   ├── TODO.md            # 无图表任务
│   ├── figures/
│   └── tables/
│
├── ch3/                   # 第三章：研究设计
│   ├── TODO.md            # 12项待办（图2 + 表10）
│   ├── figures/
│   └── tables/
│
├── ch4/                   # 第四章：实证分析
│   ├── TODO.md            # 27项待办（图9 + 表18）
│   ├── figures/           # 实验图片输出
│   └── tables/            # 实验表格输出
│
├── ch5/                   # 第五章：结论与展望
│   ├── TODO.md            # 无图表任务
│   ├── figures/
│   └── tables/
│
└── drawio/                # 绘图源文件
    ├── ch1/               # 第一章流程图源文件
    └── ch3/               # 第三章架构图源文件
```

---

## 三、命名规范

| 类型 | 格式 | 示例 |
|------|------|------|
| **图片** | `fig_章.节-序号_英文描述.png` | `fig_1.2-1_research_framework.png` |
| **表格** | `table_章.节-序号_英文描述.csv` | `table_4.1-1_sample_stats.csv` |
| **源文件** | `fig_章.节-序号_英文描述.drawio` | `fig_1.2-1_research_framework.drawio` |

---

## 四、图表工具建议

| 图类型 | 推荐工具 | 适用章节 |
|--------|----------|----------|
| 流程图/架构图 | Draw.io / Visio | ch1, ch3 |
| 数据分布图 | Python (matplotlib/seaborn) | ch4 |
| 模型对比图 | Python (matplotlib) | ch4 |
| 热力图/权重图 | Python (seaborn/plotly) | ch4 |
| SHAP可视化 | Python (shap库) | ch4 |
| 净值曲线 | Python (matplotlib) | ch4 |

---

## 五、关键里程碑

| 里程碑 | 内容 | 状态 |
|--------|------|------|
| M1 | 代码框架完成 | ✅ |
| M2 | 4.1-4.2节实验完成 | ⬜ |
| M3 | 4.3-4.4节实验完成 | ⬜ |
| M4 | 全部图表完成 | ⬜ |
| M5 | 论文初稿完成 | ⬜ |
| M6 | 格式审核完成 | ⬜ |

---

## 六、图表总清单

### 第一章（4个）

| 编号 | 文件路径 | 类型 | 工具 | 状态 |
|------|----------|------|------|------|
| 图1.2-1 | `ch1/figures/fig_1.2-1_research_framework.png` | 流程图 | Draw.io | ⬜ |
| 图1.2-2 | `ch1/figures/fig_1.2-2_experiment_design.png` | 流程图 | Draw.io | ⬜ |
| 图1.3-1 | `ch1/figures/fig_1.3-1_innovation_framework.png` | 流程图 | Draw.io | ⬜ |
| 表1.1-1 | `ch1/tables/table_1.1-1_rq_hypothesis.csv` | 表格 | 手工 | ⬜ |

### 第三章（12个）

| 编号 | 文件路径 | 类型 | 工具 | 状态 |
|------|----------|------|------|------|
| 表3.1-1 | `ch3/tables/table_3.1-1_sample_overview.csv` | 表格 | 脚本 | ⬜ |
| 表3.1-2 | `ch3/tables/table_3.1-2_data_quality.csv` | 表格 | 脚本 | ⬜ |
| 图3.1-1 | `ch3/figures/fig_3.1-1_sliding_window.png` | 示意图 | Draw.io | ⬜ |
| 表3.2-1 | `ch3/tables/table_3.2-1_feature_definition.csv` | 表格 | 手工 | ⬜ |
| 表3.2-2 | `ch3/tables/table_3.2-2_feature_groups.csv` | 表格 | 手工 | ⬜ |
| 表3.3-1 | `ch3/tables/table_3.3-1_label_definition.csv` | 表格 | 手工 | ⬜ |
| 表3.3-2 | `ch3/tables/table_3.3-2_market_state.csv` | 表格 | 手工 | ⬜ |
| 图3.4-1 | `ch3/figures/fig_3.4-1_model_architecture.png` | 架构图 | Draw.io | ⬜ |
| 表3.4-1 | `ch3/tables/table_3.4-1_hyperparameters.csv` | 表格 | 手工 | ⬜ |
| 表3.4-2 | `ch3/tables/table_3.4-2_baseline_config.csv` | 表格 | 手工 | ⬜ |
| 表3.5-1 | `ch3/tables/table_3.5-1_rolling_window.csv` | 表格 | 手工 | ⬜ |
| 表3.5-2 | `ch3/tables/table_3.5-2_reproducibility.csv` | 表格 | 手工 | ⬜ |

### 第四章（27个）

详见 [ch4/TODO.md](ch4/TODO.md)

---

## 七、实验执行命令

```bash
cd experiments

# 列出所有实验
python run_all_experiments.py --list

# 运行单个实验
python run_all_experiments.py --exp 4.1.1

# 运行某节所有实验
python run_all_experiments.py --section 4.1

# 运行全部实验
python run_all_experiments.py
```

---

## 八、更新日志

| 日期 | 操作 | 备注 |
|------|------|------|
| 2026-01-21 | 创建目录结构 | 整合experiment_results/backtest_results/shap_results/todo_list |
| 2026-01-21 | 完成实验脚本 | 27个实验脚本已创建 |
| 2026-01-21 | 更新第二章进度 | 文献综述已完成100% |

---

> **注意事项**：
> - 实验脚本依赖K线数据，需先运行 `scripts/08_fetch_kline_10years.py`
> - 数据路径：`data/processed/HK_*/kline_*.parquet`
> - 所有实验使用固定随机种子 `seed=42`
> - 环境依赖：见 `requirements.txt`
