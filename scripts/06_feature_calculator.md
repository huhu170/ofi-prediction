# K线特征计算模块说明

## 概述

`11b_kline_feature_calculator.py` 实现了论文第三章中基于分钟级K线数据的特征计算体系。

## 与论文公式的对应关系

| 论文公式 | 特征名称 | 实现方法 |
|----------|----------|----------|
| **表3.3-1** | K线形态 | `kline_position = (C - O) / (H - L)`，不乘成交量 |
| **3.2-1** | 成交不平衡(TI) | `TI = (C - O) / (H - L) × V`，边界条件：H=L时TI=0 |
| **3.2-2** | 多周期收益率 | `r^(k) = (C_t - C_{t-k}) / C_{t-k}`, k∈{1,5,20,60} |
| **3.2-3** | 相对成交量 | `RV = V / MA_20(V)` |
| **3.2-4** | 量价相关性 | `ρ_PV = Corr_20(r, V)` |
| **3.2-5** | 真实波幅(ATR) | `TR = max(H-L, |H-C_{-1}|, |L-C_{-1}|)`, `ATR = EMA_14(TR)` |
| **3.2-6** | 日内波幅比 | `Range = (H - L) / O` |
| **3.3-1** | 市场状态 | 基于波动率分位数划分：0=平稳期, 1=正常期, 2=高波动期 |
| **3.1-2** | 预测标签 | 三分类：上涨(+1), 平稳(0), 下跌(-1), 阈值α=0.002 |

## 模型输入特征列表（22维，与论文表3.3-1对齐）

> 注：以下22维为最终输入模型的特征，与论文表3.4-3对齐

| 类别 | 特征 | 维度 | 说明 |
|------|------|------|------|
| **价格动量** | `return_1`, `return_5`, `return_20`, `return_60` | 4 | 多周期收益率 |
| **K线形态** | `kline_position`, `range_pct` | 2 | 价格位置 + 波幅比 |
| **成交量** | `relative_volume`, `volume_change` | 2 | 相对量 + 变化率 |
| **量价特征** | `ti`, `ti_5`, `ti_60`, `pv_corr` | 4 | TI + 累积TI + 量价相关 |
| **波动特征** | `atr_pct`, `volatility_20` | 2 | ATR比例 + 滚动波动率 |
| **技术指标** | `rsi`, `macd_dif`, `macd_dea`, `macd`, `bb_position` | 5 | 经典技术指标 |
| **滚动统计** | `ti_zscore`, `return_zscore` | 2 | Z-score标准化 |
| **市场状态** | `market_regime` | 1 | 0=平稳, 1=正常, 2=高波动 |
| **合计** | | **22** | |

## 使用方法

```bash
# 单只股票、单一K线类型
python 11b_kline_feature_calculator.py --code HK.00700 --ktype 1M

# 单只股票、多尺度特征（1M/5M/60M/DAY）
python 11b_kline_feature_calculator.py --code HK.00700 --multi-scale

# 所有股票、多尺度（批量处理）
python 11b_kline_feature_calculator.py --all --multi-scale

# 指定日期范围
python 11b_kline_feature_calculator.py --code HK.00700 --ktype 1M \
    --start 2024-01-01 --end 2024-12-31

# 自定义标签阈值
python 11b_kline_feature_calculator.py --code HK.00700 --alpha 0.003
```

## 边界条件处理

### TI公式与K线形态的除零问题

当K线无波动时（H = L），公式分母为零。处理方式：

```python
# kline_position: 不乘成交量
kline_pos = np.where(range_hl > 0, (close - open) / range_hl, 0.0)

# TI: 乘成交量
ti = np.where(range_hl > 0, (close - open) / range_hl * volume, 0.0)
```

## 输出路径

```
data/processed/
└── HK_00700/
    ├── kline_features_1M.parquet   # 包含22维特征 + 标签
    ├── kline_features_5M.parquet
    ├── kline_features_60M.parquet
    └── kline_features_DAY.parquet
```

## 与其他脚本的关系

```
08_fetch_kline_10years.py  →  拉取K线数据到数据库
        ↓
10b_kline_data_cleaner.py  →  数据清洗
        ↓
11b_kline_feature_calculator.py  →  计算特征（本脚本，22维）
        ↓
12b_kline_dataset_builder.py  →  构建训练数据集
        ↓
13b_kline_model_trainer.py  →  模型训练
```
