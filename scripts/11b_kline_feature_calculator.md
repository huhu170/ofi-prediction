# K线特征计算模块说明

## 概述

`11b_kline_feature_calculator.py` 实现了论文第三章中基于分钟级K线数据的特征计算体系。

## 与论文公式的对应关系

| 论文公式 | 特征名称 | 实现方法 |
|----------|----------|----------|
| **3.2-1** | 成交不平衡(TI) | `TI = (C - O) / (H - L) × V`，边界条件：H=L时TI=0 |
| **3.2-2** | 多周期收益率 | `r^(k) = (C_t - C_{t-k}) / C_{t-k}`, k∈{1,5,20,60} |
| **3.2-3** | 相对成交量 | `RV = V / MA_20(V)` |
| **3.2-4** | 量价相关性 | `ρ_PV = Corr_20(r, V)` |
| **3.2-5** | 真实波幅(ATR) | `TR = max(H-L, |H-C_{-1}|, |L-C_{-1}|)`, `ATR = EMA_14(TR)` |
| **3.2-6** | 日内波幅比 | `Range = (H - L) / O` |
| **3.3-1** | 市场状态 | 基于波动率分位数划分：0=平稳期, 1=正常期, 2=高波动期 |
| **3.1-2** | 预测标签 | 三分类：上涨(+1), 平稳(0), 下跌(-1), 阈值α=0.002 |

## 输出特征列表（共43维）

### 1. 成交不平衡特征（5维）
| 特征 | 类型 | 说明 |
|------|------|------|
| `ti` | float | 单周期成交不平衡 |
| `ti_5` | float | 5周期累积TI |
| `ti_20` | float | 20周期累积TI |
| `ti_60` | float | 60周期累积TI |
| `ti_zscore` | float | TI的Z-score |

### 2. 收益率特征（7维）
| 特征 | 类型 | 说明 |
|------|------|------|
| `return_1` | float | 1周期收益率 |
| `return_5` | float | 5周期收益率 |
| `return_20` | float | 20周期收益率 |
| `return_60` | float | 60周期收益率 |
| `return_ma_20` | float | 收益率20期均值 |
| `return_std_20` | float | 收益率20期标准差 |
| `return_zscore` | float | 收益率Z-score |

### 3. 成交量特征（3维）
| 特征 | 类型 | 说明 |
|------|------|------|
| `relative_volume` | float | 相对成交量 |
| `volume_change` | float | 成交量变化率 |
| `pv_corr` | float | 量价相关性 |

### 4. 波动率特征（5维）
| 特征 | 类型 | 说明 |
|------|------|------|
| `tr` | float | 真实波幅 |
| `atr` | float | ATR(14) |
| `atr_pct` | float | ATR占价格比例 |
| `range_pct` | float | 日内波幅比 |
| `volatility_20` | float | 20期滚动波动率 |

### 5. 技术指标（5维）
| 特征 | 类型 | 说明 |
|------|------|------|
| `rsi` | float | RSI(14) |
| `macd_dif` | float | MACD的DIF |
| `macd_dea` | float | MACD的DEA |
| `macd` | float | MACD柱 |
| `bb_position` | float | 布林带位置 |

### 6. 市场状态（1维）
| 特征 | 类型 | 说明 |
|------|------|------|
| `market_regime` | int | 市场状态：0=平稳, 1=正常, 2=高波动 |

## 使用方法

```bash
# 单一K线类型
python 11b_kline_feature_calculator.py --code HK.00700 --ktype 1M

# 多尺度特征（1M/5M/60M/DAY）
python 11b_kline_feature_calculator.py --code HK.00700 --multi-scale

# 指定日期范围
python 11b_kline_feature_calculator.py --code HK.00700 --ktype 1M \
    --start 2024-01-01 --end 2024-12-31

# 自定义标签阈值
python 11b_kline_feature_calculator.py --code HK.00700 --alpha 0.003
```

## 边界条件处理

### TI公式的除零问题

当K线无波动时（H = L），公式3.2-1的分母为零。处理方式：

```python
ti = np.where(
    range_hl > 0,
    (close - open) / range_hl * volume,
    0.0  # H = L 时，TI = 0（无主导方向）
)
```

## 与其他脚本的关系

```
08_fetch_kline_10years.py  →  拉取K线数据到数据库
        ↓
11b_kline_feature_calculator.py  →  计算K线特征（本脚本）
        ↓
12b_kline_dataset_builder.py  →  构建训练数据集（待创建）
        ↓
13_model_trainer.py  →  模型训练
```
