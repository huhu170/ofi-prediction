# K线数据处理流程说明

> 创建时间：2026-01-21
> 目标：基于分钟级K线数据（1M/5M/60M/日K）的特征计算与模型训练

---

## 一、数据流程概览

```
04_fetch_kline.py         ← 从富途API拉取K线数据到数据库
        ↓
05_data_cleaner.py        ← K线数据清洗（异常、时段、缺失）
        ↓
06_feature_calculator.py  ← 计算论文定义的特征（TI、动量、波动率等）
        ↓
07_dataset_builder.py     ← 构建滑动窗口数据集（支持多尺度）
        ↓
08_model_trainer.py       ← 模型训练（PV-CrossAttention + LSF）
        ↓
09_backtest.py            ← 策略回测（含压力测试）
        ↓
10_shap_analysis.py       ← SHAP可解释性分析
```

---

## 二、与论文章节的对应关系

| 脚本 | 论文章节 | 主要功能 |
|------|----------|----------|
| `04_fetch_kline.py` | 3.1节 数据获取 | 拉取多周期K线数据 |
| `05_data_cleaner.py` | 3.1节 数据清洗 | 异常检测、时段过滤、缺失处理 |
| `06_feature_calculator.py` | 3.2节 特征工程 | 公式3.2-1~3.2-6的实现 |
| `07_dataset_builder.py` | 3.3节 数据划分 | 滚动窗口、时序划分 |
| `08_model_trainer.py` | 3.4节 模型架构 | PV-CrossAttention, LSF, Learnable PE |
| `09_backtest.py` | 4.3节 策略回测 | 交易成本、夏普比率、压力测试 |
| `10_shap_analysis.py` | 4.4节 可解释性 | SHAP特征归因、注意力可视化 |

---

## 三、核心公式实现

### 公式3.2-1：成交不平衡(TI)

```python
# 06_feature_calculator.py
def compute_ti(df):
    range_hl = df['high'] - df['low']
    ti = np.where(
        range_hl > 0,
        (df['close'] - df['open']) / range_hl * df['volume'],
        0.0
    )
    return ti
```

### 公式3.3-1：市场状态检测

```python
# regime = 0 if σ < Q50(σ)      # 平稳期
# regime = 2 if σ > Q90(σ)      # 高波动期
# regime = 1 otherwise          # 正常期
```

---

## 四、使用方法

### Step 1: 拉取K线数据

```bash
python 04_fetch_kline.py --code HK.00700
```

### Step 2: 数据清洗

```bash
# 单一K线类型
python 05_data_cleaner.py --code HK.00700 --ktype 1M

# 多尺度清洗
python 05_data_cleaner.py --code HK.00700 --multi-scale

# 清洗所有股票
python 05_data_cleaner.py --all --multi-scale
```

### Step 3: 计算特征

```bash
# 单尺度特征计算
python 06_feature_calculator.py --code HK.00700 --ktype 1M

# 多尺度特征计算（1M/5M/60M/DAY）
python 06_feature_calculator.py --code HK.00700 --multi-scale
```

### Step 4: 构建数据集

```bash
# 单尺度数据集
python 07_dataset_builder.py --code HK.00700 --ktype 1M --horizon 5

# 多尺度数据集（用于LSF模块）
python 07_dataset_builder.py --code HK.00700 --multi-scale
```

### Step 5: 模型训练

```bash
# PV-Transformer（含PV-CrossAttention）
python 08_model_trainer.py --code HK.00700 --model pv_transformer

# LSTM基准模型
python 08_model_trainer.py --code HK.00700 --model lstm

# 自定义训练参数
python 08_model_trainer.py --code HK.00700 --epochs 100 --lr 0.0001
```

### Step 6: 策略回测

```bash
# 基准回测
python 09_backtest.py --model models/pv_transformer/model.pt

# 交易成本压力测试
python 09_backtest.py --model models/pv_transformer/model.pt --stress-test

# 自定义交易成本
python 09_backtest.py --model models/pv_transformer/model.pt --cost 0.001
```

### Step 7: SHAP可解释性分析

```bash
# SHAP分析
python 10_shap_analysis.py --model models/pv_transformer/model.pt

# 指定分析样本数
python 10_shap_analysis.py --model models/pv_transformer/model.pt --samples 200
```

---

## 五、输出特征列表

### K线特征（22维）

| 类别 | 特征名 | 公式 | 说明 |
|------|--------|------|------|
| **TI** | `ti` | 3.2-1 | 成交不平衡 |
| | `ti_5, ti_20, ti_60` | - | 累积TI |
| | `ti_zscore` | - | 标准化TI |
| **收益率** | `return_1, 5, 20, 60` | 3.2-2 | 多周期收益率 |
| | `return_zscore` | - | 标准化收益率 |
| **成交量** | `relative_volume` | 3.2-3 | 相对成交量 |
| | `pv_corr` | 3.2-4 | 量价相关性 |
| **波动率** | `atr` | 3.2-5 | ATR(14) |
| | `range_pct` | 3.2-6 | 日内波幅比 |
| **技术指标** | `rsi, macd, bb_position` | - | RSI、MACD、布林带 |
| **状态** | `market_regime` | 3.3-1 | 市场状态(0/1/2) |

---

## 六、注意事项

1. **边界条件处理**：TI公式在H=L时会除零，已在代码中处理
2. **防止数据泄露**：标准化只在训练集上fit，验证/测试集使用训练集参数transform
3. **时序划分**：严格按时间顺序划分，不随机打乱
4. **多尺度对齐**：不同K线周期的样本数可能不同，需按最小长度对齐
