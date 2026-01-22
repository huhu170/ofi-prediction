# K线数据集构建模块说明

## 概述

`12b_kline_dataset_builder.py` 将K线特征数据转换为深度学习模型所需的滑动窗口序列格式，支持单尺度和多尺度（LSF模块）数据集构建。

## 与论文的对应关系

| 配置项 | 论文设定 | 实现 | 说明 |
|--------|----------|------|------|
| 输入特征维度 | 22维 | `KLINE_FEATURE_COLS` | 见论文表3.3-1 |
| 1分钟K线输入长度 | T = 60 (1小时) | `input_len = 60` | |
| 5分钟K线输入长度 | T = 24 (2小时) | `input_len = 24` | |
| 60分钟K线输入长度 | T = 12 (12小时) | `input_len = 12` | |
| 日K输入长度 | T = 20 (1个月) | `input_len = 20` | |
| 预测步长 | k ∈ {5, 15, 30}分钟 | `horizons = [5, 15, 30]` | |
| 数据划分 | 70:15:15 | `train=0.7, val=0.15, test=0.15` | |
| **Gap** | **30分钟** | `DEFAULT_GAP = 30` | **避免标签泄漏** |
| 标签编码 | {-1,0,+1} → {0,1,2} | `map_labels_for_pytorch()` | PyTorch兼容 |

## 多尺度配置

```python
KLINE_SCALES = {
    '1M': {
        'name': '1分钟',
        'input_len': 60,      # 60根 = 1小时
        'horizon_steps': {5: 5, 15: 15, 30: 30}
    },
    '5M': {
        'name': '5分钟',
        'input_len': 24,      # 24根 = 2小时
        'horizon_steps': {5: 1, 15: 3, 30: 6}
    },
    '60M': {
        'name': '60分钟',
        'input_len': 12,      # 12根 = 12小时
        'horizon_steps': {5: 1, 15: 1, 30: 1}
    },
    'DAY': {
        'name': '日K',
        'input_len': 20,      # 20根 = 1个月
        'horizon_steps': {5: 1, 15: 1, 30: 1}
    },
}
```

## 输入特征列表（22维，与论文表3.3-1对齐）

```python
KLINE_FEATURE_COLS = [
    # 价格动量 (4维)
    'return_1', 'return_5', 'return_20', 'return_60',
    # K线形态 (2维)
    'kline_position', 'range_pct',
    # 成交量 (2维)
    'relative_volume', 'volume_change',
    # 量价特征 (4维)
    'ti', 'ti_5', 'ti_60', 'pv_corr',
    # 波动特征 (2维)
    'atr_pct', 'volatility_20',
    # 技术指标 (5维)
    'rsi', 'macd_dif', 'macd_dea', 'macd', 'bb_position',
    # 滚动统计 (2维)
    'ti_zscore', 'return_zscore',
    # 市场状态 (1维)
    'market_regime'
]  # 总计: 22维
```

## 使用方法

```bash
# 单只股票、单尺度数据集
python 12b_kline_dataset_builder.py --code HK.00700 --ktype 1M --horizon 5

# 单只股票、多尺度数据集（用于LSF模块）
python 12b_kline_dataset_builder.py --code HK.00700 --multi-scale

# 所有股票、多尺度（批量处理）
python 12b_kline_dataset_builder.py --all --multi-scale

# 自定义输出目录
python 12b_kline_dataset_builder.py --code HK.00700 --output data/my_datasets
```

## 数据集格式

### 单尺度数据集

```python
{
    'train': KlineDataset(X, y),  # PyTorch Dataset
    'val': KlineDataset(X, y),
    'test': KlineDataset(X, y),
    'scaler': RollingScaler,      # 标准化参数
    'feature_names': ['ti', 'return_1', ...],
}
```

### 多尺度数据集

```python
{
    'train': {
        '1M': np.ndarray,   # (N, 60, F)
        '5M': np.ndarray,   # (N, 24, F)
        '60M': np.ndarray,  # (N, 12, F)
        'DAY': np.ndarray,  # (N, 20, F)
        'labels': np.ndarray,
    },
    'val': {...},
    'test': {...},
}
```

## 防止数据泄露

1. **时序划分**：严格按时间顺序划分，不随机打乱
2. **Gap间隔**：训练/验证/测试集之间保留30分钟间隔（论文要求）
3. **滚动标准化**：只在训练集上fit，验证/测试集用训练集参数transform
4. **标签对齐**：标签使用未来k步的收益率，确保无前视偏差

```python
# Gap配置（论文第三章第一节第5部分）
DEFAULT_GAP = 30  # 30分钟，避免标签泄漏

class TimeSeriesSplitter:
    def __init__(self, gap=DEFAULT_GAP):  # 默认Gap=30
        self.gap = gap
    
    def split(self, X, y):
        return {
            'train': (X[:train_end], y[:train_end]),
            'val': (X[train_end + self.gap : val_end], ...),  # Gap间隔
            'test': (X[val_end + self.gap:], ...),
        }
```

## 标签编码映射

论文标签 `{-1, 0, +1}` → PyTorch标签 `{0, 1, 2}`：

```python
def map_labels_for_pytorch(y):
    """映射: -1→0(下跌), 0→1(平稳), +1→2(上涨)"""
    return (y + 1).astype(int)
```

## 输出路径

```
data/datasets/
├── dataset_HK_00700_1M.pkl       # 单尺度
├── dataset_HK_00700_5M.pkl
├── dataset_HK_00700_60M.pkl
├── dataset_HK_00700_DAY.pkl
├── dataset_HK_00700_multi_scale.pkl  # 多尺度
└── scaler_HK_00700_1M.pkl        # 标准化参数
```

## 与其他脚本的关系

```
11b_kline_feature_calculator.py  →  计算特征
        ↓
12b_kline_dataset_builder.py     →  构建数据集（本脚本）
        ↓
13b_kline_model_trainer.py       →  模型训练
```
