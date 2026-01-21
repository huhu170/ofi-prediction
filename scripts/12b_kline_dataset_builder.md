# K线数据集构建模块说明

## 概述

`12b_kline_dataset_builder.py` 将K线特征数据转换为深度学习模型所需的滑动窗口序列格式，支持单尺度和多尺度（LSF模块）数据集构建。

## 与论文的对应关系

| 配置项 | 论文设定 | 实现 |
|--------|----------|------|
| 1分钟K线输入长度 | T = 60 (1小时) | `input_len = 60` |
| 5分钟K线输入长度 | T = 24 (2小时) | `input_len = 24` |
| 60分钟K线输入长度 | T = 12 (12小时) | `input_len = 12` |
| 日K输入长度 | T = 20 (1个月) | `input_len = 20` |
| 预测步长 | k ∈ {5, 15, 30}分钟 | `horizons = [5, 15, 30]` |
| 数据划分 | 7:1.5:1.5 | `train=0.7, val=0.15, test=0.15` |

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

## 使用方法

```bash
# 单尺度数据集
python 12b_kline_dataset_builder.py --code HK.00700 --ktype 1M --horizon 5

# 多尺度数据集（用于LSF模块）
python 12b_kline_dataset_builder.py --code HK.00700 --multi-scale

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
2. **滚动标准化**：只在训练集上fit，验证/测试集用训练集参数transform
3. **标签对齐**：标签使用未来k步的收益率，确保无前视偏差

```python
class RollingScaler:
    def fit(self, X_train):
        """只在训练集上计算统计量"""
        self.mean = X_train.mean(axis=0)
        self.std = X_train.std(axis=0)
    
    def transform(self, X):
        """应用训练集的统计量"""
        return (X - self.mean) / self.std
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
