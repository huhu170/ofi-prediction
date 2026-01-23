# K线模型训练模块说明

## 概述

`13b_kline_model_trainer.py` 实现了论文第三章第四节定义的模型架构，包括：

**本研究创新模型**：
- **PV-CrossAttention**：量价交叉注意力（论文公式3.4-1）
- **LSF**：可学习尺度融合（论文公式3.4-2）
- **Learnable Positional Encoding**：可学习位置编码

**基准模型**（论文表3.4-1, 3.4-2a）：
- LSTM（2层，hidden=128）
- GRU（2层，hidden=128）
- CNN-LSTM（CNN 3层 + LSTM 1层）
- Transformer（4层，d_model=256）
- LogisticRegression
- RandomForest
- XGBoost

## 模型配置（与论文表3.4-3对齐）

```python
MODEL_CONFIG = {
    'pv_transformer': {
        'd_model': 256,          # 论文表3.4-3
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 512,
        'dropout': 0.1,
    },
    'multi_scale': {
        'd_model': 256,          # 各尺度编码器输出256维
        'nhead': 8,
        'num_layers': 2,
        'dropout': 0.1,
    },
    'transformer': {
        'd_model': 256,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 512,
        'dropout': 0.1,
    },
    'lstm': {'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.2},
    'gru': {'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.2},
}

TRAIN_CONFIG = {
    'batch_size': 64,
    'learning_rate': 1e-4,
    'max_epochs': 100,
    'early_stopping_patience': 10,  # 论文：连续10个epoch不下降时终止
    'weight_decay': 1e-5,
}
```

## 特征配置（与论文表3.3-1对齐，共22维）

```python
# 价格相关特征（14维）→ PV-CrossAttention的Query
PRICE_RELATED = [
    'kline_position', 'range_pct',                    # K线形态 (2)
    'return_1', 'return_5', 'return_20', 'return_60', 'return_zscore',  # 价格动量 (5)
    'atr_pct', 'volatility_20',                       # 波动率 (2)
    'rsi', 'bb_position', 'macd_dif', 'macd_dea', 'macd'  # 技术指标 (5)
]

# 成交量相关特征（8维）→ PV-CrossAttention的Key/Value
VOLUME_RELATED = [
    'ti', 'ti_5', 'ti_60', 'ti_zscore',              # 成交不平衡 (4)
    'relative_volume', 'volume_change', 'pv_corr',   # 成交量 (3)
    'market_regime'                                   # 市场状态 (1)
]
```

## GPU加速

脚本自动检测并使用GPU：

```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

- 自动使用CUDA（如果可用）
- DataLoader启用`pin_memory`优化
- 模型和数据自动转移到GPU

## 使用方法

```bash
# 本研究模型
python scripts/13b_kline_model_trainer.py --code HK.00700 --model pv_transformer
python scripts/13b_kline_model_trainer.py --code HK.00700 --model multi_scale

# 深度学习基准
python scripts/13b_kline_model_trainer.py --code HK.00700 --model lstm
python scripts/13b_kline_model_trainer.py --code HK.00700 --model gru
python scripts/13b_kline_model_trainer.py --code HK.00700 --model cnn_lstm
python scripts/13b_kline_model_trainer.py --code HK.00700 --model transformer

# sklearn基准
python scripts/13b_kline_model_trainer.py --code HK.00700 --model xgboost
python scripts/13b_kline_model_trainer.py --code HK.00700 --model random_forest
python scripts/13b_kline_model_trainer.py --code HK.00700 --model logistic_regression

# 批量训练所有股票
python scripts/13b_kline_model_trainer.py --all --model pv_transformer

# 自定义参数
python scripts/13b_kline_model_trainer.py --code HK.00700 --epochs 100 --batch-size 128
```

## 模型架构详解

### 1. PV-CrossAttention（论文公式3.4-1）

```
CrossAttn(Q_price, K_volume, V_volume) = softmax(Q × K^T / √d) × V
```

核心思想：价格序列询问"哪些成交量变化与当前价格走势相关"

### 2. LSF（论文公式3.4-2）

```
h_fused = Σ_s (w_s × h_s), w = softmax(MLP(concat([h_1, ..., h_S])))
```

特点：权重可解释，反映各时间尺度（1M/5M/60M/DAY）的动态贡献度

### 3. CNN-LSTM（论文表3.4-2a）

```
CNN(3层卷积) → LSTM(1层) → 分类头
```

特点：CNN提取局部特征模式，LSTM捕捉时序依赖

### 4. 模型工厂函数

```python
from scripts.13b_kline_model_trainer import create_model

# 创建任意模型
model = create_model('pv_transformer', input_dim=22, seq_len=60)
model = create_model('lstm', input_dim=22, seq_len=60)
model = create_model('cnn_lstm', input_dim=22, seq_len=60)
model = create_model('xgboost', input_dim=22, seq_len=60)
```

## 输出

```
models/
├── pv_transformer_HK_00700.pt
├── multi_scale_HK_00700.pt
├── lstm_HK_00700.pt
├── gru_HK_00700.pt
├── cnn_lstm_HK_00700.pt
├── transformer_HK_00700.pt
├── xgboost_HK_00700.pkl
├── random_forest_HK_00700.pkl
└── logistic_regression_HK_00700.pkl
```

## 与其他脚本的关系

```
12b_kline_dataset_builder.py  →  构建数据集（滑动窗口+标准化）
        ↓
13b_kline_model_trainer.py    →  模型训练（本脚本）
        ↓
14b_kline_backtest.py         →  策略回测
16b_kline_shap_analysis.py    →  SHAP可解释性分析
```
