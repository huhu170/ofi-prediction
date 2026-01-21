# K线模型训练模块说明

## 概述

`13b_kline_model_trainer.py` 实现了论文第三章第四节定义的创新模型架构，包括：
- **PV-CrossAttention**：量价交叉注意力
- **LSF**：可学习尺度融合
- **Learnable Positional Encoding**：可学习位置编码

## 模型架构

### 1. PV-CrossAttention（论文公式3.4-1）

```
CrossAttn(Q_price, K_volume, V_volume) = softmax(Q × K^T / √d) × V
```

**核心思想**：
- 价格序列作为Query：询问"哪些成交量变化与当前价格走势相关"
- 成交量序列作为Key/Value：提供供需失衡信息

```python
class PVCrossAttention(nn.Module):
    def forward(self, price_features, volume_features):
        Q = self.W_q(price_features)  # 价格 → Query
        K = self.W_k(volume_features) # 成交量 → Key
        V = self.W_v(volume_features) # 成交量 → Value
        
        attn = softmax(Q @ K.T / sqrt(d_k)) @ V
        return attn
```

### 2. LSF（论文公式3.4-2）

```
h_fused = Σ_s (w_s × h_s), w = softmax(MLP(concat([h_1, ..., h_S])))
```

**特点**：
- 权重w之和为1（softmax保证）
- 权重可解释：反映各时间尺度的贡献度
- 动态权重：根据输入自适应调整

```python
class LearnableScaleFusion(nn.Module):
    def forward(self, scale_features):
        concat = torch.cat(scale_features, dim=-1)
        weights = F.softmax(self.gate(concat), dim=-1)  # Σw = 1
        fused = (stacked * weights).sum(dim=-1)
        return fused, weights  # 权重可用于可解释性
```

### 3. 可学习位置编码

```python
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
    
    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]
```

## 模型配置

```python
MODEL_CONFIG = {
    'pv_transformer': {
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 256,
        'dropout': 0.1,
    },
    'multi_scale': {
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dropout': 0.1,
    }
}

TRAIN_CONFIG = {
    'batch_size': 64,
    'learning_rate': 1e-4,
    'max_epochs': 100,
    'early_stopping_patience': 15,
    'weight_decay': 1e-5,
}
```

## 使用方法

```bash
# PV-Transformer（含PV-CrossAttention）
python 13b_kline_model_trainer.py --code HK.00700 --model pv_transformer

# LSTM基准模型
python 13b_kline_model_trainer.py --code HK.00700 --model lstm

# 自定义训练参数
python 13b_kline_model_trainer.py --code HK.00700 --epochs 100 --lr 0.0001 --batch-size 128

# 指定数据集路径
python 13b_kline_model_trainer.py --dataset data/datasets/dataset_HK_00700_1M.pkl
```

## 特征分组

模型输入自动分离为价格特征和成交量特征：

```python
# 价格相关特征 → PV-CrossAttention的Query
PRICE_FEATURES = ['return_1', 'return_5', 'return_20', 'return_zscore', 
                  'atr_pct', 'range_pct', 'rsi', 'bb_position']

# 成交量相关特征 → PV-CrossAttention的Key/Value
VOLUME_FEATURES = ['ti', 'ti_5', 'ti_20', 'ti_zscore', 
                   'relative_volume', 'volume_change', 'pv_corr']
```

## 输出

```
models/
├── pv_transformer/
│   └── model_HK_00700_1M.pt
├── lstm/
│   └── model_HK_00700_1M.pt
└── multi_scale/
    └── model_HK_00700_multi.pt
```

保存内容：
```python
{
    'model_state_dict': ...,
    'model_name': 'pv_transformer',
    'best_val_f1': 0.65,
    'history': {'train_loss': [...], 'val_loss': [...], ...}
}
```

## 与其他脚本的关系

```
12b_kline_dataset_builder.py  →  构建数据集
        ↓
13b_kline_model_trainer.py    →  模型训练（本脚本）
        ↓
14b_kline_backtest.py         →  策略回测
16b_kline_shap_analysis.py    →  SHAP分析
```
