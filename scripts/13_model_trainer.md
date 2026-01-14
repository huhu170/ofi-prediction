# 模型训练方案说明

> 可以直接作为论文第三章"模型训练配置"部分的补充材料或技术附录使用
> 对应脚本：`13_model_trainer.py`  
> 版本：v1.0  
> 更新日期：2026-01-14

---

## 1. 概述

本模块实现了论文中所有基准模型和本研究模型的定义、训练与评估，支持一键对比所有模型性能。

### 1.1 模型体系

```
┌─────────────────────────────────────────────────────────────────┐
│                       模型层级体系                               │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: 机器学习基准                                          │
│           → Logistic Regression, XGBoost, Random Forest         │
│                                                                 │
│  Layer 2: 深度学习基准（RNN系列）                                │
│           → LSTM (2层), GRU (2层)                               │
│                                                                 │
│  Layer 3: 专用架构                                              │
│           → DeepLOB (CNN+LSTM, Zhang et al. 2019)               │
│                                                                 │
│  Layer 4: Attention机制                                         │
│           → Transformer (4层Encoder)                            │
│                                                                 │
│  Layer 5: 本研究模型                                            │
│           → Smart-Transformer (协方差加权)                       │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 与论文对齐

| 论文设定 | 本实现 | 对齐状态 |
|---------|--------|---------|
| LSTM 2层堆叠, hidden=128 | ✓ | ✅ |
| GRU 2层堆叠, hidden=128 | ✓ | ✅ |
| DeepLOB (Zhang 2019) | CNN(3层)+LSTM | ✅ |
| Transformer 4层, d=256, heads=8 | ✓ | ✅ |
| Smart-Trans + 协方差加权 | ✓ | ✅ |
| Adam优化器, lr=1e-4 | ✓ | ✅ |
| 早停 patience=10 | ✓ | ✅ |
| 学习率调度 ReduceLROnPlateau | ✓ | ✅ |

---

## 2. 模型架构详解

### 2.1 LSTM模型

```
Input: (batch, 100, 25)
         │
         ▼
┌─────────────────────┐
│  LSTM Layer 1       │  hidden=128, dropout=0.2
│  LSTM Layer 2       │
└──────────┬──────────┘
           │ 取最后时间步
           ▼
┌─────────────────────┐
│  FC: 128 → 64       │  ReLU + Dropout
│  FC: 64 → 3         │  (下跌/平稳/上涨)
└──────────┬──────────┘
           ▼
Output: (batch, 3)
```

**参数量**: ~220K

### 2.2 GRU模型

与LSTM结构相同，但使用GRU单元（参数更少）

**参数量**: ~170K

### 2.3 DeepLOB模型 (Zhang et al. 2019)

```
Input: (batch, 100, 25)
         │ permute → (batch, 25, 100)
         ▼
┌─────────────────────┐
│  Conv1D: 25→32      │  kernel=3, MaxPool(2)
│  Conv1D: 32→32      │  kernel=3, MaxPool(2)
│  Conv1D: 32→32      │  kernel=3
└──────────┬──────────┘
           │ permute → (batch, 25, 32)
           ▼
┌─────────────────────┐
│  LSTM: 32→64        │  1层
└──────────┬──────────┘
           │ 取最后时间步
           ▼
┌─────────────────────┐
│  FC: 64 → 32 → 3    │
└──────────┬──────────┘
           ▼
Output: (batch, 3)
```

**参数量**: ~30K

### 2.4 Transformer模型

```
Input: (batch, 100, 25)
         │
         ▼
┌─────────────────────┐
│  Linear: 25 → 256   │  输入嵌入
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  [CLS] + Sequence   │  添加分类token
│  Positional Encoding│  位置编码
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Transformer Encoder│  4层
│  - Multi-Head Attn  │  8 heads
│  - FFN: 256→512→256 │  GELU激活
│  - Dropout: 0.1     │
└──────────┬──────────┘
           │ 取CLS token
           ▼
┌─────────────────────┐
│  LayerNorm          │
│  FC: 256→128→3      │  GELU + Dropout
└──────────┬──────────┘
           ▼
Output: (batch, 3)
```

**参数量**: ~2.1M

### 2.5 Smart-Transformer（本研究）

架构与标准Transformer相同，但训练时使用**协方差加权损失**：

```python
# 标准损失
loss = CrossEntropyLoss(logits, labels)

# 协方差加权损失（本研究创新）
w_t = 1 + γ × max(0, ρ_stock_index)
loss_weighted = w_t × CrossEntropyLoss(logits, labels)
```

**经济含义**：
- 当个股与指数高度正相关时，样本权重增加
- 高相关样本更能反映"市场共振"信号
- γ=1.0时，权重范围为 [1.0, 2.0]

---

## 3. 训练策略

### 3.1 优化配置

| 配置项 | 取值 | 说明 |
|--------|------|------|
| 优化器 | Adam | β₁=0.9, β₂=0.999 |
| 初始学习率 | 1e-4 | 深度学习标准设置 |
| 权重衰减 | 1e-5 | L2正则化 |
| 批大小 | 64 | 平衡速度与稳定性 |
| 最大轮数 | 100 | 通常早停会提前终止 |

### 3.2 早停机制

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
    save_best_model()
else:
    patience_counter += 1
    if patience_counter >= 10:
        stop_training()
        restore_best_model()
```

### 3.3 学习率调度

```python
# 验证集损失连续5个epoch不下降时，学习率减半
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=5,
    factor=0.5
)
```

### 3.4 梯度裁剪

```python
# 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 4. 评估指标

### 4.1 统计精度指标

| 指标 | 公式 | 说明 |
|------|------|------|
| Accuracy | (TP+TN)/N | 整体正确率 |
| F1 (macro) | avg(F1_class) | 各类别F1的平均 |
| F1 (weighted) | weighted_avg(F1_class) | 加权平均 |
| AUC (macro) | avg(AUC_class) | One-vs-Rest AUC |

### 4.2 混淆矩阵解读

```
           预测
         下跌  平稳  上涨
  真实 下跌   TP₀  ...  ...
       平稳  ...  TP₁  ...
       上涨  ...  ...  TP₂
```

**关注点**：
- 对角线：正确分类数
- 非对角线：错误分类数
- 下跌→上涨、上涨→下跌：严重错误（方向完全相反）

---

## 5. 使用方法

### 5.1 命令行参数

```bash
python 13_model_trainer.py [OPTIONS]

Options:
  --data PATH          数据集目录
  --model MODEL [...]  要训练的模型 (lstm, gru, deeplob, transformer, smart_trans, all, ml)
  --epochs INTEGER     训练轮数（默认50）
  --batch-size INTEGER 批大小（默认64）
  --lr FLOAT           学习率（默认1e-4）
  --output PATH        模型保存目录（默认models）
  --compare            对比所有模型
```

### 5.2 使用示例

```bash
# 训练单个模型
python scripts/13_model_trainer.py --model lstm --epochs 50

# 训练多个模型
python scripts/13_model_trainer.py --model lstm gru transformer

# 训练所有深度学习模型
python scripts/13_model_trainer.py --model all --epochs 100

# 训练机器学习基准（需要sklearn）
python scripts/13_model_trainer.py --model ml

# 完整对比实验
python scripts/13_model_trainer.py --model all ml --compare
```

### 5.3 Python API使用

```python
from scripts.model_trainer import (
    create_model, Trainer, Evaluator, 
    load_dataset, create_dataloaders
)

# 加载数据
splits = load_dataset(Path('data/processed/combined/dataset_T100_k20'))
loaders = create_dataloaders(splits)

# 创建模型
model = create_model('transformer', input_dim=25, seq_len=100)

# 训练
trainer = Trainer(model)
history = trainer.train(loaders['train'], loaders['val'], epochs=50)

# 评估
evaluator = Evaluator()
metrics = evaluator.evaluate(model, loaders['test'])
evaluator.print_metrics(metrics, 'Transformer')
```

---

## 6. 输出文件

### 6.1 目录结构

```
models/
├── lstm/
│   ├── model.pt       # 模型权重
│   └── metrics.json   # 评估指标 + 训练历史
├── gru/
├── deeplob/
├── transformer/
├── smart_trans/
└── comparison.csv     # 所有模型对比表
```

### 6.2 metrics.json 格式

```json
{
  "accuracy": 0.4012,
  "f1_macro": 0.3856,
  "f1_weighted": 0.3901,
  "auc_macro": 0.5823,
  "confusion_matrix": [[20, 15, 15], [10, 25, 22], [8, 18, 34]],
  "history": {
    "train_loss": [1.09, 1.05, ...],
    "val_loss": [1.10, 1.03, ...],
    "train_acc": [0.35, 0.40, ...],
    "val_acc": [0.33, 0.42, ...]
  }
}
```

### 6.3 comparison.csv 格式

| Model | Accuracy | F1_macro | F1_weighted | AUC_macro |
|-------|----------|----------|-------------|-----------|
| smart_trans | 0.4251 | 0.4102 | 0.4156 | 0.6012 |
| transformer | 0.4012 | 0.3856 | 0.3901 | 0.5823 |
| lstm | 0.3892 | 0.3712 | 0.3789 | 0.5634 |
| ... | ... | ... | ... | ... |

---

## 7. 实测结果示例

以 HK.00700 单日数据（约1000样本）测试：

### 7.1 LSTM (10 epochs, CPU)

```
参数量: 219,907
训练时间: ~20秒

  Epoch   1: train_loss=1.0933, val_loss=1.1189, train_acc=0.3771
  Epoch  10: train_loss=0.9747, val_loss=0.9594, train_acc=0.5019

测试集:
  Accuracy:     0.3413
  混淆矩阵:
           预测
         下跌  平稳  上涨
  真实 下跌    6   22   22
       平稳   14    4   39
       上涨    4    9   47
```

**分析**：
- 单日数据量不足（~800训练样本）
- 三分类随机基准为33.3%
- 需要更多数据才能发挥模型能力

---

## 8. 注意事项

### 8.1 数据量要求

| 模型 | 建议最小样本数 | 原因 |
|------|---------------|------|
| Logistic | 1,000+ | 线性模型，过拟合风险低 |
| LSTM/GRU | 5,000+ | RNN需要学习时序模式 |
| Transformer | 10,000+ | 参数量大，需要更多数据 |

### 8.2 计算资源

| 模型 | CPU训练速度 | GPU训练速度 | 建议 |
|------|------------|------------|------|
| LSTM | ~2秒/epoch | ~0.3秒/epoch | CPU可用 |
| GRU | ~1.5秒/epoch | ~0.2秒/epoch | CPU可用 |
| DeepLOB | ~1秒/epoch | ~0.1秒/epoch | CPU可用 |
| Transformer | ~30秒/epoch | ~2秒/epoch | **建议GPU** |

### 8.3 标签转换

```
原始标签: -1 (下跌), 0 (平稳), 1 (上涨)
模型标签:  0 (下跌), 1 (平稳), 2 (上涨)

# 在数据加载时自动转换: y = y + 1
```

### 8.4 依赖项

```
必需:
- torch>=2.0.0         # 深度学习模型

可选:
- scikit-learn>=1.3.0  # 评估指标 + ML基准
- xgboost>=2.0.0       # XGBoost模型
```

---

## 9. 后续步骤

本模块训练完成后，可进行：

1. **`14_model_evaluator.py`** - 详细评估与可视化
2. **`15_backtest.py`** - 策略回测与经济价值评估
3. **`16_shap_analysis.py`** - SHAP特征归因分析

---

## 10. 常见问题

### Q1: sklearn未安装怎么办？

```bash
pip install scikit-learn
```

或直接使用深度学习模型（不依赖sklearn）：
```bash
python 13_model_trainer.py --model lstm  # 只需PyTorch
```

### Q2: Transformer训练太慢？

建议使用GPU：
```python
# 检查GPU是否可用
import torch
print(torch.cuda.is_available())  # 应为True
```

或减少模型参数：
```python
MODEL_CONFIG['transformer']['d_model'] = 128  # 减小维度
MODEL_CONFIG['transformer']['num_layers'] = 2  # 减少层数
```

### Q3: 准确率只有33%左右？

这是正常的，因为：
1. 三分类随机基准是33.3%
2. 数据量不足（需要更多交易日数据）
3. 金融预测本身具有挑战性

建议：积累更多数据后再训练。
