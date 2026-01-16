# 数据集构建方案说明

> 可以直接作为论文第三章"模型训练配置"部分的补充材料或技术附录使用
> 对应脚本：`12_dataset_builder.py`  
> 版本：v1.0  
> 更新日期：2026-01-14

---

## 1. 概述

本模块将特征数据转换为深度学习模型所需的**滑动窗口序列格式**，实现从"时间点级数据"到"样本级数据"的转换。

### 1.1 核心功能

```
┌─────────────────────────────────────────────────────────────────┐
│                       数据集构建流程                             │
├─────────────────────────────────────────────────────────────────┤
│  Step 1: 滑动窗口切片                                           │
│          → 将时序数据切成 (T=100, F=43) 的输入序列               │
│                                                                 │
│  Step 2: 时序数据划分                                           │
│          → 训练70% / 验证15% / 测试15%                          │
│          → 严格按时间顺序，防止数据泄露                          │
│                                                                 │
│  Step 3: 特征标准化                                             │
│          → Z-score标准化（仅在训练集上拟合）                     │
│                                                                 │
│  Step 4: 格式转换                                               │
│          → NumPy数组 + PyTorch Dataset + DataLoader             │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 输入输出

| 项目 | 格式 | 说明 |
|------|------|------|
| **输入** | `features_*.parquet` | 特征计算模块输出的Parquet文件 |
| **输出** | `X_{split}.npy` | NumPy格式的特征数组 |
| **输出** | `y_{split}.npy` | NumPy格式的标签数组 |
| **输出** | `dataset_{split}.pt` | PyTorch Dataset对象 |
| **输出** | `scaler.pkl` | 标准化参数（用于推理）|
| **输出** | `config.pkl` | 数据集配置参数 |

---

## 2. 滑动窗口原理

### 2.1 为什么需要滑动窗口？

深度学习模型（如Transformer、LSTM）需要**序列输入**，而原始数据是**逐时间点的记录**：

```
原始数据（每行一个时间点）:
┌──────────┬──────────┬──────────┬─────────┐
│    ts    │  ofi_l1  │ smart_ofi│ label_20│
├──────────┼──────────┼──────────┼─────────┤
│ 09:30:10 │   1000   │   500    │    1    │
│ 09:30:20 │   1200   │   600    │    0    │
│ 09:30:30 │   800    │   400    │   -1    │
│   ...    │   ...    │   ...    │   ...   │
│ 09:47:00 │   950    │   480    │    1    │  ← t=100
│ 09:47:10 │   1100   │   550    │    0    │  ← t=101
└──────────┴──────────┴──────────┴─────────┘

滑动窗口转换后（每行一个样本）:
┌────────────────────────────────────┬─────────┐
│         X (T=100, F=43)            │    y    │
├────────────────────────────────────┼─────────┤
│ [t1 features, t2, ..., t100]       │ label_120 │  ← Sample 1
│ [t2 features, t3, ..., t101]       │ label_121 │  ← Sample 2
│ [t3 features, t4, ..., t102]       │ label_122 │  ← Sample 3
│   ...                              │   ...   │
└────────────────────────────────────┴─────────┘
```

### 2.2 关键参数

| 参数 | 默认值 | 含义 | 论文设定 |
|------|--------|------|---------|
| `seq_len` (T) | 100 | 输入序列长度 | 100步 ≈ 16.7分钟 |
| `horizon` (k) | 20 | 预测步长 | k ∈ {20, 50, 100} |
| `step` | 1 | 滑动步长 | 每步生成一个样本 |

### 2.3 样本数计算

```
num_samples = (N - T - k + 1) / step

示例：
- 原始数据: N = 1,230 个时间点
- 序列长度: T = 100
- 预测步长: k = 20
- 滑动步长: step = 1

→ num_samples = (1230 - 100 - 20 + 1) / 1 = 1,111 个样本
```

---

## 3. 时序数据划分

### 3.1 为什么不用随机划分？

**随机划分会导致数据泄露**：

```
❌ 随机划分问题：
训练集可能包含 t=500 的样本
测试集可能包含 t=480 的样本
→ 模型在训练时"偷看"了测试时间段的未来信息

✅ 时序划分正确做法：
训练集: t = 1 ~ 700
验证集: t = 701 ~ 850  
测试集: t = 851 ~ 1000
→ 模型只用过去数据预测未来
```

### 3.2 划分比例

```
┌────────────────┬────────────────┬────────────────┐
│    训练集       │    验证集       │    测试集       │
│    70%         │    15%         │    15%         │
├────────────────┼────────────────┼────────────────┤
│ 历史 ─────────────────────────────────────→ 未来 │
└────────────────┴────────────────┴────────────────┘
```

### 3.3 各集合用途

| 集合 | 用途 | 操作 |
|------|------|------|
| 训练集 | 模型参数学习 | 前向+反向传播 |
| 验证集 | 超参数调优 | 前向传播（不更新参数）|
| 测试集 | 最终性能评估 | 前向传播（仅评估一次）|

---

## 4. 特征标准化

### 4.1 Z-score标准化

```python
x_scaled = (x - mean) / std

# 重要：只在训练集上计算 mean 和 std
scaler.fit(X_train)           # 拟合
X_train = scaler.transform(X_train)  # 转换训练集
X_val = scaler.transform(X_val)      # 转换验证集（用训练集参数）
X_test = scaler.transform(X_test)    # 转换测试集（用训练集参数）
```

### 4.2 为什么只在训练集上拟合？

```
❌ 错误做法：在全量数据上计算 mean/std
→ 测试集的统计信息"泄露"到标准化过程中

✅ 正确做法：只用训练集的 mean/std
→ 模拟真实场景：部署时只有历史数据
```

### 4.3 标准化的作用

| 问题 | 未标准化 | 标准化后 |
|------|---------|---------|
| 特征量纲不一致 | ofi_l1 ∈ [-162200, 189300] | ofi_l1 ∈ [-5, 5] |
| 梯度爆炸/消失 | 可能发生 | 缓解 |
| 收敛速度 | 慢 | 快 |

---

## 5. 输出数据结构

### 5.1 NumPy格式

```
dataset_T100_k20/
├── X_train.npy      # (777, 100, 43) - 训练特征
├── y_train.npy      # (777,) - 训练标签
├── X_val.npy        # (167, 100, 43) - 验证特征
├── y_val.npy        # (167,) - 验证标签
├── X_test.npy       # (167, 100, 43) - 测试特征
├── y_test.npy       # (167,) - 测试标签
├── scaler.pkl       # 标准化参数
└── config.pkl       # 配置参数
```

### 5.2 数组维度说明

```python
X.shape = (N, T, F)
# N = 样本数 (如777)
# T = 序列长度 (100)
# F = 特征数 (43)

y.shape = (N,)
# 标签: 0=下跌(-1), 1=平稳(0), 2=上涨(+1)
# 注意: PyTorch需要标签从0开始，所以做了+1映射
```

### 5.3 PyTorch格式

```python
# 加载Dataset
dataset = torch.load('dataset_train.pt')

# 获取一个样本
x, y = dataset[0]
print(x.shape)  # torch.Size([100, 43])
print(y)        # tensor(1)  # 1=平稳

# 创建DataLoader
loader = DataLoader(dataset, batch_size=64, shuffle=True)

for X_batch, y_batch in loader:
    # X_batch: (64, 100, 43)
    # y_batch: (64,)
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
```

---

## 6. 特征列表

本模块使用的**43个特征**（与 `11_feature_calculator.py` 输出对齐）：

| 类别 | 特征名 | 数量 | 说明 |
|------|--------|------|------|
| **价格** | `spread_bps`, `return_pct` | 2 | 价差、收益率 |
| **OFI聚合** | `ofi_l1`, `ofi_l5`, `ofi_l10`, `smart_ofi` | 4 | 单档/多档/撤单率修正OFI |
| **分档OFI** | `ofi_level_1` ~ `ofi_level_10` | 10 | 各档独立OFI（用于SHAP分析）|
| **OFI滚动** | `ofi_ma_10`, `ofi_std_10`, `ofi_zscore` | 3 | OFI滚动统计 |
| **Smart-OFI滚动** | `smart_ofi_ma_10`, `smart_ofi_std_10`, `smart_ofi_zscore` | 3 | Smart-OFI滚动统计 |
| **收益率滚动** | `return_ma_10`, `return_std_10` | 2 | 收益率滚动统计 |
| **深度** | `bid_depth_5`, `ask_depth_5`, `depth_imbalance_5` | 3 | 5档深度 |
| **深度** | `bid_depth_10`, `ask_depth_10`, `depth_imbalance_10` | 3 | 10档深度 |
| **深度不平衡滚动** | `depth_imb_ma_10`, `depth_imb_std_10`, `depth_imb_zscore` | 3 | 深度不平衡滚动统计 |
| **成交** | `buy_volume`, `sell_volume`, `trade_count`, `trade_imbalance` | 4 | 成交量、成交不平衡 |
| **成交不平衡滚动** | `trade_imb_ma_10`, `trade_imb_std_10`, `trade_imb_zscore` | 3 | 成交不平衡滚动统计 |
| **协方差** | `cov_stock_index`, `corr_stock_index` | 2 | 动态协方差、相关系数 |
| **市场状态** | `market_regime` | 1 | 市场状态 (0=平稳, 1=波动, 2=极端) |
| **合计** | | **43** | |

---

## 7. 使用方法

### 7.1 命令行参数

```bash
python 12_dataset_builder.py [OPTIONS]

Options:
  --input PATH [PATH ...]  指定输入特征文件（可多个）
  --code TEXT              股票代码（如 HK.00700）
  --days INTEGER           使用最近N天数据（默认30）
  --seq-len INTEGER        序列长度（默认100）
  --horizon INTEGER        预测步长（默认20）
  --batch-size INTEGER     批大小（默认64）
  --output PATH            输出目录
  --no-normalize           不进行标准化
```

### 7.2 使用示例

```bash
# 处理单个文件
python scripts/12_dataset_builder.py \
    --input data/processed/HK_00700/features_20260114.parquet

# 处理多天数据
python scripts/12_dataset_builder.py \
    --code HK.00700 --days 30

# 不同预测步长
python scripts/12_dataset_builder.py \
    --code HK.00700 --horizon 50

# 自定义输出目录
python scripts/12_dataset_builder.py \
    --code HK.00700 --output data/model_input
```

### 7.3 Python API使用

```python
from scripts.dataset_builder import DatasetBuilder, create_dataloaders

# 构建数据集
builder = DatasetBuilder(seq_len=100, horizon=20)
result = builder.build(
    file_paths=[Path('features_20260114.parquet')],
    normalize=True
)

# 获取划分后的数据
X_train, y_train = result['splits']['train']
X_val, y_val = result['splits']['val']
X_test, y_test = result['splits']['test']

# 创建DataLoader
loaders = create_dataloaders(result['splits'], batch_size=64)
train_loader = loaders['train']

# 训练循环
for epoch in range(100):
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
```

---

## 8. 实测结果

以 HK.00700 (腾讯控股) 2026-01-14 数据为例：

### 8.1 数据量统计

| 指标 | 数值 |
|------|------|
| 原始时间点 | 1,230 |
| 生成样本数 | 1,111 |
| 序列长度 | 100 |
| 特征维度 | 43 |

### 8.2 数据集划分

| 集合 | 样本数 | 比例 |
|------|--------|------|
| 训练集 | 777 | 70% |
| 验证集 | 167 | 15% |
| 测试集 | 167 | 15% |

### 8.3 标签分布

| 集合 | 下跌(-1) | 平稳(0) | 上涨(+1) |
|------|---------|---------|---------|
| 训练集 | 36.4% | 26.1% | 37.5% |
| 验证集 | 28.1% | 61.7% | 10.2% |
| 测试集 | 29.9% | 34.1% | 35.9% |

### 8.4 DataLoader配置

| 配置 | 数值 |
|------|------|
| batch_size | 64 |
| 训练batch数 | 13 |
| 验证batch数 | 3 |
| 测试batch数 | 3 |

---

## 9. 与论文对齐

| 论文设定 | 本实现 | 对齐状态 |
|---------|--------|---------|
| 输入序列长度 W=100 | seq_len=100 | ✅ |
| 预测步长 k∈{20,50,100} | horizon=20/50/100 | ✅ |
| 时序划分（防泄露） | 按时间顺序70/15/15 | ✅ |
| Z-score标准化 | 仅在训练集fit | ✅ |
| 三分类标签 | 下跌/平稳/上涨 | ✅ |

---

## 10. 后续步骤

本模块输出的数据集将用于：

1. **`13_model_trainer.py`** - 模型训练（Transformer/DeepLOB/LSTM）
2. **`14_model_evaluator.py`** - 模型评估（Accuracy/F1/AUC）
3. **`15_backtest.py`** - 策略回测与经济价值评估

---

## 11. 注意事项

1. **数据量要求**：需要至少 `seq_len + horizon` 个时间点才能生成样本
2. **内存管理**：大规模数据建议分批加载，避免OOM
3. **标签映射**：PyTorch要求标签从0开始，已做 `+1` 映射
4. **Scaler保存**：推理时需加载训练集的scaler参数
5. **GPU加速**：DataLoader设置 `pin_memory=True` 可加速CPU→GPU数据传输
