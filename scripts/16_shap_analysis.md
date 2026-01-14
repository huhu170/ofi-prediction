# SHAP特征归因分析说明

> 对应脚本：`16_shap_analysis.py`  
> 版本：v1.0  
> 更新日期：2026-01-14

---

## 1. 概述

本模块使用SHAP（SHapley Additive exPlanations）方法分析OFI预测模型的特征重要性和决策机制，提升模型可解释性。

### 1.1 为什么需要SHAP分析？

| 问题 | SHAP能回答 |
|------|-----------|
| 哪个特征最重要？ | 全局特征重要性排序 |
| 为什么模型预测上涨？ | 单样本归因分解 |
| Smart-OFI有效吗？ | 比较OFI系列特征的贡献 |
| 特征间有交互吗？ | 特征交互效应分析 |

### 1.2 SHAP原理

```
模型预测 = 基线值 + Σ(各特征的SHAP值)

例如：
预测上涨概率 = 0.5 + 0.15(Smart-OFI) + 0.10(OFI-L1) - 0.05(价差) + ...
                     ↑              ↑             ↑
                  正贡献         正贡献        负贡献
```

---

## 2. 分析方法

### 2.1 两种方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **积分梯度** | 快速、稳定 | 近似值 | 大规模分析 |
| **SHAP库** | 精确、功能全 | 慢 | 小样本深度分析 |

默认使用**积分梯度**方法（更快）。

### 2.2 积分梯度公式

```
IG_i = (x_i - x'_i) × ∫[0,1] (∂F/∂x_i)(x' + α(x-x')) dα

其中：
- x: 输入样本
- x': 基线（通常为0）
- F: 模型输出
- α: 积分路径参数
```

---

## 3. 使用方法

### 3.1 命令行参数

```bash
python 16_shap_analysis.py [OPTIONS]

Options:
  --model TEXT         模型名称 (默认 transformer)
  --data PATH          数据目录
  --samples INTEGER    分析样本数 (默认 100)
  --output PATH        输出目录
  --use-shap           使用SHAP库（较慢但更精确）
```

### 3.2 使用示例

```bash
# 快速分析（积分梯度）
python scripts/16_shap_analysis.py --model transformer --samples 100

# 精确分析（SHAP库）
python scripts/16_shap_analysis.py --model transformer --use-shap --samples 50

# 分析Smart-Trans模型
python scripts/16_shap_analysis.py --model smart_trans
```

---

## 4. 输出结果

### 4.1 特征重要性表

```
   feature      feature_cn  importance  importance_pct
0  smart_ofi    Smart-OFI      0.0892         18.45%
1  ofi_l1       OFI-L1         0.0723         14.96%
2  ofi_zscore   OFI Z-score    0.0654         13.53%
3  return_pct   收益率(%)       0.0512         10.59%
...
```

### 4.2 输出文件

```
shap_results/
├── feature_importance_transformer.csv   # 特征重要性表
├── shap_importance_transformer.png      # 重要性条形图
└── shap_class_comparison_transformer.png # 分类对比图
```

---

## 5. 可视化图表

### 5.1 特征重要性排序图

```
Smart-OFI        ████████████████████ 18.5%
OFI-L1           ████████████████ 15.0%
OFI Z-score      ██████████████ 13.5%
收益率(%)         ███████████ 10.6%
深度不平衡        █████████ 8.2%
...
```

### 5.2 分类对比图

对比预测"上涨"、"平稳"、"下跌"时各特征的重要性差异。

---

## 6. 与论文对齐

### 6.1 论文4.4节要求

| 图表 | 内容 | 对应输出 |
|------|------|---------|
| 图4-6 | SHAP特征重要性排序 | `shap_importance_*.png` |
| 图4-7 | 极端样本归因分析 | 需单独分析 |
| 表4-14 | 分组检验结果 | 结合回测数据 |

### 6.2 预期发现

1. **Smart-OFI排名靠前**: 验证特征工程有效性
2. **OFI系列特征集中**: 证明订单流信息的预测价值
3. **近期时间步更重要**: 符合金融直觉

---

## 7. 注意事项

### 7.1 计算资源

- 积分梯度: ~1分钟/100样本
- SHAP库: ~10分钟/100样本

### 7.2 依赖项

```bash
# 可选（SHAP库方法）
pip install shap

# 必需（可视化）
pip install matplotlib
```

### 7.3 样本选择

建议分析：
- 正确预测的样本（验证模型逻辑）
- 错误预测的样本（发现模型弱点）
- 极端行情样本（验证鲁棒性）
