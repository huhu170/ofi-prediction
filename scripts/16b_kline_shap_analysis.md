# K线SHAP可解释性分析模块说明

## 概述

`16b_kline_shap_analysis.py` 实现了论文第四章第四节的模型可解释性分析，包括SHAP特征归因、PV-CrossAttention注意力可视化和LSF尺度权重分析。

## 与论文章节的对应关系

| 论文内容 | 脚本功能 |
|----------|----------|
| 4.4节 线路一：可解释性 | SHAP特征重要性排序 |
| 表4-X：特征贡献度 | `get_feature_importance()` |
| 图4-X：注意力热力图 | `AttentionVisualizer.plot_attention_heatmap()` |
| 表4-X：尺度权重分布 | `LSFAnalyzer.analyze_weights_by_regime()` |

## 核心功能

### 1. SHAP特征归因

```python
class KlineSHAPAnalyzer:
    """计算并分析SHAP值"""
    
    def compute_shap_values(data, n_samples=100):
        """使用KernelExplainer计算SHAP值"""
        
    def get_feature_importance() -> pd.DataFrame:
        """获取特征重要性排序（|SHAP|均值）"""
        
    def get_group_importance() -> pd.DataFrame:
        """获取特征组重要性（按类别聚合）"""
```

### 2. PV-CrossAttention注意力可视化

```python
class AttentionVisualizer:
    """可视化量价交叉注意力"""
    
    def get_attention_weights(price_features, volume_features):
        """提取注意力权重矩阵"""
        
    def plot_attention_heatmap(attn_weights, sample_idx, head_idx):
        """绘制注意力热力图"""
```

### 3. LSF尺度权重分析

```python
class LSFAnalyzer:
    """分析多尺度融合权重"""
    
    def get_scale_weights(scale_data) -> np.ndarray:
        """获取各尺度的融合权重"""
        
    def analyze_weights_by_regime(weights, regimes) -> pd.DataFrame:
        """按市场状态分析权重变化"""
        
    def plot_weights_distribution(weights):
        """绘制权重分布（箱线图+柱状图）"""
```

## 特征分组

| 特征组 | 包含特征 |
|--------|----------|
| **价格动量** | return_1, return_5, return_20, return_zscore |
| **波动率** | atr_pct, range_pct, volatility_20 |
| **成交不平衡** | ti, ti_5, ti_20, ti_zscore |
| **成交量** | relative_volume, volume_change, pv_corr |
| **技术指标** | rsi, bb_position, macd_dif, macd_dea, macd |
| **市场状态** | market_regime |

## 使用方法

```bash
# 基础SHAP分析
python 16b_kline_shap_analysis.py --model models/pv_transformer/model.pt

# 指定分析样本数
python 16b_kline_shap_analysis.py --model models/pv_transformer/model.pt --samples 200

# 指定数据集
python 16b_kline_shap_analysis.py --model models/pv_transformer/model.pt \
    --dataset data/datasets/dataset_HK_00700_1M.pkl

# 指定输出目录
python 16b_kline_shap_analysis.py --model models/pv_transformer/model.pt \
    --output shap_results/HK_00700
```

## 输出文件

运行后在输出目录生成：

```
shap_results/
├── feature_importance.csv      # 特征重要性排序
├── feature_importance.png      # 特征重要性柱状图
├── group_importance.csv        # 特征组重要性
├── group_importance.png        # 特征组饼图
├── attention_heatmap.png       # PV-CrossAttention热力图（如有）
└── lsf_weights.png             # LSF尺度权重分布（如有）
```

## 输出示例

### 特征重要性排序

```
 rank        feature_cn  importance
    1     成交不平衡(TI)       0.150
    2        TI Z-score       0.120
    3      1分钟收益率       0.100
    4       相对成交量       0.090
    5       量价相关性       0.080
    ...
```

### 特征组贡献度

```
        group  importance
  成交不平衡       0.370
    价格动量       0.210
      成交量       0.170
    技术指标       0.150
      波动率       0.080
    市场状态       0.020
```

## 与其他脚本的关系

```
13b_kline_model_trainer.py  →  训练模型
        ↓
14b_kline_backtest.py       →  策略回测
        ↓
16b_kline_shap_analysis.py  →  可解释性分析（本脚本）
```

## 依赖

- `shap`: SHAP值计算
- `matplotlib`: 可视化
- `seaborn`: 热力图

安装：
```bash
pip install shap matplotlib seaborn
```
