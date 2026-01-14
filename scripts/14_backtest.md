# 策略回测方案说明

> 对应脚本：`14_backtest.py`  
> 版本：v1.1  
> 更新日期：2026-01-14

---

## 1. 概述

本模块基于历史数据评估OFI预测模型的交易策略表现，计算夏普比率、最大回撤等关键指标。

### 1.1 回测 vs 实时交易

| 模块 | 用途 | 数据源 | 下单 |
|------|------|--------|------|
| `14_backtest.py` | **历史回测** | Parquet文件 | ❌ 纯计算 |
| `15_paper_trader.py` | **实时交易** | 富途API | ✅ 模拟下单 |

### 1.2 回测流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      回测流程                                    │
├─────────────────────────────────────────────────────────────────┤
│  1. 加载历史数据 (features_*.parquet)                           │
│                                                                 │
│  2. 加载训练好的模型                                            │
│                                                                 │
│  3. 生成交易信号                                                │
│     └─ 模型预测: 下跌/平稳/上涨 + 置信度                        │
│                                                                 │
│  4. 模拟交易执行                                                │
│     └─ 买入/卖出（含佣金、滑点）                                │
│                                                                 │
│  5. 计算回测指标                                                │
│     └─ 夏普比率、最大回撤、胜率等                               │
│                                                                 │
│  6. 生成回测报告                                                │
│     └─ 权益曲线、回撤曲线、交易记录                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 配置说明

### 2.1 回测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| initial_capital | 1,000,000 | 初始资金（港币）|
| commission_rate | 0.03% | 佣金率 |
| slippage_bps | 1 | 滑点（基点）|
| position_size | 30% | 单次交易仓位 |
| stop_loss_pct | 2% | 止损比例 |
| take_profit_pct | 5% | 止盈比例 |
| min_confidence | 50% | 最小置信度 |
| min_trade_interval | 60秒 | 最小交易间隔 |

### 2.2 交易规则

```python
# 买入条件
if signal == 'BUY' and confidence >= 50% and 可交易:
    买入 = 可用资金 × 30%

# 卖出条件
if signal == 'SELL' and 有持仓 and 可交易:
    卖出全部持仓

# 止损止盈
if 持仓亏损 >= 2%: 触发止损
if 持仓盈利 >= 5%: 触发止盈
```

---

## 3. 回测指标

### 3.1 收益指标

| 指标 | 公式 | 说明 |
|------|------|------|
| 总收益率 | (期末权益-初始资金)/初始资金 | 整体收益 |
| 年化收益 | (1+总收益)^(年化因子)-1 | 标准化比较 |
| 超额收益 | 策略收益-基准收益 | 相对买入持有 |

### 3.2 风险指标

| 指标 | 公式 | 说明 |
|------|------|------|
| 波动率 | std(收益率)×√年化因子 | 收益稳定性 |
| 最大回撤 | max((峰值-谷值)/峰值) | 最大亏损 |
| 夏普比率 | 年化收益/年化波动率 | 风险调整收益 |
| Sortino比率 | 年化收益/下行波动率 | 只考虑亏损 |
| Calmar比率 | 年化收益/最大回撤 | 回撤敏感 |

### 3.3 交易统计

| 指标 | 说明 |
|------|------|
| 总交易次数 | 买入+卖出次数 |
| 胜率 | 盈利次数/总次数 |
| 盈亏比 | 平均盈利/平均亏损 |
| 利润因子 | 总盈利/总亏损 |

---

## 4. 使用方法

### 4.1 支持的模型

| 类型 | 模型名称 | 说明 |
|------|----------|------|
| **ML模型** | arima | ARIMA时间序列 |
| **ML模型** | logistic | 逻辑回归 |
| **ML模型** | rf | 随机森林 |
| **ML模型** | xgboost | XGBoost |
| **DL模型** | lstm | LSTM |
| **DL模型** | gru | GRU |
| **DL模型** | deeplob | DeepLOB |
| **DL模型** | transformer | Transformer |
| **DL模型** | smart_trans | Smart-Transformer（本文） |

### 4.2 命令行参数

```bash
python 14_backtest.py [OPTIONS]

Options:
  --model MODEL [...]  要回测的模型 (arima, logistic, xgboost, rf, lstm, gru, deeplob, transformer, smart_trans, all/ml/deep)
  --data PATH          数据目录
  --output PATH        输出目录
  --seq-len INTEGER    序列长度
  --compare            对比所有模型
```

### 4.3 使用示例

```bash
# 回测单个模型
python scripts/14_backtest.py --model transformer

# 回测多个模型并对比
python scripts/14_backtest.py --model lstm gru transformer --compare

# 回测所有ML模型
python scripts/14_backtest.py --model ml --compare

# 回测所有深度学习模型
python scripts/14_backtest.py --model deep --compare

# 回测全部9个模型
python scripts/14_backtest.py --model all --compare
```

---

## 5. 输出文件

```
backtest_results/
├── # ML模型回测结果
├── backtest_arima.png
├── backtest_logistic.png
├── backtest_rf.png
├── backtest_xgboost.png
├── metrics_arima.json
├── metrics_logistic.json
├── metrics_rf.json
├── metrics_xgboost.json
│
├── # DL模型回测结果
├── backtest_lstm.png
├── backtest_gru.png
├── backtest_deeplob.png
├── backtest_transformer.png
├── backtest_smart_trans.png
├── metrics_lstm.json
├── metrics_gru.json
├── metrics_deeplob.json
├── metrics_transformer.json
├── metrics_smart_trans.json
│
└── comparison.csv             # 模型对比表
```

### 5.1 metrics.json 格式

```json
{
  "model_name": "TRANSFORMER",
  "total_return": 0.0523,
  "annual_return": 0.1245,
  "sharpe_ratio": 1.82,
  "max_drawdown": 0.0312,
  "win_rate": 0.58,
  "total_trades": 45,
  "profit_factor": 1.65
}
```

---

## 6. 注意事项

1. **数据要求**: 需要先运行 `11_feature_calculator.py` 和 `12_dataset_builder.py`
2. **DL模型要求**: 需要先运行 `13_model_trainer.py --model all` 训练深度学习模型
3. **ML模型要求**: 需要先运行 `13_model_trainer.py --model ml` 训练机器学习模型
4. **历史限制**: 回测结果不代表未来表现
5. **成本假设**: 实际交易成本可能更高

### 6.1 训练模型示例

```bash
# 训练所有深度学习模型
python scripts/13_model_trainer.py --model all --epochs 50

# 训练所有机器学习模型（包括ARIMA）
python scripts/13_model_trainer.py --model ml

# 完整训练（DL + ML）
python scripts/13_model_trainer.py --model all ml --epochs 50
```
