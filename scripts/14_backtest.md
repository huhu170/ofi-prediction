# 策略回测方案说明

> 对应脚本：`14_backtest.py`  
> 版本：v1.0  
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

### 4.1 命令行参数

```bash
python 14_backtest.py [OPTIONS]

Options:
  --model MODEL [...]  要回测的模型 (lstm, gru, transformer, all)
  --data PATH          数据目录
  --output PATH        输出目录
  --seq-len INTEGER    序列长度
  --compare            对比所有模型
```

### 4.2 使用示例

```bash
# 回测单个模型
python scripts/14_backtest.py --model transformer

# 回测多个模型并对比
python scripts/14_backtest.py --model lstm gru transformer --compare

# 回测所有模型
python scripts/14_backtest.py --model all --compare
```

---

## 5. 输出文件

```
backtest_results/
├── backtest_transformer.png   # 权益曲线图
├── backtest_lstm.png
├── metrics_transformer.json   # 回测指标
├── metrics_lstm.json
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
2. **模型要求**: 需要先运行 `13_model_trainer.py` 训练模型
3. **历史限制**: 回测结果不代表未来表现
4. **成本假设**: 实际交易成本可能更高
