# K线策略回测模块说明

## 概述

`14b_kline_backtest.py` 实现了论文第四章第三节的策略回测与经济价值评估，包括交易模拟、指标计算和压力测试。

## 回测配置（论文表4-6）

```python
BACKTEST_CONFIG = {
    'initial_capital': 1_000_000,     # 初始资金（港币）
    'commission_rate': 0.0003,        # 基准佣金率 0.03%
    'slippage_bps': 1,                # 滑点（基点）
    'min_trade_interval': 5,          # 最小交易间隔（K线根数）
    'position_size': 0.3,             # 单次交易仓位比例
    'stop_loss_pct': 0.02,            # 止损比例 2%
    'take_profit_pct': 0.05,          # 止盈比例 5%
    'min_confidence': 0.5,            # 最小置信度
}
```

## 交易规则

| 规则 | 说明 |
|------|------|
| 开多仓 | 预测信号=2（上涨）且无持仓 |
| 平多仓 | 预测信号=0（下跌）且有持仓 |
| 仓位控制 | 单次交易使用30%资金 |
| 交易间隔 | 最少间隔5根K线 |
| 置信度过滤 | 只执行置信度>50%的信号 |

## 回测指标

| 指标 | 公式 | 说明 |
|------|------|------|
| 总收益率 | `(终值 - 初值) / 初值` | 策略总收益 |
| 年化收益 | `(1 + 总收益)^(年化因子) - 1` | 年化后的收益率 |
| 夏普比率 | `√(年化因子) × 均值 / 标准差` | 风险调整收益 |
| 最大回撤 | `max((peak - trough) / peak)` | 最大亏损幅度 |
| 胜率 | `盈利交易数 / 总交易数` | 交易成功率 |
| 盈亏比 | `总盈利 / 总亏损` | 盈利能力 |

## 使用方法

```bash
# 基准回测
python 14b_kline_backtest.py --model models/pv_transformer/model.pt

# 自定义交易成本
python 14b_kline_backtest.py --model models/pv_transformer/model.pt --cost 0.001

# 交易成本压力测试
python 14b_kline_backtest.py --model models/pv_transformer/model.pt --stress-test

# 指定输出目录
python 14b_kline_backtest.py --model models/pv_transformer/model.pt --output my_backtest
```

## 压力测试（论文表4-7）

测试不同交易成本下策略的稳健性：

```python
STRESS_TEST_COSTS = [0.0003, 0.0005, 0.001]  # 0.03%, 0.05%, 0.1%
```

输出示例：

```
==================================================
  交易成本压力测试结果
==================================================
 交易成本   总收益率   年化收益   夏普比率   最大回撤    胜率
   0.03%    25.30%    18.50%     1.85     12.30%   58.5%
   0.05%    22.10%    16.20%     1.62     12.80%   57.2%
   0.10%    15.80%    11.50%     1.15     13.50%   55.0%
```

## 输出文件

```
backtest_results/
├── backtest_result.png      # 回测图表（价格、权益、回撤）
├── stress_test.csv          # 压力测试结果
└── trades.csv               # 交易记录
```

## 回测图表

生成三联图：
1. **价格走势与交易信号**：显示买入/卖出点位
2. **权益曲线**：策略净值变化
3. **回撤曲线**：动态回撤幅度

## 与其他脚本的关系

```
13b_kline_model_trainer.py  →  模型训练
        ↓
14b_kline_backtest.py       →  策略回测（本脚本）
        ↓
输出回测报告和图表
```
