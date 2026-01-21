"""
K线策略回测模块
基于K线预测模型进行历史回测评估

功能（与论文第四章第三节对齐）:
1. 模拟交易执行（含交易成本、滑点）
2. 计算回测指标（夏普比率、最大回撤、胜率等）
3. 交易成本压力测试（0.03%, 0.1%）
4. 生成回测报告和可视化

使用方法:
    python 14b_kline_backtest.py --model models/pv_transformer/model.pt
    python 14b_kline_backtest.py --model models/pv_transformer/model.pt --cost 0.001
"""

import os
import sys
import io
import json
import argparse
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

# 解决Windows编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd

# PyTorch
import torch
import torch.nn.functional as F

# 可视化
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# 导入模型（从13b）
from importlib import import_module

# ============================================================
# 配置（论文表4-6回测假设）
# ============================================================

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

# 压力测试成本配置（论文表4-7）
STRESS_TEST_COSTS = [0.0003, 0.0005, 0.001]  # 0.03%, 0.05%, 0.1%

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# 数据类
# ============================================================

@dataclass
class Trade:
    """交易记录"""
    timestamp: int          # 时间索引
    side: str               # 'BUY' or 'SELL'
    price: float
    quantity: int
    value: float
    commission: float
    signal: int             # 预测信号
    confidence: float       # 预测置信度


@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    equity_curve: np.ndarray
    trades: List[Trade]


# ============================================================
# 回测引擎
# ============================================================

class KlineBacktester:
    """K线策略回测器"""
    
    def __init__(
        self,
        initial_capital: float = BACKTEST_CONFIG['initial_capital'],
        commission_rate: float = BACKTEST_CONFIG['commission_rate'],
        slippage_bps: float = BACKTEST_CONFIG['slippage_bps']
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        
        # 状态
        self.capital = initial_capital
        self.position = 0
        self.avg_cost = 0
        self.equity_curve = []
        self.trades = []
        self.last_trade_idx = -100
    
    def reset(self):
        """重置回测状态"""
        self.capital = self.initial_capital
        self.position = 0
        self.avg_cost = 0
        self.equity_curve = []
        self.trades = []
        self.last_trade_idx = -100
    
    def get_slippage_price(self, price: float, side: str) -> float:
        """计算滑点后的成交价"""
        slippage = price * self.slippage_bps / 10000
        if side == 'BUY':
            return price + slippage
        else:
            return price - slippage
    
    def execute_trade(
        self, 
        idx: int, 
        price: float, 
        signal: int, 
        confidence: float
    ) -> Optional[Trade]:
        """执行交易"""
        
        # 检查交易间隔
        if idx - self.last_trade_idx < BACKTEST_CONFIG['min_trade_interval']:
            return None
        
        # 检查置信度
        if confidence < BACKTEST_CONFIG['min_confidence']:
            return None
        
        trade = None
        
        if signal == 2 and self.position <= 0:  # 上涨信号，买入
            exec_price = self.get_slippage_price(price, 'BUY')
            trade_value = self.capital * BACKTEST_CONFIG['position_size']
            quantity = int(trade_value / exec_price / 100) * 100  # 港股100股一手
            
            if quantity > 0:
                commission = trade_value * self.commission_rate
                self.capital -= (trade_value + commission)
                self.position += quantity
                self.avg_cost = exec_price
                self.last_trade_idx = idx
                
                trade = Trade(
                    timestamp=idx,
                    side='BUY',
                    price=exec_price,
                    quantity=quantity,
                    value=trade_value,
                    commission=commission,
                    signal=signal,
                    confidence=confidence
                )
        
        elif signal == 0 and self.position > 0:  # 下跌信号，卖出
            exec_price = self.get_slippage_price(price, 'SELL')
            trade_value = self.position * exec_price
            commission = trade_value * self.commission_rate
            
            self.capital += (trade_value - commission)
            
            trade = Trade(
                timestamp=idx,
                side='SELL',
                price=exec_price,
                quantity=self.position,
                value=trade_value,
                commission=commission,
                signal=signal,
                confidence=confidence
            )
            
            self.position = 0
            self.avg_cost = 0
            self.last_trade_idx = idx
        
        if trade:
            self.trades.append(trade)
        
        return trade
    
    def run(
        self,
        prices: np.ndarray,
        predictions: np.ndarray,
        confidences: np.ndarray
    ) -> BacktestResult:
        """
        运行回测
        
        Args:
            prices: 价格序列 (N,)
            predictions: 预测标签 (N,) - 0:下跌, 1:平稳, 2:上涨
            confidences: 预测置信度 (N,)
            
        Returns:
            BacktestResult
        """
        self.reset()
        n = len(prices)
        
        for i in range(n):
            price = prices[i]
            signal = predictions[i]
            conf = confidences[i]
            
            # 执行交易
            self.execute_trade(i, price, signal, conf)
            
            # 计算当前权益
            position_value = self.position * price if self.position > 0 else 0
            total_equity = self.capital + position_value
            self.equity_curve.append(total_equity)
        
        # 强制平仓
        if self.position > 0:
            final_price = prices[-1]
            final_value = self.position * final_price
            commission = final_value * self.commission_rate
            self.capital += (final_value - commission)
            self.position = 0
        
        return self._compute_metrics()
    
    def _compute_metrics(self) -> BacktestResult:
        """计算回测指标"""
        equity = np.array(self.equity_curve)
        
        # 总收益率
        total_return = (equity[-1] / self.initial_capital - 1) * 100
        
        # 年化收益率（假设每年250个交易日，每天6小时，1分钟K线）
        n_periods = len(equity)
        periods_per_year = 250 * 6 * 60  # 1分钟K线
        annualized_return = ((1 + total_return/100) ** (periods_per_year / n_periods) - 1) * 100
        
        # 夏普比率
        returns = np.diff(equity) / equity[:-1]
        sharpe = np.sqrt(periods_per_year) * np.mean(returns) / (np.std(returns) + 1e-8)
        
        # 最大回撤
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown) * 100
        
        # 胜率和盈亏比
        trade_returns = []
        for i in range(0, len(self.trades) - 1, 2):  # 配对买卖
            if i + 1 < len(self.trades):
                buy = self.trades[i]
                sell = self.trades[i + 1]
                if buy.side == 'BUY' and sell.side == 'SELL':
                    pnl = (sell.price - buy.price) / buy.price
                    trade_returns.append(pnl)
        
        trade_returns = np.array(trade_returns)
        win_rate = np.mean(trade_returns > 0) * 100 if len(trade_returns) > 0 else 0
        
        gains = trade_returns[trade_returns > 0].sum() if len(trade_returns) > 0 else 0
        losses = abs(trade_returns[trade_returns < 0].sum()) if len(trade_returns) > 0 else 1
        profit_factor = gains / (losses + 1e-8)
        
        avg_trade_return = np.mean(trade_returns) * 100 if len(trade_returns) > 0 else 0
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            avg_trade_return=avg_trade_return,
            equity_curve=equity,
            trades=self.trades
        )


# ============================================================
# 模型预测
# ============================================================

def load_model_and_predict(
    model_path: Path,
    test_data: torch.Tensor,
    device: torch.device = DEVICE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载模型并预测
    
    Returns:
        predictions: 预测标签
        confidences: 预测置信度
    """
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    model_name = checkpoint.get('model_name', 'unknown')
    
    print(f"加载模型: {model_name}")
    
    # 根据模型类型重建模型
    # TODO: 根据实际保存的模型信息重建
    
    # 暂时使用简单的随机预测作为占位
    n_samples = len(test_data)
    predictions = np.random.randint(0, 3, n_samples)
    confidences = np.random.uniform(0.3, 1.0, n_samples)
    
    return predictions, confidences


def predict_with_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device = DEVICE
) -> Tuple[np.ndarray, np.ndarray]:
    """使用模型进行预测"""
    model.eval()
    all_preds = []
    all_confs = []
    
    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            logits = model(X)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            confs = probs.max(dim=1).values.cpu().numpy()
            
            all_preds.extend(preds)
            all_confs.extend(confs)
    
    return np.array(all_preds), np.array(all_confs)


# ============================================================
# 可视化
# ============================================================

def plot_backtest_result(
    result: BacktestResult,
    prices: np.ndarray,
    save_path: Path = None
):
    """绘制回测结果"""
    if not HAS_PLOT:
        print("[WARN] matplotlib未安装，跳过可视化")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. 价格和交易信号
    ax1 = axes[0]
    ax1.plot(prices, label='价格', alpha=0.7)
    
    buys = [t for t in result.trades if t.side == 'BUY']
    sells = [t for t in result.trades if t.side == 'SELL']
    
    if buys:
        ax1.scatter([t.timestamp for t in buys], [t.price for t in buys], 
                   marker='^', color='green', s=100, label='买入')
    if sells:
        ax1.scatter([t.timestamp for t in sells], [t.price for t in sells],
                   marker='v', color='red', s=100, label='卖出')
    
    ax1.set_title('价格走势与交易信号')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 权益曲线
    ax2 = axes[1]
    ax2.plot(result.equity_curve, label='策略权益', color='blue')
    ax2.axhline(y=result.equity_curve[0], color='gray', linestyle='--', label='初始资金')
    ax2.fill_between(range(len(result.equity_curve)), 
                     result.equity_curve[0], result.equity_curve, 
                     alpha=0.3, color='blue')
    ax2.set_title(f'权益曲线 (总收益: {result.total_return:.2f}%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 回撤
    ax3 = axes[2]
    equity = np.array(result.equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100
    ax3.fill_between(range(len(drawdown)), 0, drawdown, alpha=0.5, color='red')
    ax3.set_title(f'回撤 (最大回撤: {result.max_drawdown:.2f}%)')
    ax3.set_ylabel('回撤 (%)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    plt.show()


def print_backtest_report(result: BacktestResult, config_name: str = "基准"):
    """打印回测报告"""
    print(f"\n{'='*50}")
    print(f"  回测报告 ({config_name})")
    print('='*50)
    print(f"  总收益率:       {result.total_return:>10.2f}%")
    print(f"  年化收益率:     {result.annualized_return:>10.2f}%")
    print(f"  夏普比率:       {result.sharpe_ratio:>10.2f}")
    print(f"  最大回撤:       {result.max_drawdown:>10.2f}%")
    print(f"  胜率:           {result.win_rate:>10.2f}%")
    print(f"  盈亏比:         {result.profit_factor:>10.2f}")
    print(f"  总交易次数:     {result.total_trades:>10}")
    print(f"  平均交易收益:   {result.avg_trade_return:>10.2f}%")
    print('='*50)


# ============================================================
# 压力测试
# ============================================================

def run_stress_test(
    prices: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    cost_rates: List[float] = STRESS_TEST_COSTS
) -> pd.DataFrame:
    """
    交易成本压力测试（论文表4-7）
    """
    results = []
    
    for cost in cost_rates:
        backtester = KlineBacktester(commission_rate=cost)
        result = backtester.run(prices, predictions, confidences)
        
        results.append({
            '交易成本': f'{cost*100:.2f}%',
            '总收益率': f'{result.total_return:.2f}%',
            '年化收益': f'{result.annualized_return:.2f}%',
            '夏普比率': f'{result.sharpe_ratio:.2f}',
            '最大回撤': f'{result.max_drawdown:.2f}%',
            '胜率': f'{result.win_rate:.1f}%',
        })
    
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("  交易成本压力测试结果")
    print("="*60)
    print(df.to_string(index=False))
    
    return df


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='K线策略回测')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--dataset', type=str, help='数据集路径')
    parser.add_argument('--cost', type=float, default=BACKTEST_CONFIG['commission_rate'],
                        help='交易成本率')
    parser.add_argument('--stress-test', action='store_true', help='运行压力测试')
    parser.add_argument('--output', type=str, default='backtest_results', help='输出目录')
    
    args = parser.parse_args()
    
    print("="*60)
    print("  K线策略回测")
    print("="*60)
    print(f"  模型: {args.model}")
    print(f"  交易成本: {args.cost*100:.2f}%")
    
    # 加载测试数据
    if args.dataset:
        with open(args.dataset, 'rb') as f:
            dataset = pickle.load(f)
        test_data = dataset.get('test')
    else:
        print("[WARN] 使用模拟数据进行演示")
        n_samples = 5000
        prices = 100 * np.cumprod(1 + np.random.randn(n_samples) * 0.001)
        predictions = np.random.randint(0, 3, n_samples)
        confidences = np.random.uniform(0.4, 0.9, n_samples)
    
    # 回测
    backtester = KlineBacktester(commission_rate=args.cost)
    result = backtester.run(prices, predictions, confidences)
    
    # 报告
    print_backtest_report(result)
    
    # 压力测试
    if args.stress_test:
        stress_df = run_stress_test(prices, predictions, confidences)
        
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        stress_df.to_csv(output_dir / 'stress_test.csv', index=False)
    
    # 可视化
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    plot_backtest_result(result, prices, save_path=output_dir / 'backtest_result.png')
    
    print("\n[DONE] 回测完成！")


if __name__ == "__main__":
    main()
