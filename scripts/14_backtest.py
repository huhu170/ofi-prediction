"""
策略回测模块
基于历史数据评估OFI预测模型的交易策略表现

功能:
1. 读取历史特征数据和模型预测
2. 模拟交易执行（含交易成本）
3. 计算回测指标（夏普比率、最大回撤、胜率等）
4. 生成回测报告和可视化

使用方法:
    python 14_backtest.py --model transformer --data data/processed/HK_00700
    python 14_backtest.py --model all --compare
"""

import os
import sys
import io
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

# 解决Windows编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 加载环境变量
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".apikey.env"
load_dotenv(env_path, override=True)

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

# ============================================================
# 配置
# ============================================================

# 回测配置
BACKTEST_CONFIG = {
    'initial_capital': 1_000_000,     # 初始资金（港币）
    'commission_rate': 0.0003,        # 佣金率 0.03%
    'slippage_bps': 1,                # 滑点（基点）
    'min_trade_interval': 6,          # 最小交易间隔（时间步数，60秒）
    'position_size': 0.3,             # 单次交易仓位比例
    'stop_loss_pct': 0.02,            # 止损比例
    'take_profit_pct': 0.05,          # 止盈比例
    'min_confidence': 0.5,            # 最小置信度
}

# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据路径
DATA_PROCESSED = Path(os.getenv("DATA_PROCESSED", "data/processed"))
MODEL_DIR = Path("models")


# ============================================================
# 数据类
# ============================================================

@dataclass
class Trade:
    """交易记录"""
    timestamp: datetime
    side: str           # 'BUY' or 'SELL'
    price: float
    quantity: int
    value: float
    commission: float
    signal_confidence: float


@dataclass
class Position:
    """持仓信息"""
    quantity: int = 0
    avg_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class BacktestResult:
    """回测结果"""
    # 基本信息
    model_name: str = ""
    start_date: str = ""
    end_date: str = ""
    trading_days: int = 0
    
    # 收益指标
    total_return: float = 0.0
    annual_return: float = 0.0
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    
    # 风险指标
    volatility: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # 交易统计
    total_trades: int = 0
    win_trades: int = 0
    loss_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # 时间序列
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    
    # 交易记录
    trades: List[Trade] = field(default_factory=list)


# ============================================================
# 信号生成器
# ============================================================

class SignalGenerator:
    """
    交易信号生成器（深度学习模型）
    
    基于模型预测生成交易信号
    """
    
    def __init__(self, model, scaler_params: dict = None, seq_len: int = 100):
        self.model = model
        self.scaler_params = scaler_params
        self.seq_len = seq_len
        self.model.eval()
    
    def generate_signals(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量生成信号
        
        Args:
            features: 特征数据 (N, seq_len, num_features)
            
        Returns:
            signals: 信号数组 (N,) - 0=SELL, 1=HOLD, 2=BUY
            confidences: 置信度数组 (N,)
        """
        # 标准化
        if self.scaler_params is not None:
            mean = self.scaler_params['mean']
            std = self.scaler_params['std']
            features = (features - mean) / std
        
        # 转为tensor
        X = torch.FloatTensor(features).to(DEVICE)
        
        # 批量预测
        signals = []
        confidences = []
        
        batch_size = 64
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            
            with torch.no_grad():
                logits = self.model(batch)
                probs = F.softmax(logits, dim=1).cpu().numpy()
            
            pred_classes = probs.argmax(axis=1)
            pred_confidences = probs.max(axis=1)
            
            signals.extend(pred_classes)
            confidences.extend(pred_confidences)
        
        return np.array(signals), np.array(confidences)


class MLSignalGenerator:
    """
    交易信号生成器（ML模型）
    
    基于ML模型预测生成交易信号
    """
    
    def __init__(self, ml_instance, model_name: str):
        self.ml_instance = ml_instance
        self.model_name = model_name
    
    def generate_signals(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量生成信号
        
        Args:
            features: 特征数据 (N, seq_len, num_features)
            
        Returns:
            signals: 信号数组 (N,) - 0=SELL, 1=HOLD, 2=BUY
            confidences: 置信度数组 (N,)
        """
        # 获取预测
        predictions = self.ml_instance.predict(self.model_name, features)
        probabilities = self.ml_instance.predict_proba(self.model_name, features)
        
        # 提取置信度
        confidences = probabilities.max(axis=1)
        
        return predictions, confidences


# ============================================================
# 回测引擎
# ============================================================

class BacktestEngine:
    """
    回测引擎
    
    模拟交易执行并计算回测指标
    """
    
    def __init__(self, config: dict = None):
        self.config = config or BACKTEST_CONFIG
        
        # 账户状态
        self.capital = self.config['initial_capital']
        self.cash = self.capital
        self.position = Position()
        
        # 交易记录
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.timestamps = []
        
        # 状态
        self.last_trade_idx = -999  # 上次交易的索引
    
    def reset(self):
        """重置回测状态"""
        self.capital = self.config['initial_capital']
        self.cash = self.capital
        self.position = Position()
        self.trades = []
        self.equity_curve = []
        self.timestamps = []
        self.last_trade_idx = -999
    
    def _calculate_commission(self, value: float) -> float:
        """计算交易佣金"""
        return value * self.config['commission_rate']
    
    def _calculate_slippage(self, price: float, side: str) -> float:
        """计算滑点"""
        slippage = price * self.config['slippage_bps'] / 10000
        if side == 'BUY':
            return price + slippage
        else:
            return price - slippage
    
    def _can_trade(self, current_idx: int) -> bool:
        """检查是否可以交易（冷却时间）"""
        return (current_idx - self.last_trade_idx) >= self.config['min_trade_interval']
    
    def _execute_buy(self, price: float, timestamp: datetime, confidence: float) -> Optional[Trade]:
        """执行买入"""
        # 计算可买金额
        available = self.cash * self.config['position_size']
        
        # 考虑滑点
        exec_price = self._calculate_slippage(price, 'BUY')
        
        # 计算数量（港股100股一手）
        lot_size = 100
        quantity = int(available / exec_price / lot_size) * lot_size
        
        if quantity <= 0:
            return None
        
        # 计算成本
        value = quantity * exec_price
        commission = self._calculate_commission(value)
        total_cost = value + commission
        
        if total_cost > self.cash:
            return None
        
        # 更新账户
        self.cash -= total_cost
        
        # 更新持仓
        if self.position.quantity > 0:
            # 加仓：计算平均成本
            total_qty = self.position.quantity + quantity
            total_cost_basis = self.position.avg_price * self.position.quantity + exec_price * quantity
            self.position.avg_price = total_cost_basis / total_qty
            self.position.quantity = total_qty
        else:
            self.position.quantity = quantity
            self.position.avg_price = exec_price
        
        # 记录交易
        trade = Trade(
            timestamp=timestamp,
            side='BUY',
            price=exec_price,
            quantity=quantity,
            value=value,
            commission=commission,
            signal_confidence=confidence
        )
        self.trades.append(trade)
        
        return trade
    
    def _execute_sell(self, price: float, timestamp: datetime, confidence: float) -> Optional[Trade]:
        """执行卖出"""
        if self.position.quantity <= 0:
            return None
        
        # 考虑滑点
        exec_price = self._calculate_slippage(price, 'SELL')
        
        # 卖出全部持仓
        quantity = self.position.quantity
        value = quantity * exec_price
        commission = self._calculate_commission(value)
        net_proceeds = value - commission
        
        # 计算盈亏
        cost_basis = self.position.avg_price * quantity
        pnl = net_proceeds - cost_basis
        
        # 更新账户
        self.cash += net_proceeds
        self.position.realized_pnl += pnl
        
        # 清空持仓
        self.position.quantity = 0
        self.position.avg_price = 0
        
        # 记录交易
        trade = Trade(
            timestamp=timestamp,
            side='SELL',
            price=exec_price,
            quantity=quantity,
            value=value,
            commission=commission,
            signal_confidence=confidence
        )
        self.trades.append(trade)
        
        return trade
    
    def _update_position_value(self, current_price: float):
        """更新持仓市值"""
        if self.position.quantity > 0:
            self.position.market_value = self.position.quantity * current_price
            self.position.unrealized_pnl = (
                self.position.market_value - 
                self.position.avg_price * self.position.quantity
            )
    
    def _get_equity(self) -> float:
        """获取当前权益"""
        return self.cash + self.position.market_value
    
    def _check_stop_loss_take_profit(self, current_price: float) -> Optional[str]:
        """检查止损止盈"""
        if self.position.quantity <= 0:
            return None
        
        pnl_pct = (current_price - self.position.avg_price) / self.position.avg_price
        
        if pnl_pct <= -self.config['stop_loss_pct']:
            return 'STOP_LOSS'
        elif pnl_pct >= self.config['take_profit_pct']:
            return 'TAKE_PROFIT'
        
        return None
    
    def run(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        confidences: np.ndarray,
        timestamps: List[datetime]
    ) -> BacktestResult:
        """
        运行回测
        
        Args:
            prices: 价格数组 (N,)
            signals: 信号数组 (N,) - 0=SELL, 1=HOLD, 2=BUY
            confidences: 置信度数组 (N,)
            timestamps: 时间戳列表
            
        Returns:
            回测结果
        """
        self.reset()
        
        n = len(prices)
        
        for i in range(n):
            price = prices[i]
            signal = signals[i]
            confidence = confidences[i]
            ts = timestamps[i]
            
            # 更新持仓市值
            self._update_position_value(price)
            
            # 检查止损止盈
            sl_tp = self._check_stop_loss_take_profit(price)
            if sl_tp:
                self._execute_sell(price, ts, confidence)
                self.last_trade_idx = i
            
            # 检查交易信号
            elif self._can_trade(i) and confidence >= self.config['min_confidence']:
                if signal == 2:  # BUY
                    if self._execute_buy(price, ts, confidence):
                        self.last_trade_idx = i
                elif signal == 0:  # SELL
                    if self._execute_sell(price, ts, confidence):
                        self.last_trade_idx = i
            
            # 记录权益
            self.equity_curve.append(self._get_equity())
            self.timestamps.append(ts)
        
        # 计算回测指标
        result = self._calculate_metrics(prices, timestamps)
        
        return result
    
    def _calculate_metrics(self, prices: np.ndarray, timestamps: List[datetime]) -> BacktestResult:
        """计算回测指标"""
        result = BacktestResult()
        
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        
        # 基本信息
        result.start_date = timestamps[0].strftime('%Y-%m-%d') if timestamps else ""
        result.end_date = timestamps[-1].strftime('%Y-%m-%d') if timestamps else ""
        result.trading_days = len(set(ts.date() for ts in timestamps))
        
        # 收益指标
        result.total_return = (equity[-1] - self.capital) / self.capital if len(equity) > 0 else 0
        
        # 年化收益（假设每年252个交易日，每天约360个10秒窗口）
        n_periods = len(equity)
        periods_per_year = 252 * 360
        result.annual_return = (1 + result.total_return) ** (periods_per_year / max(n_periods, 1)) - 1
        
        # 基准收益（买入持有）
        result.benchmark_return = (prices[-1] - prices[0]) / prices[0] if len(prices) > 0 else 0
        result.excess_return = result.total_return - result.benchmark_return
        
        # 风险指标
        if len(returns) > 0:
            result.volatility = np.std(returns) * np.sqrt(periods_per_year)
            
            # 最大回撤
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak
            result.max_drawdown = np.max(drawdown)
            
            # 夏普比率（假设无风险利率为0）
            if result.volatility > 0:
                result.sharpe_ratio = result.annual_return / result.volatility
            
            # Sortino比率
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns) * np.sqrt(periods_per_year)
                if downside_std > 0:
                    result.sortino_ratio = result.annual_return / downside_std
            
            # Calmar比率
            if result.max_drawdown > 0:
                result.calmar_ratio = result.annual_return / result.max_drawdown
        
        # 交易统计
        result.total_trades = len(self.trades)
        
        # 计算胜率
        if result.total_trades > 0:
            # 配对交易计算盈亏
            buy_trades = [t for t in self.trades if t.side == 'BUY']
            sell_trades = [t for t in self.trades if t.side == 'SELL']
            
            wins = []
            losses = []
            
            for i, sell in enumerate(sell_trades):
                if i < len(buy_trades):
                    buy = buy_trades[i]
                    pnl = (sell.price - buy.price) * buy.quantity - buy.commission - sell.commission
                    if pnl > 0:
                        wins.append(pnl)
                    else:
                        losses.append(pnl)
            
            result.win_trades = len(wins)
            result.loss_trades = len(losses)
            result.win_rate = result.win_trades / (result.win_trades + result.loss_trades) if (result.win_trades + result.loss_trades) > 0 else 0
            result.avg_win = np.mean(wins) if wins else 0
            result.avg_loss = np.mean(losses) if losses else 0
            
            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses)) if losses else 0
            result.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # 时间序列
        result.equity_curve = self.equity_curve
        result.drawdown_curve = list(drawdown) if len(returns) > 0 else []
        result.timestamps = self.timestamps
        result.trades = self.trades
        
        return result


# ============================================================
# 可视化
# ============================================================

def plot_backtest_result(result: BacktestResult, save_path: Path = None):
    """绘制回测结果图表"""
    if not HAS_PLOT:
        print("  [WARN] matplotlib未安装，跳过绘图")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    timestamps = result.timestamps
    
    # 1. 权益曲线
    ax1 = axes[0]
    ax1.plot(timestamps, result.equity_curve, 'b-', linewidth=1, label='策略权益')
    ax1.axhline(y=BACKTEST_CONFIG['initial_capital'], color='gray', linestyle='--', alpha=0.5, label='初始资金')
    ax1.set_ylabel('权益 (HKD)')
    ax1.set_title(f'{result.model_name} 策略回测 | 总收益: {result.total_return*100:.2f}% | 夏普: {result.sharpe_ratio:.2f}')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 标记交易点
    for trade in result.trades:
        color = 'green' if trade.side == 'BUY' else 'red'
        marker = '^' if trade.side == 'BUY' else 'v'
        idx = timestamps.index(trade.timestamp) if trade.timestamp in timestamps else -1
        if idx >= 0:
            ax1.scatter(trade.timestamp, result.equity_curve[idx], color=color, marker=marker, s=50, zorder=5)
    
    # 2. 回撤曲线
    ax2 = axes[1]
    if result.drawdown_curve:
        dd_len = len(result.drawdown_curve)
        ts_for_dd = timestamps[:dd_len]
        ax2.fill_between(ts_for_dd, 0, [-d*100 for d in result.drawdown_curve], color='red', alpha=0.3)
        ax2.plot(ts_for_dd, [-d*100 for d in result.drawdown_curve], 'r-', linewidth=1)
    ax2.set_ylabel('回撤 (%)')
    ax2.set_title(f'最大回撤: {result.max_drawdown*100:.2f}%')
    ax2.grid(True, alpha=0.3)
    
    # 3. 收益分布
    ax3 = axes[2]
    if len(result.equity_curve) > 1:
        returns = np.diff(result.equity_curve) / np.array(result.equity_curve[:-1]) * 100
        ax3.plot(timestamps[1:len(returns)+1], returns, 'b-', linewidth=0.5, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('收益率 (%)')
    ax3.set_xlabel('时间')
    ax3.set_title(f'胜率: {result.win_rate*100:.1f}% | 盈亏比: {result.profit_factor:.2f}')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  图表已保存: {save_path}")
    
    plt.close()


def print_backtest_result(result: BacktestResult):
    """打印回测结果"""
    print("\n" + "="*60)
    print(f"  {result.model_name} 回测结果")
    print("="*60)
    
    print("\n  【收益指标】")
    print(f"    总收益率:      {result.total_return*100:>10.2f}%")
    print(f"    年化收益率:    {result.annual_return*100:>10.2f}%")
    print(f"    基准收益率:    {result.benchmark_return*100:>10.2f}%")
    print(f"    超额收益:      {result.excess_return*100:>10.2f}%")
    
    print("\n  【风险指标】")
    print(f"    年化波动率:    {result.volatility*100:>10.2f}%")
    print(f"    最大回撤:      {result.max_drawdown*100:>10.2f}%")
    print(f"    夏普比率:      {result.sharpe_ratio:>10.2f}")
    print(f"    Sortino比率:   {result.sortino_ratio:>10.2f}")
    print(f"    Calmar比率:    {result.calmar_ratio:>10.2f}")
    
    print("\n  【交易统计】")
    print(f"    总交易次数:    {result.total_trades:>10d}")
    print(f"    盈利次数:      {result.win_trades:>10d}")
    print(f"    亏损次数:      {result.loss_trades:>10d}")
    print(f"    胜率:          {result.win_rate*100:>10.2f}%")
    print(f"    平均盈利:      {result.avg_win:>10.2f}")
    print(f"    平均亏损:      {result.avg_loss:>10.2f}")
    print(f"    盈亏比:        {result.profit_factor:>10.2f}")
    
    print("="*60)


# ============================================================
# 主流程
# ============================================================

def load_model_for_backtest(model_name: str, input_dim: int = 25, seq_len: int = 100):
    """加载深度学习模型用于回测"""
    sys.path.insert(0, str(Path(__file__).parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_trainer", Path(__file__).parent / "13_model_trainer.py")
    model_trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_trainer)
    create_model = model_trainer.create_model
    
    model_path = MODEL_DIR / model_name / 'model.pt'
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model = create_model(model_name, input_dim, seq_len)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    return model


def load_ml_model_for_backtest(model_name: str):
    """加载ML模型用于回测"""
    sys.path.insert(0, str(Path(__file__).parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_trainer", Path(__file__).parent / "13_model_trainer.py")
    model_trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_trainer)
    MLBaselines = model_trainer.MLBaselines
    
    model_path = MODEL_DIR / model_name / 'model.pkl'
    
    if not model_path.exists():
        raise FileNotFoundError(f"ML模型文件不存在: {model_path}")
    
    ml_instance, ml_type = MLBaselines.load_model(model_path)
    return ml_instance, ml_type


# ML模型列表
ML_MODELS = ['arima', 'logistic', 'xgboost', 'rf']
DL_MODELS = ['lstm', 'gru', 'deeplob', 'transformer', 'smart_trans']


def run_backtest(
    model_name: str,
    data_dir: Path,
    seq_len: int = 100,
    output_dir: Path = None
) -> BacktestResult:
    """
    运行单个模型的回测
    
    Args:
        model_name: 模型名称
        data_dir: 数据目录
        seq_len: 序列长度
        output_dir: 输出目录
        
    Returns:
        回测结果
    """
    print(f"\n{'='*60}")
    print(f"  回测模型: {model_name.upper()}")
    print("="*60)
    
    # 判断模型类型
    is_ml_model = model_name.lower() in ML_MODELS
    
    # 加载数据
    print("\n[1] 加载数据...")
    dataset_dir = data_dir / f'dataset_T{seq_len}_k20'
    
    X_test = np.load(dataset_dir / 'X_test.npy')
    y_test = np.load(dataset_dir / 'y_test.npy')
    
    print(f"  测试集: {len(X_test)} 样本")
    print(f"  模型类型: {'ML' if is_ml_model else 'DL'}")
    
    # 加载标准化器
    import pickle
    scaler_path = dataset_dir / 'scaler.pkl'
    scaler_params = None
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler_params = pickle.load(f)
    
    # 加载特征数据（用于获取价格）
    feature_files = sorted(data_dir.glob('features_*.parquet'))
    if feature_files:
        df = pd.read_parquet(feature_files[-1])
        prices = df['mid_price'].values[-len(X_test):]
        timestamps = pd.to_datetime(df['ts']).tolist()[-len(X_test):]
    else:
        # 使用模拟价格
        prices = np.cumsum(np.random.randn(len(X_test)) * 0.1) + 400
        timestamps = [datetime.now() for _ in range(len(X_test))]
    
    # 确保长度匹配
    min_len = min(len(prices), len(X_test))
    prices = prices[:min_len]
    X_test = X_test[:min_len]
    timestamps = timestamps[:min_len]
    
    # 加载模型并生成信号
    print("\n[2] 加载模型...")
    
    if is_ml_model:
        # 加载ML模型
        ml_instance, ml_type = load_ml_model_for_backtest(model_name)
        print(f"  ML模型类型: {ml_type}")
        
        # 生成信号
        print("\n[3] 生成交易信号...")
        signal_gen = MLSignalGenerator(ml_instance, ml_type)
        signals, confidences = signal_gen.generate_signals(X_test)
    else:
        # 加载DL模型
        model = load_model_for_backtest(model_name, input_dim=X_test.shape[2], seq_len=seq_len)
        
        # 生成信号
        print("\n[3] 生成交易信号...")
        signal_gen = SignalGenerator(model, scaler_params, seq_len)
        signals, confidences = signal_gen.generate_signals(X_test)
    
    signal_counts = {
        'BUY': (signals == 2).sum(),
        'HOLD': (signals == 1).sum(),
        'SELL': (signals == 0).sum()
    }
    print(f"  信号分布: {signal_counts}")
    
    # 运行回测
    print("\n[4] 运行回测...")
    engine = BacktestEngine()
    result = engine.run(prices, signals, confidences, timestamps)
    result.model_name = model_name.upper()
    
    # 打印结果
    print_backtest_result(result)
    
    # 保存结果
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存图表
        plot_backtest_result(result, output_dir / f'backtest_{model_name}.png')
        
        # 保存指标
        metrics = {
            'model_name': result.model_name,
            'total_return': result.total_return,
            'annual_return': result.annual_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'total_trades': result.total_trades,
            'profit_factor': result.profit_factor
        }
        with open(output_dir / f'metrics_{model_name}.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    
    return result


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='策略回测模块')
    parser.add_argument('--model', type=str, nargs='+', default=['transformer'],
                        help='要回测的模型 (arima, logistic, xgboost, rf, lstm, gru, deeplob, transformer, smart_trans, all/ml/deep)')
    parser.add_argument('--data', type=str, default='data/processed/combined',
                        help='数据目录')
    parser.add_argument('--output', type=str, default='backtest_results',
                        help='输出目录')
    parser.add_argument('--seq-len', type=int, default=100, help='序列长度')
    parser.add_argument('--compare', action='store_true', help='对比所有模型')
    
    args = parser.parse_args()
    
    print("="*60)
    print("  OFI论文 - 策略回测模块")
    print("="*60)
    print(f"  数据目录: {args.data}")
    print(f"  初始资金: ${BACKTEST_CONFIG['initial_capital']:,.0f}")
    print(f"  佣金率: {BACKTEST_CONFIG['commission_rate']*100:.2f}%")
    
    # 确定要回测的模型
    if 'all' in args.model:
        # 所有9个模型 (ML + DL)
        model_names = ML_MODELS + DL_MODELS
    elif 'ml' in args.model:
        # 仅ML模型
        model_names = ML_MODELS
    elif 'deep' in args.model or 'dl' in args.model:
        # 仅深度学习模型
        model_names = DL_MODELS
    else:
        model_names = args.model
    
    # 运行回测
    results = {}
    output_dir = Path(args.output)
    
    for model_name in model_names:
        try:
            result = run_backtest(
                model_name=model_name,
                data_dir=Path(args.data),
                seq_len=args.seq_len,
                output_dir=output_dir
            )
            results[model_name] = result
        except Exception as e:
            print(f"  [ERROR] {model_name} 回测失败: {e}")
    
    # 对比结果
    if args.compare and len(results) > 1:
        print("\n" + "="*60)
        print("  模型回测对比")
        print("="*60)
        
        comparison = []
        for name, result in results.items():
            comparison.append({
                'Model': name.upper(),
                'Return': f"{result.total_return*100:.2f}%",
                'Sharpe': f"{result.sharpe_ratio:.2f}",
                'MaxDD': f"{result.max_drawdown*100:.2f}%",
                'WinRate': f"{result.win_rate*100:.1f}%",
                'Trades': result.total_trades
            })
        
        df = pd.DataFrame(comparison)
        print(df.to_string(index=False))
        
        # 保存对比结果
        df.to_csv(output_dir / 'comparison.csv', index=False)
    
    print("\n[DONE] 回测完成！")


if __name__ == "__main__":
    main()
