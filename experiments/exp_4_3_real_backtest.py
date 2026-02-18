# -*- coding: utf-8 -*-
"""
真实数据回测脚本
从数据库获取数据，加载训练好的模型，进行预测和回测

输出:
- table_4_3_1_economic_value.csv  (替换模拟数据版本)
- fig_4_3_1_equity_curves.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from exp_config import *
import pandas as pd
import numpy as np
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
setup_plot()

# PyTorch
import torch
import torch.nn as nn

# 导入模型定义
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "kline_model_trainer", 
        PROJECT_ROOT / "scripts" / "13b_kline_model_trainer.py"
    )
    trainer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer_module)
    
    # 获取模型类（正确的类名）
    LSTMBaseline = trainer_module.LSTMBaseline
    GRUBaseline = trainer_module.GRUBaseline
    CNNLSTMBaseline = trainer_module.CNNLSTMBaseline
    TransformerBaseline = trainer_module.TransformerBaseline
    PVTransformer = trainer_module.PVTransformer
    MultiScalePVTransformer = trainer_module.MultiScalePVTransformer
    SklearnModelWrapper = trainer_module.SklearnModelWrapper
    
    # 注入到__main__命名空间（用于pickle加载）
    import __main__
    __main__.SklearnModelWrapper = SklearnModelWrapper
    __main__.LSTMBaseline = LSTMBaseline
    __main__.GRUBaseline = GRUBaseline
    __main__.CNNLSTMBaseline = CNNLSTMBaseline
    __main__.TransformerBaseline = TransformerBaseline
    __main__.PVTransformer = PVTransformer
    __main__.MultiScalePVTransformer = MultiScalePVTransformer
    
    print("[OK] Model classes imported")
except Exception as e:
    print(f"[ERROR] Could not import model classes: {e}")
    sys.exit(1)


@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    equity_curve: np.ndarray


def get_db_connection():
    """获取数据库连接"""
    import psycopg2
    import os
    
    conn = psycopg2.connect(
        host="127.0.0.1",
        port=int(os.getenv("DB_PORT", "5433")),
        database=os.getenv("DB_NAME", "futu_ofi"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "ofi123456")
    )
    return conn


def fetch_kline_data(code: str, n_samples: int = 10000) -> pd.DataFrame:
    """从数据库获取K线数据"""
    conn = get_db_connection()
    
    query = f"""
    SELECT ts as time_key, open_price as open, high_price as high, 
           low_price as low, close_price as close, volume
    FROM kline
    WHERE code = '{code}' AND ktype = 'K_1M'
    ORDER BY ts DESC
    LIMIT {n_samples}
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    if df.empty:
        return None
    
    # 按时间正序
    df = df.sort_values('time_key').reset_index(drop=True)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算特征（与训练时一致）"""
    # 基础特征
    df['return_1'] = df['close'].pct_change()
    df['return_5'] = df['close'].pct_change(5)
    df['return_20'] = df['close'].pct_change(20)
    df['return_60'] = df['close'].pct_change(60)
    df['kline_position'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df['range_pct'] = (df['high'] - df['low']) / df['open']
    df['relative_volume'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_change'] = df['volume'].pct_change()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_position'] = (df['close'] - sma20) / (2 * std20 + 1e-8)
    
    # ATR
    df['atr_pct'] = df['range_pct'].rolling(14).mean()
    df['volatility_20'] = df['return_1'].rolling(20).std()
    
    # TI (Trade Imbalance)
    df['ti'] = df['kline_position'] * df['volume']
    df['ti_5'] = df['ti'].rolling(5).sum()
    df['ti_60'] = df['ti'].rolling(60).sum()
    df['ti_zscore'] = (df['ti'] - df['ti'].rolling(10).mean()) / (df['ti'].rolling(10).std() + 1e-8)
    df['return_zscore'] = (df['return_1'] - df['return_1'].rolling(10).mean()) / (df['return_1'].rolling(10).std() + 1e-8)
    df['pv_corr'] = df['return_1'].rolling(20).corr(df['volume'])
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd_dif'] = ema12 - ema26
    df['macd_dea'] = df['macd_dif'].ewm(span=9).mean()
    df['macd'] = df['macd_dif'] - df['macd_dea']
    
    # Market regime
    df['market_regime'] = 1
    
    # 标签（未来5分钟收益率）
    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
    df['label'] = np.where(df['future_return'] > 0.002, 2, 
                          np.where(df['future_return'] < -0.002, 0, 1))
    
    return df


def prepare_sequences(df: pd.DataFrame, seq_len: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """准备序列数据"""
    feature_cols = [
        'kline_position', 'range_pct', 'return_1', 'return_5', 'return_20', 
        'return_60', 'return_zscore', 'atr_pct', 'volatility_20', 'ti', 
        'ti_5', 'ti_60', 'ti_zscore', 'relative_volume', 'volume_change',
        'pv_corr', 'rsi', 'bb_position', 'macd_dif', 'macd_dea', 'macd', 'market_regime'
    ]
    
    df = df.dropna()
    X = df[feature_cols].values
    y = df['label'].values
    prices = df['close'].values
    
    # 处理inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)
    
    # Z-score标准化
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8
    X = (X - mean) / std
    
    # 创建滑动窗口
    X_seq, y_seq, prices_seq = [], [], []
    for i in range(len(X) - seq_len - 5):  # 预留5步用于标签
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len-1])
        prices_seq.append(prices[i+seq_len-1:i+seq_len+5])  # 当前价格及未来5步
    
    return np.array(X_seq), np.array(y_seq), np.array(prices_seq)


def load_pytorch_model(model_path: Path, model_type: str, input_dim: int = 22, seq_len: int = 60):
    """加载PyTorch模型"""
    device = torch.device('cpu')
    
    # 特征分离（与训练脚本一致）
    # 前14维是价格相关特征，后8维是成交量相关特征
    price_dim = 14
    volume_dim = 8
    
    # 根据模型类型创建模型（正确的参数）
    if model_type == 'lstm':
        model = LSTMBaseline(input_dim=input_dim, seq_len=seq_len, hidden_dim=128, num_layers=2, num_classes=3)
    elif model_type == 'gru':
        model = GRUBaseline(input_dim=input_dim, seq_len=seq_len, hidden_dim=128, num_layers=2, num_classes=3)
    elif model_type == 'cnn_lstm':
        model = CNNLSTMBaseline(input_dim=input_dim, seq_len=seq_len, num_classes=3)
    elif model_type == 'transformer':
        model = TransformerBaseline(input_dim=input_dim, seq_len=seq_len, num_classes=3)
    elif model_type == 'pv_transformer':
        # PVTransformer需要分离的价格和成交量维度
        model = PVTransformer(price_dim=price_dim, volume_dim=volume_dim, seq_len=seq_len, num_classes=3)
    elif model_type == 'multi_scale':
        # MultiScalePVTransformer需要多尺度序列长度
        scale_seq_lens = {'1M': 60, '5M': 24, '60M': 12, 'DAY': 20}
        model = MultiScalePVTransformer(price_dim=price_dim, volume_dim=volume_dim, 
                                        scale_seq_lens=scale_seq_lens, num_classes=3)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model, model_type


def load_sklearn_model(model_path: Path):
    """加载sklearn模型"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def predict_pytorch(model, X: np.ndarray, model_type: str) -> np.ndarray:
    """PyTorch模型预测"""
    device = torch.device('cpu')
    model.to(device)
    
    # 特征分离索引（与训练一致）
    # 前14维: kline_position, range_pct, return_1, return_5, return_20, return_60, 
    #        return_zscore, atr_pct, volatility_20, ti, ti_5, ti_60, ti_zscore, relative_volume
    # 后8维: volume_change, pv_corr, rsi, bb_position, macd_dif, macd_dea, macd, market_regime
    price_idx = list(range(14))
    volume_idx = list(range(14, 22))
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        
        if model_type == 'pv_transformer':
            # 分离价格和成交量特征
            price_feat = X_tensor[:, :, price_idx]
            volume_feat = X_tensor[:, :, volume_idx]
            outputs = model(price_feat, volume_feat)
            
        elif model_type == 'multi_scale':
            # 多尺度模型需要特殊处理
            # 分离价格和成交量特征
            price_feat = X_tensor[:, :, price_idx]  # (batch, 60, 14)
            volume_feat = X_tensor[:, :, volume_idx]  # (batch, 60, 8)
            
            # 创建多尺度输入（下采样）
            scale_data = {
                '1M': (price_feat, volume_feat),
                '5M': (price_feat[:, ::3, :] if price_feat.shape[1] >= 24 else price_feat[:, :24, :],
                       volume_feat[:, ::3, :] if volume_feat.shape[1] >= 24 else volume_feat[:, :24, :]),
                '60M': (price_feat[:, ::5, :] if price_feat.shape[1] >= 12 else price_feat[:, :12, :],
                        volume_feat[:, ::5, :] if volume_feat.shape[1] >= 12 else volume_feat[:, :12, :]),
                'DAY': (price_feat[:, ::3, :] if price_feat.shape[1] >= 20 else price_feat[:, :20, :],
                        volume_feat[:, ::3, :] if volume_feat.shape[1] >= 20 else volume_feat[:, :20, :]),
            }
            outputs = model(scale_data)
        else:
            outputs = model(X_tensor)
        
        _, predictions = torch.max(outputs, 1)
        return predictions.cpu().numpy()


def predict_sklearn(model, X: np.ndarray) -> np.ndarray:
    """sklearn模型预测"""
    # 展平 (N, T, F) -> (N, T*F)
    if X.ndim == 3:
        X_flat = X.reshape(X.shape[0], -1)
    else:
        X_flat = X
    
    X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)
    X_flat = np.clip(X_flat, -1e6, 1e6)
    
    return model.predict(X_flat)


def run_backtest(predictions: np.ndarray, prices_seq: np.ndarray, 
                 cost_rate: float = 0.0005,
                 holding_period: int = 5,
                 stop_loss: float = 0.02,
                 take_profit: float = 0.03) -> BacktestResult:
    """
    执行回测 - 与论文对齐的交易策略
    
    策略逻辑：
    - 预测=2（上涨）→ 做多
    - 预测=0（下跌）→ 做空
    - 预测=1（平稳）→ 维持当前仓位
    
    风控机制：
    - 最小持仓期：5分钟（对应论文的预测步长）
    - 止损：-2%
    - 止盈：+3%
    
    交易成本：单边0.05%（论文基准）
    """
    n = len(predictions)
    initial_capital = 1_000_000
    
    # 计算每步收益率（5分钟累积收益）
    returns_5m = np.zeros(n)
    for i in range(n):
        if i + holding_period < len(prices_seq):
            entry_price = prices_seq[i, 0]
            exit_price = prices_seq[min(i + holding_period, len(prices_seq)-1), 0]
            if entry_price > 0:
                returns_5m[i] = (exit_price - entry_price) / entry_price
    
    # 按holding_period间隔决策（每5分钟决策一次）
    decision_points = list(range(0, n - holding_period, holding_period))
    
    # 模拟交易
    equity = initial_capital
    equity_curve = [equity]
    position = 0  # -1=空, 0=无, +1=多
    entry_price = 0
    n_trades = 0
    n_wins = 0
    n_active = 0
    
    for i, dp in enumerate(decision_points):
        pred = predictions[dp]
        current_price = prices_seq[dp, 0]
        
        # 检查是否需要止损/止盈
        if position != 0 and entry_price > 0:
            pnl_pct = position * (current_price - entry_price) / entry_price
            
            # 触发止损或止盈
            if pnl_pct <= -stop_loss or pnl_pct >= take_profit:
                # 平仓
                trade_return = pnl_pct - cost_rate  # 扣除平仓成本
                equity *= (1 + trade_return)
                if trade_return > 0:
                    n_wins += 1
                n_active += 1
                position = 0
                entry_price = 0
        
        # 根据预测决定新仓位
        target_position = 0
        if pred == 2:  # 预测上涨
            target_position = 1
        elif pred == 0:  # 预测下跌
            target_position = -1
        # pred == 1 时维持当前仓位
        else:
            target_position = position
        
        # 仓位变化时执行交易
        if target_position != position:
            # 先平旧仓（如果有）
            if position != 0 and entry_price > 0:
                pnl_pct = position * (current_price - entry_price) / entry_price
                trade_return = pnl_pct - cost_rate
                equity *= (1 + trade_return)
                if trade_return > 0:
                    n_wins += 1
                n_active += 1
            
            # 开新仓
            if target_position != 0:
                entry_price = current_price
                equity *= (1 - cost_rate)  # 开仓成本
                n_trades += 1
            else:
                entry_price = 0
            
            position = target_position
        
        equity_curve.append(equity)
    
    # 处理最后的持仓
    if position != 0 and entry_price > 0:
        final_price = prices_seq[-1, 0]
        pnl_pct = position * (final_price - entry_price) / entry_price
        trade_return = pnl_pct - cost_rate
        equity *= (1 + trade_return)
        if trade_return > 0:
            n_wins += 1
        n_active += 1
    equity_curve.append(equity)
    
    equity_curve = np.array(equity_curve)
    
    # 计算指标
    total_return = (equity_curve[-1] / initial_capital - 1) * 100
    total_return = np.clip(total_return, -99, 1000)
    
    # 年化收益
    n_decisions = len(decision_points)
    n_days = n_decisions * holding_period / 390
    n_years = max(n_days / 250, 0.01)
    if total_return > -99:
        annual_return = ((1 + total_return/100) ** (1/n_years) - 1) * 100
        annual_return = np.clip(annual_return, -99, 500)
    else:
        annual_return = total_return
    
    # 夏普比率
    if len(equity_curve) > 2:
        period_returns = np.diff(equity_curve) / equity_curve[:-1]
        # 转换为日收益
        periods_per_day = 390 // holding_period
        if len(period_returns) >= periods_per_day:
            n_complete = (len(period_returns) // periods_per_day) * periods_per_day
            daily_returns = period_returns[:n_complete].reshape(-1, periods_per_day).sum(axis=1)
            sharpe = np.sqrt(252) * np.mean(daily_returns) / (np.std(daily_returns) + 1e-8)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0
    sharpe = np.clip(sharpe, -10, 10)
    
    # 最大回撤
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / (peak + 1e-8) * 100
    max_drawdown = np.clip(np.max(drawdown), 0, 100)
    
    # 胜率
    win_rate = (n_wins / max(n_active, 1)) * 100
    
    return BacktestResult(
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        total_trades=n_trades,
        equity_curve=equity_curve
    )


def run_buyhold_backtest(prices_seq: np.ndarray, holding_period: int = 5) -> BacktestResult:
    """
    Buy & Hold 基准策略
    始终持有多头仓位，用于对比
    """
    n = len(prices_seq)
    initial_capital = 1_000_000
    
    # 计算总收益（从第一个价格到最后一个价格）
    start_price = prices_seq[0, 0]
    end_price = prices_seq[-1, 0]
    
    if start_price > 0:
        total_return = (end_price - start_price) / start_price * 100
    else:
        total_return = 0.0
    total_return = np.clip(total_return, -99, 1000)
    
    # 构建净值曲线
    prices = prices_seq[:, 0]
    equity_curve = initial_capital * prices / prices[0]
    
    # 年化收益
    n_minutes = n
    n_days = n_minutes / 390
    n_years = max(n_days / 250, 0.01)
    annual_return = ((1 + total_return/100) ** (1/n_years) - 1) * 100 if total_return > -99 else total_return
    annual_return = np.clip(annual_return, -99, 500)
    
    # 夏普比率
    returns = np.diff(prices) / prices[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    minutes_per_day = 390
    if len(returns) >= minutes_per_day:
        n_complete_days = len(returns) // minutes_per_day
        daily_returns = returns[:n_complete_days * minutes_per_day].reshape(n_complete_days, minutes_per_day).sum(axis=1)
        sharpe = np.sqrt(252) * np.mean(daily_returns) / (np.std(daily_returns) + 1e-8)
    else:
        sharpe = 0.0
    sharpe = np.clip(sharpe, -10, 10)
    
    # 最大回撤
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / (peak + 1e-8) * 100
    max_drawdown = np.clip(np.max(drawdown), 0, 100)
    
    # 胜率
    winning = np.sum(returns > 0)
    win_rate = winning / len(returns) * 100 if len(returns) > 0 else 0
    
    return BacktestResult(
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        total_trades=1,
        equity_curve=equity_curve
    )


def main():
    print("=" * 70)
    print("  真实数据回测 - 使用训练好的模型")
    print("=" * 70)
    
    # 模型配置
    model_configs = [
        ('lstm', 'lstm', 'LSTM'),
        ('gru', 'gru', 'GRU'),
        ('cnn_lstm', 'cnn_lstm', 'CNN-LSTM'),
        ('transformer', 'transformer', 'Transformer'),
        ('pv_transformer', 'pv_transformer', 'PV-Transformer'),
        ('multi_scale', 'multi_scale', 'PV-Transformer+LSF'),
        ('logistic_regression', 'sklearn', 'LogisticRegression'),
        ('random_forest', 'sklearn', 'RandomForest'),
        ('xgboost', 'sklearn', 'XGBoost'),
    ]
    
    all_results = {}
    all_equity_curves = {}
    
    # 测试股票（用腾讯作为代表）
    test_code = 'HK.00700'
    print(f"\n[1/2] 获取 {test_code} 数据...")
    
    # 获取数据：3个月测试期 ≈ 60天 × 390分钟 = 23,400分钟
    # 总共获取50000条，保证有足够的训练+测试数据
    df = fetch_kline_data(test_code, n_samples=50000)
    if df is None:
        print("[ERROR] 无法获取数据")
        return
    
    print(f"  获取 {len(df)} 条K线数据")
    
    # 计算特征
    print("[2/2] 计算特征...")
    df = compute_features(df)
    
    # 准备序列
    X, y, prices_seq = prepare_sequences(df, seq_len=60)
    print(f"  生成 {len(X)} 个样本序列")
    
    # 测试期：3个月数据（约60个交易日）
    # 覆盖不同市场状态
    test_size = min(60 * 390, len(X) // 3)  # 3个月或33%数据
    test_start = len(X) - test_size
    X_test = X[test_start:]
    y_test = y[test_start:]
    prices_test = prices_seq[test_start:]
    
    n_test_days = len(X_test) / 390
    print(f"  测试集: {len(X_test)} 个样本 (约{n_test_days:.1f}个交易日)")
    
    # 交易参数设定（与论文对齐：预测步长5-30分钟）
    COST_RATE = 0.0005    # 单边0.05%（论文基准）
    HOLDING_PERIOD = 30   # 持仓周期30分钟（论文预测步长上限，降低交易频率）
    STOP_LOSS = 0.015     # 止损1.5%（日内合理风控）
    TAKE_PROFIT = 0.02    # 止盈2%（日内合理预期）
    
    print(f"\n交易参数: 成本={COST_RATE*100:.2f}%, 持仓={HOLDING_PERIOD}分钟, "
          f"止损={STOP_LOSS*100:.1f}%, 止盈={TAKE_PROFIT*100:.1f}%")
    
    # 先计算Buy&Hold基准
    print("\n[基准] 计算 Buy&Hold 策略...")
    buyhold_result = run_buyhold_backtest(prices_test, holding_period=HOLDING_PERIOD)
    all_results['Buy&Hold'] = buyhold_result
    all_equity_curves['Buy&Hold'] = buyhold_result.equity_curve
    print(f"  Buy&Hold: 收益={buyhold_result.total_return:.2f}%, 夏普={buyhold_result.sharpe_ratio:.2f}")
    
    # 对每个模型进行回测
    code_str = test_code.replace('.', '_')
    
    for model_dir, model_type, display_name in model_configs:
        print(f"\n回测 {display_name}...")
        
        if model_type == 'sklearn':
            model_path = MODELS_DIR / model_dir / f"model_{code_str}_1M.pkl"
        else:
            model_path = MODELS_DIR / model_dir / f"model_{code_str}_1M.pt"
        
        if not model_path.exists():
            print(f"  [SKIP] 模型文件不存在: {model_path}")
            continue
        
        try:
            # 加载模型
            if model_type == 'sklearn':
                model = load_sklearn_model(model_path)
                predictions = predict_sklearn(model, X_test)
            else:
                model, _ = load_pytorch_model(model_path, model_type)
                predictions = predict_pytorch(model, X_test, model_type)
            
            # 打印预测分布（检查模型是否在有效预测）
            n_long = np.sum(predictions == 2)   # 做多
            n_short = np.sum(predictions == 0)  # 做空
            n_neutral = np.sum(predictions == 1)  # 空仓
            print(f"  预测分布: 做多={n_long} ({100*n_long/len(predictions):.1f}%), "
                  f"做空={n_short} ({100*n_short/len(predictions):.1f}%), "
                  f"空仓={n_neutral} ({100*n_neutral/len(predictions):.1f}%)")
            
            # 执行回测（使用论文设定的参数）
            result = run_backtest(predictions, prices_test, 
                                  cost_rate=COST_RATE,
                                  holding_period=HOLDING_PERIOD,
                                  stop_loss=STOP_LOSS,
                                  take_profit=TAKE_PROFIT)
            all_results[display_name] = result
            all_equity_curves[display_name] = result.equity_curve
            
            print(f"  [OK] 收益: {result.total_return:.2f}%, 夏普: {result.sharpe_ratio:.2f}, "
                  f"回撤: {result.max_drawdown:.2f}%, 交易: {result.total_trades}")
            
        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()
    
    # 保存结果
    if all_results:
        # 表格
        table_data = []
        for model_name, result in all_results.items():
            table_data.append({
                '模型': model_name,
                '总收益率(%)': f"{result.total_return:.2f}",
                '年化收益率(%)': f"{result.annual_return:.2f}",
                '夏普比率': f"{result.sharpe_ratio:.2f}",
                '最大回撤(%)': f"{result.max_drawdown:.2f}",
                '胜率(%)': f"{result.win_rate:.1f}",
                '交易次数': result.total_trades,
            })
        
        df_results = pd.DataFrame(table_data)
        table_path = TABLES_DIR / 'table_4_3_1_economic_value.csv'
        df_results.to_csv(table_path, index=False, encoding='utf-8-sig')
        print(f"\n表格已保存: {table_path}")
        
        # 图表
        plt.figure(figsize=(12, 6))
        for model_name, equity in all_equity_curves.items():
            normalized = equity / equity[0]
            plt.plot(normalized, label=f'{model_name}', linewidth=1.5)
        
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='基准')
        plt.xlabel('时间步')
        plt.ylabel('归一化净值')
        plt.title('图 4.3-1: 各模型策略净值曲线对比（真实数据）')
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        
        fig_path = FIGURES_DIR / 'fig_4_3_1_equity_curves.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"图表已保存: {fig_path}")
        
        # 打印汇总
        print("\n" + "=" * 70)
        print("  表 4.3-1: 模型经济价值对比（真实数据）")
        print("=" * 70)
        print(df_results.to_string(index=False))


if __name__ == "__main__":
    set_seed()
    main()
