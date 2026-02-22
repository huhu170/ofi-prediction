# -*- coding: utf-8 -*-
"""
单模型回测脚本 - 完整版（详细统计+分批预测）
用法: python backtest_single.py lstm [stock_code]
"""

import sys
import gc
import io
from pathlib import Path

# 修复Windows编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

import pandas as pd
import numpy as np
import pickle
import torch
from datetime import datetime
import importlib.util

# 预加载模型定义模块（只加载一次，避免stdout冲突）
_trainer_module = None

def _get_trainer_module():
    global _trainer_module
    if _trainer_module is None:
        spec = importlib.util.spec_from_file_location(
            "trainer", Path(__file__).parent.parent / "scripts" / "08_model_trainer.py")
        _trainer_module = importlib.util.module_from_spec(spec)
        
        # 临时保存stdout
        import sys as _sys
        _old_stdout = _sys.stdout
        _old_stderr = _sys.stderr
        
        try:
            spec.loader.exec_module(_trainer_module)
        except:
            pass
        finally:
            # 恢复stdout
            _sys.stdout = _old_stdout
            _sys.stderr = _old_stderr
        
        # 注册到__main__
        import __main__
        __main__.SklearnModelWrapper = _trainer_module.SklearnModelWrapper
    
    return _trainer_module

# 配置
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
TABLES_DIR = PROJECT_ROOT / 'outputs' / 'ch4' / 'tables'

# ===== 参数配置 =====
N_SAMPLES = 40000       # 数据量（约5个月）
BATCH_SIZE = 512        # 分批预测大小
USE_GPU = True          # 是否使用GPU

# 交易参数
COST_RATE = 0.0         # 去除交易成本
DECISION_INTERVAL = 5   # 每5分钟决策
CONFIRM_BARS = 1        # 单个信号就开仓
PROB_THRESHOLD = 0.0    # 无概率阈值限制
STOP_LOSS = 0.01        # 止损1%
TAKE_PROFIT = 0.02      # 止盈2%
NEUTRAL_CLOSE = True    # neutral信号是否平仓（False=仅反向信号平仓）

# 10只股票列表（用英文名避免乱码）
STOCK_LIST = [
    ('HK.00700', 'Tencent'),
    ('HK.00005', 'HSBC'),
    ('HK.09988', 'Alibaba'),
    ('HK.01810', 'Xiaomi'),
    ('HK.00939', 'CCB'),
    ('HK.01299', 'AIA'),
    ('HK.00941', 'ChinaMobile'),
    ('HK.03690', 'Meituan'),
    ('HK.01211', 'BYD'),
    ('HK.00388', 'HKEX'),
]

BARS_PER_DAY = 78  # 390 min / DECISION_INTERVAL(5 min)

# 检测GPU
DEVICE = 'cuda' if USE_GPU and torch.cuda.is_available() else 'cpu'


def compute_sharpe(equity_history, bars_per_day=BARS_PER_DAY):
    """从5分钟级净值曲线计算年化夏普比率 (risk-free = 0)"""
    eq = np.array(equity_history)
    if len(eq) < bars_per_day * 5:
        return 0.0
    daily_eq = [eq[0]]
    for d in range(bars_per_day, len(eq), bars_per_day):
        daily_eq.append(eq[min(d, len(eq) - 1)])
    daily_eq = np.array(daily_eq)
    daily_returns = np.diff(daily_eq) / daily_eq[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]
    if len(daily_returns) < 10 or np.std(daily_returns, ddof=1) == 0:
        return 0.0
    return (np.mean(daily_returns) / np.std(daily_returns, ddof=1)) * np.sqrt(252)


def log(msg):
    """安全打印"""
    try:
        print(msg, flush=True)
    except:
        pass


DATA_DIR = PROJECT_ROOT / 'data' / 'processed'


def fetch_data(code: str, n_samples: int = N_SAMPLES):
    """从本地parquet文件获取K线数据（原DB版已替换）"""
    code_dir = code.replace('.', '_')
    parquet_path = DATA_DIR / code_dir / 'kline_cleaned_1M.parquet'
    if not parquet_path.exists():
        log(f"    [WARN] {parquet_path} not found")
        return None, None, None
    df = pd.read_parquet(parquet_path)
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ('open_price',): col_map[c] = 'open'
        elif cl in ('high_price',): col_map[c] = 'high'
        elif cl in ('low_price',): col_map[c] = 'low'
        elif cl in ('close_price',): col_map[c] = 'close'
        elif cl in ('turnover_vol',): col_map[c] = 'volume'
    if col_map:
        df = df.rename(columns=col_map)
    if 'ts' in df.columns:
        df = df.sort_values('ts').reset_index(drop=True)
    elif 'time_key' in df.columns:
        df = df.rename(columns={'time_key': 'ts'}).sort_values('ts').reset_index(drop=True)
    df = df.tail(n_samples).reset_index(drop=True)
    if df.empty:
        return None, None, None
    start_date = df.iloc[0].get('ts', df.index[0])
    end_date = df.iloc[-1].get('ts', df.index[-1])
    return df, start_date, end_date


def compute_features(df):
    """计算特征"""
    df['return_1'] = df['close'].pct_change()
    df['return_5'] = df['close'].pct_change(5)
    df['return_20'] = df['close'].pct_change(20)
    df['return_60'] = df['close'].pct_change(60)
    df['kline_position'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df['range_pct'] = (df['high'] - df['low']) / df['open']
    df['relative_volume'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_change'] = df['volume'].pct_change()
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
    
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_position'] = (df['close'] - sma20) / (2 * std20 + 1e-8)
    df['atr_pct'] = df['range_pct'].rolling(14).mean()
    df['volatility_20'] = df['return_1'].rolling(20).std()
    
    df['ti'] = df['kline_position'] * df['volume']
    df['ti_5'] = df['ti'].rolling(5).sum()
    df['ti_60'] = df['ti'].rolling(60).sum()
    df['ti_zscore'] = (df['ti'] - df['ti'].rolling(10).mean()) / (df['ti'].rolling(10).std() + 1e-8)
    df['return_zscore'] = (df['return_1'] - df['return_1'].rolling(10).mean()) / (df['return_1'].rolling(10).std() + 1e-8)
    df['pv_corr'] = df['return_1'].rolling(20).corr(df['volume'])
    
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd_dif'] = ema12 - ema26
    df['macd_dea'] = df['macd_dif'].ewm(span=9).mean()
    df['macd'] = df['macd_dif'] - df['macd_dea']
    df['market_regime'] = 1
    
    return df


def prepare_sequences(df, seq_len=60):
    """准备序列（全量数据，不分割）"""
    feature_cols = [
        'kline_position', 'range_pct', 'return_1', 'return_5', 'return_20', 
        'return_60', 'return_zscore', 'atr_pct', 'volatility_20', 'ti', 
        'ti_5', 'ti_60', 'ti_zscore', 'relative_volume', 'volume_change',
        'pv_corr', 'rsi', 'bb_position', 'macd_dif', 'macd_dea', 'macd', 'market_regime'
    ]
    
    df = df.dropna()
    X = df[feature_cols].values.astype(np.float32)
    prices = df['close'].values.astype(np.float32)
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)
    
    # 标准化
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8
    X = (X - mean) / std
    
    n = len(X) - seq_len - 5
    X_seq = np.zeros((n, seq_len, X.shape[1]), dtype=np.float32)
    prices_seq = np.zeros((n, 6), dtype=np.float32)
    
    for i in range(n):
        X_seq[i] = X[i:i+seq_len]
        prices_seq[i] = prices[i+seq_len-1:i+seq_len+5]
    
    return X_seq, prices_seq


def run_backtest(predictions, probs, prices_seq):
    """回测策略 - 详细统计版"""
    n = len(predictions)
    equity = 1_000_000
    position = 0
    entry_price = 0
    equity_history = [equity]
    
    # 详细统计
    long_trades = 0      # 做多笔数
    long_wins = 0        # 做多盈利笔数
    short_trades = 0     # 做空笔数
    short_wins = 0       # 做空盈利笔数
    current_direction = 0  # 当前持仓方向
    
    long_confirm = 0
    short_confirm = 0
    
    for i in range(0, n, DECISION_INTERVAL):
        pred = predictions[i]
        prob = probs[i] if probs is not None else 1.0
        price = prices_seq[i, 0]
        
        # 止损止盈
        if position != 0 and entry_price > 0:
            pnl_pct = position * (price - entry_price) / entry_price
            if pnl_pct <= -STOP_LOSS or pnl_pct >= TAKE_PROFIT:
                equity *= (1 + pnl_pct - COST_RATE)
                # 统计
                if current_direction == 1:
                    long_trades += 1
                    if pnl_pct > 0: long_wins += 1
                else:
                    short_trades += 1
                    if pnl_pct > 0: short_wins += 1
                position = 0
                entry_price = 0
                current_direction = 0
                long_confirm = short_confirm = 0
                equity_history.append(equity)
                continue
        
        # neutral平仓（带止损保护）
        if NEUTRAL_CLOSE and pred == 1 and position != 0:
            pnl_pct = position * (price - entry_price) / entry_price
            if pnl_pct < -STOP_LOSS:
                pnl_pct = -STOP_LOSS
            equity *= (1 + pnl_pct - COST_RATE)
            # 统计
            if current_direction == 1:
                long_trades += 1
                if pnl_pct > 0: long_wins += 1
            else:
                short_trades += 1
                if pnl_pct > 0: short_wins += 1
            position = 0
            entry_price = 0
            current_direction = 0
            long_confirm = short_confirm = 0
            equity_history.append(equity)
            continue
        
        # 信号确认
        if pred == 2:
            long_confirm += 1
            short_confirm = 0
        elif pred == 0:
            short_confirm += 1
            long_confirm = 0
        else:
            long_confirm = short_confirm = 0
        
        # 开仓
        if position == 0:
            if long_confirm >= CONFIRM_BARS and prob >= PROB_THRESHOLD:
                position = 1
                current_direction = 1
                entry_price = price
                equity *= (1 - COST_RATE)
                long_confirm = 0
            elif short_confirm >= CONFIRM_BARS and prob >= PROB_THRESHOLD:
                position = -1
                current_direction = -1
                entry_price = price
                equity *= (1 - COST_RATE)
                short_confirm = 0
        
        # 反手
        elif position == 1 and short_confirm >= CONFIRM_BARS and prob >= PROB_THRESHOLD:
            pnl_pct = (price - entry_price) / entry_price
            equity *= (1 + pnl_pct - COST_RATE)
            long_trades += 1
            if pnl_pct > 0: long_wins += 1
            position = -1
            current_direction = -1
            entry_price = price
            equity *= (1 - COST_RATE)
            short_confirm = 0
            
        elif position == -1 and long_confirm >= CONFIRM_BARS and prob >= PROB_THRESHOLD:
            pnl_pct = -(price - entry_price) / entry_price
            equity *= (1 + pnl_pct - COST_RATE)
            short_trades += 1
            if pnl_pct > 0: short_wins += 1
            position = 1
            current_direction = 1
            entry_price = price
            equity *= (1 - COST_RATE)
            long_confirm = 0
        
        equity_history.append(equity)
    
    # 最终平仓
    if position != 0 and entry_price > 0:
        final_price = prices_seq[-1, 0]
        pnl_pct = position * (final_price - entry_price) / entry_price
        equity *= (1 + pnl_pct - COST_RATE)
        if current_direction == 1:
            long_trades += 1
            if pnl_pct > 0: long_wins += 1
        else:
            short_trades += 1
            if pnl_pct > 0: short_wins += 1
    
    # 计算统计
    total_return = (equity / 1_000_000 - 1) * 100
    total_trades = long_trades + short_trades
    total_wins = long_wins + short_wins
    
    equity_arr = np.array(equity_history)
    peak = np.maximum.accumulate(equity_arr)
    max_dd = np.max((peak - equity_arr) / peak) * 100
    
    return {
        'total_return': total_return,
        'max_drawdown': max_dd,
        'long_trades': long_trades,
        'long_wins': long_wins,
        'long_winrate': long_wins / max(long_trades, 1) * 100,
        'short_trades': short_trades,
        'short_wins': short_wins,
        'short_winrate': short_wins / max(short_trades, 1) * 100,
        'total_trades': total_trades,
        'total_wins': total_wins,
        'total_winrate': total_wins / max(total_trades, 1) * 100,
        'equity_history': equity_arr.tolist(),
    }


def load_model(model_path, model_type):
    """加载模型到GPU/CPU"""
    m = _get_trainer_module()
    
    if model_type == 'sklearn':
        with open(model_path, 'rb') as f:
            return pickle.load(f), None
    else:
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if model_type == 'lstm':
            model = m.LSTMBaseline(input_dim=22, seq_len=60)
        elif model_type == 'gru':
            model = m.GRUBaseline(input_dim=22, seq_len=60)
        elif model_type == 'cnn_lstm':
            model = m.CNNLSTMBaseline(input_dim=22, seq_len=60)
        elif model_type == 'transformer':
            model = m.TransformerBaseline(input_dim=22, seq_len=60)
        elif model_type == 'pv_transformer':
            model = m.PVTransformer(price_dim=14, volume_dim=8, seq_len=60)
        elif model_type == 'multi_scale':
            model = m.MultiScalePVTransformer(price_dim=14, volume_dim=8,
                scale_seq_lens={'1M': 60, '5M': 24, '60M': 12})
        
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        
        model = model.to(DEVICE)
        model.eval()
        return model, DEVICE


def predict_batch(model, X, model_type, device):
    """分批预测（优化内存）"""
    n = len(X)
    all_preds = []
    all_probs = []
    
    if model_type == 'sklearn':
        X_flat = X.reshape(n, -1)
        X_flat = np.nan_to_num(X_flat, nan=0, posinf=0, neginf=0)
        preds = model.predict(X_flat)
        try:
            probs_all = model.predict_proba(X_flat)
            probs = np.array([probs_all[i, preds[i]] for i in range(len(preds))])
        except:
            probs = np.ones(len(preds))
        return preds, probs
    
    # PyTorch分批预测
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            batch = X[i:i+BATCH_SIZE]
            X_t = torch.FloatTensor(batch).to(device)
            
            if model_type == 'pv_transformer':
                out = model(X_t[:, :, :14], X_t[:, :, 14:])
            elif model_type == 'multi_scale':
                p, v = X_t[:, :, :14], X_t[:, :, 14:]
                scale_data = {
                    '1M': (p, v),
                    '5M': (p[:, ::3, :], v[:, ::3, :]),
                    '60M': (p[:, ::5, :], v[:, ::5, :]),
                }
                out = model(scale_data)
            else:
                out = model(X_t)
            
            probs_batch = torch.softmax(out, dim=1).cpu().numpy()
            preds_batch = np.argmax(probs_batch, axis=1)
            probs_max = np.array([probs_batch[j, preds_batch[j]] for j in range(len(preds_batch))])
            
            all_preds.append(preds_batch)
            all_probs.append(probs_max)
            
            del X_t, out
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    return np.concatenate(all_preds), np.concatenate(all_probs)


def backtest_single_stock(model_name, code, model_map):
    """单只股票回测 - 使用全量数据"""
    code_str = code.replace('.', '_')
    
    try:
        # 获取数据（含日期范围）
        df, start_date, end_date = fetch_data(code, n_samples=N_SAMPLES)
        if df is None:
            return None
        
        # 计算特征
        df = compute_features(df)
        
        # 准备序列（全量，不分割）
        X, prices = prepare_sequences(df)
        
        del df
        gc.collect()
        
        if len(X) < 5000:
            return None
        
        log(f"    Data: {len(X)} samples ({len(X)//390:.0f} days)")
        log(f"    Period: {str(start_date)[:10]} ~ {str(end_date)[:10]}")
        
        # ARIMA趋势基准（无需预训练模型）
        if model_name == 'arima':
            trends = np.array([np.mean(np.diff(X[i, -6:, 0])) for i in range(len(X))])
            lower_th = np.percentile(trends, 33)
            upper_th = np.percentile(trends, 67)
            preds = np.where(trends < lower_th, 0, np.where(trends > upper_th, 2, 1))
            diff = np.abs(trends - (lower_th + upper_th) / 2)
            probs = 0.34 + 0.33 * (diff / (np.max(diff) + 1e-8))
            result = run_backtest(preds, probs, prices)
            result['start_date'] = str(start_date)[:10]
            result['end_date'] = str(end_date)[:10]
            result['sharpe'] = round(compute_sharpe(result['equity_history']), 2)
            return result
        
        # Buy&Hold
        if model_name == 'buyhold':
            all_prices = prices[:, 0]
            bh_equity = 1_000_000 * all_prices / all_prices[0]
            bh_peak = np.maximum.accumulate(bh_equity)
            bh_dd = np.max((bh_peak - bh_equity) / bh_peak) * 100
            bh_sharpe = round(compute_sharpe(bh_equity.tolist()), 2)
            return {
                'total_return': (all_prices[-1] - all_prices[0]) / all_prices[0] * 100,
                'max_drawdown': bh_dd,
                'sharpe': bh_sharpe,
                'long_trades': 0, 'long_wins': 0, 'long_winrate': 0,
                'short_trades': 0, 'short_wins': 0, 'short_winrate': 0,
                'total_trades': 0, 'total_wins': 0, 'total_winrate': 0,
                'start_date': str(start_date)[:10],
                'end_date': str(end_date)[:10],
            }
        
        # 加载模型
        dir_name, model_type = model_map[model_name]
        ext = 'pkl' if model_type == 'sklearn' else 'pt'
        model_path = MODELS_DIR / dir_name / f"model_{code_str}_1M.{ext}"
        
        if not model_path.exists():
            return None
        
        model, device = load_model(model_path, model_type)
        
        # 分批预测
        preds, probs = predict_batch(model, X, model_type, device)
        
        del model
        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        
        # 回测
        result = run_backtest(preds, probs, prices)
        result['start_date'] = str(start_date)[:10]
        result['end_date'] = str(end_date)[:10]
        result['sharpe'] = round(compute_sharpe(result['equity_history']), 2)
        
        del X, prices, preds, probs
        gc.collect()
        
        return result
        
    except Exception as e:
        log(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    if len(sys.argv) < 2:
        log("Usage: python backtest_single.py <model_name> [stock_code]")
        log("Models: lstm, gru, cnn_lstm, transformer, pv_transformer, multi_scale")
        log("        logistic_regression, random_forest, xgboost, arima, buyhold")
        return
    
    model_name = sys.argv[1].lower()
    single_stock = sys.argv[2] if len(sys.argv) > 2 else None
    
    log("=" * 70)
    log(f"  Backtest: {model_name}")
    log(f"  Device: {DEVICE}")
    log(f"  Data: {N_SAMPLES} samples (~{N_SAMPLES//390} days), Batch: {BATCH_SIZE}")
    log(f"  Params: confirm={CONFIRM_BARS}, prob>={PROB_THRESHOLD*100:.0f}%")
    log(f"  Risk: SL={STOP_LOSS*100}%, TP={TAKE_PROFIT*100}%")
    log("=" * 70)
    
    model_map = {
        'lstm': ('lstm', 'lstm'),
        'gru': ('gru', 'gru'),
        'cnn_lstm': ('cnn_lstm', 'cnn_lstm'),
        'transformer': ('transformer', 'transformer'),
        'pv_transformer': ('pv_transformer', 'pv_transformer'),
        'multi_scale': ('multi_scale', 'multi_scale'),
        'logistic_regression': ('logistic_regression', 'sklearn'),
        'random_forest': ('random_forest', 'sklearn'),
        'xgboost': ('xgboost', 'sklearn'),
    }
    
    if model_name not in ('buyhold', 'arima') and model_name not in model_map:
        log(f"Unknown model: {model_name}")
        return
    
    stocks = [(single_stock, single_stock)] if single_stock else STOCK_LIST
    all_results = []
    
    for code, name in stocks:
        log(f"\n[{code}] {name}...")
        result = backtest_single_stock(model_name, code, model_map)
        
        if result:
            log(f"  Return: {result['total_return']:.2f}%, MaxDD: {result['max_drawdown']:.2f}%")
            log(f"  Long: {result['long_trades']} trades, {result['long_winrate']:.1f}% win")
            log(f"  Short: {result['short_trades']} trades, {result['short_winrate']:.1f}% win")
            log(f"  Total: {result['total_trades']} trades, {result['total_winrate']:.1f}% win")
            
            all_results.append({
                'stock': code,
                'name': name,
                'start_date': result['start_date'],
                'end_date': result['end_date'],
                'return_pct': round(result['total_return'], 2),
                'max_dd_pct': round(result['max_drawdown'], 2),
                'sharpe': result.get('sharpe', 0.0),
                'long_trades': result['long_trades'],
                'long_wins': result['long_wins'],
                'long_winrate': round(result['long_winrate'], 1),
                'short_trades': result['short_trades'],
                'short_wins': result['short_wins'],
                'short_winrate': round(result['short_winrate'], 1),
                'total_trades': result['total_trades'],
                'total_wins': result['total_wins'],
                'total_winrate': round(result['total_winrate'], 1),
            })
        else:
            log(f"  [SKIP]")
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        # 计算汇总
        avg_return = df['return_pct'].mean()
        avg_dd = df['max_dd_pct'].mean()
        avg_sharpe = df['sharpe'].mean()
        total_long = df['long_trades'].sum()
        total_long_wins = df['long_wins'].sum()
        total_short = df['short_trades'].sum()
        total_short_wins = df['short_wins'].sum()
        total_all = df['total_trades'].sum()
        total_all_wins = df['total_wins'].sum()
        
        log("\n" + "=" * 70)
        log(f"  {model_name} Summary ({len(all_results)} stocks)")
        log("=" * 70)
        log(f"  Avg Return: {avg_return:.2f}%")
        log(f"  Avg MaxDD: {avg_dd:.2f}%")
        log(f"  Avg Sharpe: {avg_sharpe:.2f}")
        log(f"  Long: {total_long} trades, {total_long_wins}/{total_long} wins ({total_long_wins/max(total_long,1)*100:.1f}%)")
        log(f"  Short: {total_short} trades, {total_short_wins}/{total_short} wins ({total_short_wins/max(total_short,1)*100:.1f}%)")
        log(f"  Total: {total_all} trades, {total_all_wins}/{total_all} wins ({total_all_wins/max(total_all,1)*100:.1f}%)")
        
        # 保存详细结果
        df.to_csv(TABLES_DIR / f"backtest_{model_name}_detail.csv", index=False, encoding='utf-8-sig')
        
        # 保存汇总
        summary = pd.DataFrame([{
            'model': model_name,
            'n_stocks': len(all_results),
            'avg_return_pct': round(avg_return, 2),
            'avg_max_dd_pct': round(avg_dd, 2),
            'avg_sharpe': round(avg_sharpe, 2),
            'long_trades': total_long,
            'long_wins': total_long_wins,
            'long_winrate': round(total_long_wins/max(total_long,1)*100, 1),
            'short_trades': total_short,
            'short_wins': total_short_wins,
            'short_winrate': round(total_short_wins/max(total_short,1)*100, 1),
            'total_trades': total_all,
            'total_wins': total_all_wins,
            'total_winrate': round(total_all_wins/max(total_all,1)*100, 1),
        }])
        summary.to_csv(TABLES_DIR / f"backtest_{model_name}.csv", index=False, encoding='utf-8-sig')
        
        log(f"\nSaved: backtest_{model_name}_detail.csv")


if __name__ == "__main__":
    main()
