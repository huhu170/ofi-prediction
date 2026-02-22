# -*- coding: utf-8 -*-
"""
从本地parquet文件计算标准夏普比率（绕过PostgreSQL）
复用 backtest_single.py 的回测逻辑，仅替换数据源
"""
import sys, io, gc, pickle, importlib.util
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
TABLES_DIR = PROJECT_ROOT / 'outputs' / 'ch4' / 'tables'

N_SAMPLES = 40000
SEQ_LEN = 60
BARS_PER_DAY = 390 // 5  # 78 bars per day (5-min decision interval)
STOP_LOSS = 0.01
TAKE_PROFIT = 0.02
COST_RATE = 0.0
CONFIRM_BARS = 1
PROB_THRESHOLD = 0.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

STOCK_LIST = [
    ('HK.00700', 'Tencent'), ('HK.00005', 'HSBC'), ('HK.09988', 'Alibaba'),
    ('HK.01810', 'Xiaomi'), ('HK.00939', 'CCB'), ('HK.01299', 'AIA'),
    ('HK.00941', 'ChinaMobile'), ('HK.03690', 'Meituan'),
    ('HK.01211', 'BYD'), ('HK.00388', 'HKEX'),
]

_trainer_module = None
def _get_trainer():
    global _trainer_module
    if _trainer_module is None:
        spec = importlib.util.spec_from_file_location(
            "trainer", PROJECT_ROOT / "scripts" / "08_model_trainer.py")
        _trainer_module = importlib.util.module_from_spec(spec)
        _old = sys.stdout, sys.stderr
        try: spec.loader.exec_module(_trainer_module)
        except: pass
        finally: sys.stdout, sys.stderr = _old
        import __main__
        __main__.SklearnModelWrapper = _trainer_module.SklearnModelWrapper
    return _trainer_module


def fetch_data_from_parquet(code: str, n_samples: int = N_SAMPLES):
    """从本地parquet读取K线数据（替代DB查询）"""
    code_dir = code.replace('.', '_')
    parquet_path = DATA_DIR / code_dir / 'kline_cleaned_1M.parquet'
    if not parquet_path.exists():
        print(f"  [WARN] {parquet_path} not found")
        return None, None, None
    df = pd.read_parquet(parquet_path)
    if 'ts' in df.columns:
        df = df.sort_values('ts').reset_index(drop=True)
    elif 'time_key' in df.columns:
        df = df.rename(columns={'time_key': 'ts'}).sort_values('ts').reset_index(drop=True)
    else:
        df = df.sort_index().reset_index(drop=True)

    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ('open_price', 'open'): col_map[c] = 'open'
        elif cl in ('high_price', 'high'): col_map[c] = 'high'
        elif cl in ('low_price', 'low'): col_map[c] = 'low'
        elif cl in ('close_price', 'close'): col_map[c] = 'close'
        elif cl in ('volume', 'turnover_vol'): col_map[c] = 'volume'
    df = df.rename(columns=col_map)

    df = df.tail(n_samples).reset_index(drop=True)
    start_date = df.iloc[0].get('ts', df.index[0])
    end_date = df.iloc[-1].get('ts', df.index[-1])
    return df, start_date, end_date


def compute_features(df):
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


def prepare_sequences(df, seq_len=SEQ_LEN):
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
    """与 backtest_single.py 完全一致的回测逻辑"""
    n = len(predictions)
    equity = 1_000_000
    position = 0; entry_price = 0
    equity_history = [equity]
    long_trades = long_wins = short_trades = short_wins = 0
    current_direction = 0
    long_confirm = short_confirm = 0

    for i in range(0, n, 5):
        price = prices_seq[i, 0]
        if price <= 0: continue
        pred = predictions[i]
        prob = probs[i] if probs is not None else 1.0

        if position != 0 and entry_price > 0:
            pnl_pct = position * (price - entry_price) / entry_price
            if pnl_pct <= -STOP_LOSS or pnl_pct >= TAKE_PROFIT:
                equity *= (1 + pnl_pct - COST_RATE)
                if current_direction == 1:
                    long_trades += 1
                    if pnl_pct > 0: long_wins += 1
                else:
                    short_trades += 1
                    if pnl_pct > 0: short_wins += 1
                position = 0; entry_price = 0; current_direction = 0
                long_confirm = short_confirm = 0
                equity_history.append(equity)
                continue

        if pred == 0 and position != 0:
            pnl_pct = position * (price - entry_price) / entry_price
            if pnl_pct < -STOP_LOSS: pnl_pct = -STOP_LOSS
            equity *= (1 + pnl_pct - COST_RATE)
            if current_direction == 1:
                long_trades += 1
                if pnl_pct > 0: long_wins += 1
            else:
                short_trades += 1
                if pnl_pct > 0: short_wins += 1
            position = 0; entry_price = 0; current_direction = 0
            long_confirm = short_confirm = 0
            equity_history.append(equity)
            continue

        if pred == 2: long_confirm += 1; short_confirm = 0
        elif pred == 1: short_confirm += 1; long_confirm = 0

        if position == 0:
            if long_confirm >= CONFIRM_BARS and prob >= PROB_THRESHOLD:
                position = 1; current_direction = 1; entry_price = price
                equity *= (1 - COST_RATE); long_confirm = 0
            elif short_confirm >= CONFIRM_BARS and prob >= PROB_THRESHOLD:
                position = -1; current_direction = -1; entry_price = price
                equity *= (1 - COST_RATE); short_confirm = 0
        elif position == 1 and short_confirm >= CONFIRM_BARS and prob >= PROB_THRESHOLD:
            pnl_pct = (price - entry_price) / entry_price
            equity *= (1 + pnl_pct - COST_RATE)
            long_trades += 1
            if pnl_pct > 0: long_wins += 1
            position = -1; current_direction = -1; entry_price = price
            equity *= (1 - COST_RATE); short_confirm = 0
        elif position == -1 and long_confirm >= CONFIRM_BARS and prob >= PROB_THRESHOLD:
            pnl_pct = -(price - entry_price) / entry_price
            equity *= (1 + pnl_pct - COST_RATE)
            short_trades += 1
            if pnl_pct > 0: short_wins += 1
            position = 1; current_direction = 1; entry_price = price
            equity *= (1 - COST_RATE); long_confirm = 0

        equity_history.append(equity)

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

    total_return = (equity / 1_000_000 - 1) * 100
    equity_arr = np.array(equity_history)
    peak = np.maximum.accumulate(equity_arr)
    max_dd = np.max((peak - equity_arr) / peak) * 100

    return {
        'total_return': total_return, 'max_drawdown': max_dd,
        'long_trades': long_trades, 'long_wins': long_wins,
        'short_trades': short_trades, 'short_wins': short_wins,
        'total_trades': long_trades + short_trades,
        'total_wins': long_wins + short_wins,
        'equity_history': equity_arr.tolist(),
    }


def compute_sharpe(equity_history, bars_per_day=BARS_PER_DAY):
    """标准夏普比率: mean(daily_returns) / std(daily_returns) * sqrt(252)"""
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


def load_model(model_name, code):
    code_dir = code.replace('.', '_')
    trainer = _get_trainer()
    if model_name in ('logistic_regression', 'random_forest', 'xgboost'):
        model_path = MODELS_DIR / model_name / f'model_{code_dir}_1M.pkl'
        if not model_path.exists(): return None
        with open(model_path, 'rb') as f: return pickle.load(f)
    else:
        model_path = MODELS_DIR / model_name / f'model_{code_dir}_1M.pt'
        if not model_path.exists(): return None
        input_dim = 22
        seq_len = SEQ_LEN
        model = trainer.create_model(model_name, input_dim, seq_len)
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        state = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        return model


PRICE_IDX = list(range(0, 2)) + list(range(2, 7)) + list(range(7, 9)) + list(range(16, 21))
VOLUME_IDX = list(range(9, 13)) + list(range(13, 16)) + [21]

def predict(model, X, model_name, batch_size=512):
    if model_name in ('logistic_regression', 'random_forest', 'xgboost'):
        X_flat = X.reshape(X.shape[0], -1)
        preds = model.predict(X_flat)
        try:
            probs_raw = model.predict_proba(X_flat)
            probs = np.max(probs_raw, axis=1)
        except:
            probs = np.ones(len(preds))
        return preds, probs
    else:
        all_preds, all_probs = [], []
        needs_split = model_name in ('pv_transformer',)
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                batch = torch.FloatTensor(X[start:start+batch_size]).to(DEVICE)
                if needs_split:
                    price_batch = batch[:, :, PRICE_IDX]
                    vol_batch = batch[:, :, VOLUME_IDX]
                    out = model(price_batch, vol_batch)
                else:
                    out = model(batch)
                prob = torch.softmax(out, dim=1)
                pred = torch.argmax(prob, dim=1)
                all_preds.append(pred.cpu().numpy())
                all_probs.append(torch.max(prob, dim=1).values.cpu().numpy())
        return np.concatenate(all_preds), np.concatenate(all_probs)


MODEL_LIST = ['cnn_lstm', 'lstm', 'gru', 'transformer', 'pv_transformer',
              'random_forest', 'logistic_regression', 'xgboost']

def main():
    print("=" * 70)
    print("  Standard Sharpe Ratio Computation (from parquet, no DB)")
    print("=" * 70)

    all_results = []

    for model_name in MODEL_LIST:
        print(f"\n--- {model_name} ---")
        stock_sharpes = []
        stock_returns = []

        for code, name in STOCK_LIST:
            df, sd, ed = fetch_data_from_parquet(code)
            if df is None:
                print(f"  [{code}] SKIP (no data)")
                continue
            df = compute_features(df)
            X, prices = prepare_sequences(df)
            if len(X) == 0:
                print(f"  [{code}] SKIP (no sequences)")
                continue

            model = load_model(model_name, code)
            if model is None:
                print(f"  [{code}] SKIP (no model)")
                continue

            preds, probs = predict(model, X, model_name)
            result = run_backtest(preds, probs, prices)
            sharpe = compute_sharpe(result['equity_history'])
            stock_sharpes.append(sharpe)
            stock_returns.append(result['total_return'])
            print(f"  [{code}] {name}: Return={result['total_return']:.2f}%, Sharpe={sharpe:.2f}")

            del model, X, prices, preds, probs
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        if stock_sharpes:
            avg_sharpe = np.mean(stock_sharpes)
            avg_ret = np.mean(stock_returns)
            print(f"  >> {model_name} AVG: Return={avg_ret:.2f}%, Sharpe={avg_sharpe:.2f} (n={len(stock_sharpes)})")
            all_results.append({
                'model': model_name,
                'avg_return_pct': round(avg_ret, 2),
                'avg_sharpe_standard': round(avg_sharpe, 2),
                'n_stocks': len(stock_sharpes),
            })

    # Buy & Hold
    print(f"\n--- buyhold ---")
    bh_sharpes = []
    bh_returns = []
    for code, name in STOCK_LIST:
        df, _, _ = fetch_data_from_parquet(code)
        if df is None: continue
        prices = df['close'].values
        bh_equity = 1_000_000 * prices / prices[0]
        bh_return = (prices[-1] / prices[0] - 1) * 100
        bh_sharpe = compute_sharpe(bh_equity.tolist())
        bh_sharpes.append(bh_sharpe)
        bh_returns.append(bh_return)
        print(f"  [{code}] {name}: Return={bh_return:.2f}%, Sharpe={bh_sharpe:.2f}")
    if bh_sharpes:
        avg = np.mean(bh_sharpes)
        avg_r = np.mean(bh_returns)
        print(f"  >> buyhold AVG: Return={avg_r:.2f}%, Sharpe={avg:.2f}")
        all_results.append({'model': 'buyhold', 'avg_return_pct': round(avg_r, 2),
                            'avg_sharpe_standard': round(avg, 2), 'n_stocks': len(bh_sharpes)})

    if all_results:
        out = pd.DataFrame(all_results)
        outfile = TABLES_DIR / 'sharpe_standard.csv'
        out.to_csv(outfile, index=False, encoding='utf-8-sig')
        print(f"\n\nSaved: {outfile}")
        print("\n" + out.to_string(index=False))


if __name__ == "__main__":
    main()
