# -*- coding: utf-8 -*-
"""
实验 4.4.2: 市场状态异质性检验（真实数据版）

按波动率分位数将测试数据分为平稳期/正常期/高波动期，
分别评估各模型在不同市场状态下的 Accuracy / F1-macro。

对应论文:
- 表 4.4-3: 市场状态异质性检验

输出:
- table_4_4_3_regime_heterogeneity.csv
"""

import sys, io, gc, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from exp_config import *
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

import torch
from sklearn.metrics import accuracy_score, f1_score
import importlib.util

_trainer_module = None

def _get_trainer():
    global _trainer_module
    if _trainer_module is None:
        spec = importlib.util.spec_from_file_location(
            "kline_model_trainer",
            PROJECT_ROOT / "scripts" / "13b_kline_model_trainer.py"
        )
        _trainer_module = importlib.util.module_from_spec(spec)
        devnull = open(os.devnull, 'w')
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            spec.loader.exec_module(_trainer_module)
        except Exception:
            pass
        finally:
            devnull.close()
            sys.stdout = io.TextIOWrapper(
                open(sys.__stdout__.fileno(), 'wb', closefd=False),
                encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(
                open(sys.__stderr__.fileno(), 'wb', closefd=False),
                encoding='utf-8', errors='replace')
        import __main__
        __main__.SklearnModelWrapper = _trainer_module.SklearnModelWrapper
    return _trainer_module

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODELS_TO_TEST = ['lstm', 'cnn_lstm', 'transformer', 'pv_transformer', 'xgboost']

MODEL_DISPLAY = {
    'lstm': 'LSTM',
    'cnn_lstm': 'CNN-LSTM',
    'transformer': 'Transformer',
    'pv_transformer': 'PV-Transformer',
    'xgboost': 'XGBoost',
}

N_SAMPLES = 40000
SEQ_LEN = 60
VOL_WINDOW = 20

FEATURE_COLS = [
    'kline_position', 'range_pct', 'return_1', 'return_5', 'return_20',
    'return_60', 'return_zscore', 'atr_pct', 'volatility_20', 'ti',
    'ti_5', 'ti_60', 'ti_zscore', 'relative_volume', 'volume_change',
    'pv_corr', 'rsi', 'bb_position', 'macd_dif', 'macd_dea', 'macd',
    'market_regime',
]


def get_db_connection():
    import psycopg2
    return psycopg2.connect(
        host="127.0.0.1", port=5433,
        database="futu_ofi", user="postgres", password="ofi123456"
    )


def fetch_and_prepare(code: str):
    """获取数据、计算特征、生成序列、标注波动率分组"""
    conn = get_db_connection()
    query = f"""
    SELECT ts, open_price as open, high_price as high,
           low_price as low, close_price as close, volume
    FROM kline WHERE code = '{code}' AND ktype = 'K_1M'
    ORDER BY ts DESC LIMIT {N_SAMPLES}
    """
    df = pd.read_sql(query, conn)
    conn.close()
    if df.empty or len(df) < 500:
        return None, None, None, None

    df = df.sort_values('ts').reset_index(drop=True)

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

    # adaptive alpha labeling (same logic as training)
    future_ret = df['close'].pct_change(5).shift(-5)
    rolling_vol = df['return_1'].rolling(60).std()
    alpha = (rolling_vol * 2).clip(lower=0.001, upper=0.01)
    labels = pd.Series(1, index=df.index)
    labels[future_ret > alpha] = 2
    labels[future_ret < -alpha] = 0
    df['label'] = labels

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    vol_series = df['volatility_20'].values
    p50 = np.percentile(vol_series, 50)
    p90 = np.percentile(vol_series, 90)
    regime = np.where(vol_series < p50, 0, np.where(vol_series > p90, 2, 1))

    X_raw = df[FEATURE_COLS].values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_raw = np.clip(X_raw, -1e6, 1e6)
    mean_x = X_raw.mean(axis=0)
    std_x = X_raw.std(axis=0) + 1e-8
    X_norm = (X_raw - mean_x) / std_x

    y_all = df['label'].values.astype(np.int64)

    n = len(X_norm) - SEQ_LEN - 5
    if n < 100:
        return None, None, None, None

    X_seq = np.zeros((n, SEQ_LEN, X_norm.shape[1]), dtype=np.float32)
    y_seq = np.zeros(n, dtype=np.int64)
    regime_seq = np.zeros(n, dtype=np.int64)

    for i in range(n):
        X_seq[i] = X_norm[i:i + SEQ_LEN]
        y_seq[i] = y_all[i + SEQ_LEN]
        regime_seq[i] = regime[i + SEQ_LEN]

    return X_seq, y_seq, regime_seq, df


def load_model(model_name: str, code: str):
    """加载已训练的模型"""
    code_clean = code.replace('.', '_')
    m = _get_trainer()

    if model_name in ('xgboost', 'random_forest', 'logistic_regression'):
        model_path = MODELS_DIR / model_name / f'model_{code_clean}_1M.pkl'
        if not model_path.exists():
            return None
        with open(model_path, 'rb') as f:
            wrapper = pickle.load(f)
        return ('sklearn', wrapper.model if hasattr(wrapper, 'model') else wrapper)

    model_dir = MODELS_DIR / model_name
    ckpt_path = model_dir / f'model_{code_clean}_1M.pt'
    if not ckpt_path.exists():
        return None

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    if model_name == 'lstm':
        model = m.LSTMBaseline(input_dim=22, seq_len=SEQ_LEN)
    elif model_name == 'gru':
        model = m.GRUBaseline(input_dim=22, seq_len=SEQ_LEN)
    elif model_name == 'cnn_lstm':
        model = m.CNNLSTMBaseline(input_dim=22, seq_len=SEQ_LEN)
    elif model_name == 'transformer':
        model = m.TransformerBaseline(input_dim=22, seq_len=SEQ_LEN)
    elif model_name == 'pv_transformer':
        model = m.PVTransformer(price_dim=14, volume_dim=8, seq_len=SEQ_LEN)
    else:
        return None

    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)

    model.to(DEVICE)
    model.eval()
    return ('torch', model)


def predict_batch(model_tuple, X: np.ndarray, model_name: str):
    """使用模型进行批量预测"""
    mtype, model = model_tuple

    if mtype == 'sklearn':
        X_flat = X.reshape(X.shape[0], -1)
        X_flat = np.nan_to_num(X_flat, nan=0, posinf=0, neginf=0)
        return model.predict(X_flat)

    all_preds = []
    BATCH = 512
    with torch.no_grad():
        for start in range(0, len(X), BATCH):
            batch = X[start:start + BATCH]
            X_t = torch.FloatTensor(batch).to(DEVICE)
            if model_name == 'pv_transformer':
                out = model(X_t[:, :, :14], X_t[:, :, 14:])
            else:
                out = model(X_t)
            all_preds.append(out.argmax(dim=1).cpu().numpy())
            del X_t, out
    return np.concatenate(all_preds)


def run_experiment():
    log_experiment('4.4.2', '开始市场状态异质性检验（真实数据）')

    regime_names = {0: '平稳期(<P50)', 1: '正常期(P50-P90)', 2: '高波动期(>P90)'}
    all_results = []

    for code, name, sector in STOCK_LIST:
        log_experiment('4.4.2', f'处理 {code} ({name})')
        X_seq, y_seq, regime_seq, df = fetch_and_prepare(code)
        if X_seq is None:
            log_experiment('4.4.2', f'  {code} 数据不足，跳过')
            continue

        log_experiment('4.4.2', f'  样本数: {len(y_seq)}, regime分布: '
                       f'平稳={np.sum(regime_seq==0)}, '
                       f'正常={np.sum(regime_seq==1)}, '
                       f'高波动={np.sum(regime_seq==2)}')

        for model_name in MODELS_TO_TEST:
            model_tuple = load_model(model_name, code)
            if model_tuple is None:
                log_experiment('4.4.2', f'  {model_name} 模型不存在，跳过')
                continue

            BATCH = 512
            preds = predict_batch(model_tuple, X_seq, model_name)

            for regime_id in [0, 1, 2]:
                mask = regime_seq == regime_id
                if mask.sum() < 30:
                    continue
                y_sub = y_seq[mask]
                p_sub = preds[mask]
                acc = accuracy_score(y_sub, p_sub)
                f1 = f1_score(y_sub, p_sub, average='macro', zero_division=0)
                all_results.append({
                    'stock': code,
                    'model': model_name,
                    'model_display': MODEL_DISPLAY.get(model_name, model_name),
                    'regime': regime_id,
                    'regime_name': regime_names[regime_id],
                    'n_samples': int(mask.sum()),
                    'accuracy': round(acc * 100, 2),
                    'f1_macro': round(f1 * 100, 2),
                })

            if model_tuple[0] == 'torch':
                del model_tuple
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    df_all = pd.DataFrame(all_results)

    detail_path = get_output_path('table_4_4_3_regime_detail', 'csv')
    df_all.to_csv(detail_path, index=False, encoding='utf-8-sig')
    log_experiment('4.4.2', f'明细表已保存: {detail_path}')

    if df_all.empty:
        log_experiment('4.4.2', '无有效结果')
        return df_all

    summary = (
        df_all.groupby(['model_display', 'regime', 'regime_name'])
        .agg(
            mean_acc=('accuracy', 'mean'),
            mean_f1=('f1_macro', 'mean'),
            n_stocks=('stock', 'nunique'),
        )
        .reset_index()
    )

    summary_path = get_output_path('table_4_4_3_regime_heterogeneity', 'csv')
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    log_experiment('4.4.2', f'汇总表已保存: {summary_path}')

    print("\n" + "=" * 70)
    print("  市场状态异质性检验（F1-macro %，10只股票均值）")
    print("=" * 70)
    print(f"  {'模型':<20s} {'平稳期':>8s} {'正常期':>8s} {'高波动期':>8s} {'高波动相对正常':>12s}")
    print("-" * 70)
    for model_disp in ['LSTM', 'CNN-LSTM', 'Transformer', 'PV-Transformer', 'XGBoost']:
        row_data = summary[summary['model_display'] == model_disp]
        if row_data.empty:
            continue
        calm = row_data[row_data['regime'] == 0]['mean_f1'].values
        normal = row_data[row_data['regime'] == 1]['mean_f1'].values
        high = row_data[row_data['regime'] == 2]['mean_f1'].values
        calm_v = calm[0] if len(calm) > 0 else np.nan
        normal_v = normal[0] if len(normal) > 0 else np.nan
        high_v = high[0] if len(high) > 0 else np.nan
        diff = (high_v - normal_v) if (not np.isnan(high_v) and not np.isnan(normal_v)) else np.nan
        print(f"  {model_disp:<20s} {calm_v:>7.1f}% {normal_v:>7.1f}% {high_v:>7.1f}% {diff:>+10.1f}pp")

    return df_all, summary


if __name__ == "__main__":
    set_seed()
    run_experiment()
