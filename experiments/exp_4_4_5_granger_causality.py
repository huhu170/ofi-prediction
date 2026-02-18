# -*- coding: utf-8 -*-
"""
实验 4.4.5: Granger因果检验（真实数据版）

对22维特征逐一进行Granger因果检验，检验其对未来收益率的预测性因果关系。
使用 statsmodels.tsa.stattools.grangercausalitytests，滞后阶数=5。

对应论文:
- 表 4.4-2: Granger因果检验结果

输出:
- table_4_4_2_granger_causality.csv
"""

import sys, io
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

from exp_config import *
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import grangercausalitytests, adfuller


FEATURE_COLS = [
    'kline_position', 'range_pct', 'return_1', 'return_5', 'return_20',
    'return_60', 'return_zscore', 'atr_pct', 'volatility_20', 'ti',
    'ti_5', 'ti_60', 'ti_zscore', 'relative_volume', 'volume_change',
    'pv_corr', 'rsi', 'bb_position', 'macd_dif', 'macd_dea', 'macd',
    'market_regime',
]

HORIZONS = [5, 15, 30]
MAX_LAG = 5
N_SAMPLES = 40000


def get_db_connection():
    import psycopg2
    return psycopg2.connect(
        host="127.0.0.1", port=5433,
        database="futu_ofi", user="postgres", password="ofi123456"
    )


def fetch_and_compute(code: str) -> pd.DataFrame:
    """获取K线数据并计算全部22维特征 + 未来收益率"""
    conn = get_db_connection()
    query = f"""
    SELECT ts, open_price as open, high_price as high,
           low_price as low, close_price as close, volume
    FROM kline WHERE code = '{code}' AND ktype = 'K_1M'
    ORDER BY ts DESC LIMIT {N_SAMPLES}
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty or len(df) < 200:
        return pd.DataFrame()

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

    for h in HORIZONS:
        df[f'future_ret_{h}'] = df['close'].pct_change(h).shift(-h)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def granger_test_single(y: np.ndarray, x: np.ndarray, maxlag: int = MAX_LAG):
    """对单个 (y, x) 对执行 Granger 因果检验，返回最优滞后阶的 F 统计量和 p 值"""
    data = np.column_stack([y, x])

    if np.std(data[:, 0]) < 1e-10 or np.std(data[:, 1]) < 1e-10:
        return np.nan, 1.0

    try:
        result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
        best_f, best_p = 0.0, 1.0
        for lag in range(1, maxlag + 1):
            f_val = result[lag][0]['ssr_ftest'][0]
            p_val = result[lag][0]['ssr_ftest'][1]
            if p_val < best_p:
                best_f, best_p = f_val, p_val
        return best_f, best_p
    except Exception:
        return np.nan, 1.0


def run_experiment():
    log_experiment('4.4.5', '开始Granger因果检验（真实数据）')

    all_results = []

    for code, name, sector in STOCK_LIST:
        log_experiment('4.4.5', f'处理 {code} ({name})')
        df = fetch_and_compute(code)
        if df.empty:
            log_experiment('4.4.5', f'  {code} 数据不足，跳过')
            continue

        log_experiment('4.4.5', f'  数据行数: {len(df)}')

        for feat in FEATURE_COLS:
            if feat == 'market_regime':
                continue

            x_series = df[feat].values
            for h in HORIZONS:
                y_series = df[f'future_ret_{h}'].values
                f_val, p_val = granger_test_single(y_series, x_series, MAX_LAG)
                all_results.append({
                    'stock': code,
                    'feature': feat,
                    'feature_cn': FEATURE_NAMES_CN.get(feat, feat),
                    'horizon_min': h,
                    'F_statistic': round(f_val, 4) if not np.isnan(f_val) else np.nan,
                    'p_value': round(p_val, 6) if not np.isnan(p_val) else np.nan,
                    'significant_005': p_val < 0.05,
                    'significant_001': p_val < 0.01,
                })

    df_all = pd.DataFrame(all_results)

    summary = (
        df_all.groupby(['feature', 'feature_cn', 'horizon_min'])
        .agg(
            mean_F=('F_statistic', 'mean'),
            mean_p=('p_value', 'mean'),
            n_sig_005=('significant_005', 'sum'),
            n_sig_001=('significant_001', 'sum'),
            n_stocks=('stock', 'count'),
        )
        .reset_index()
    )
    summary['sig_rate'] = summary['n_sig_005'] / summary['n_stocks']

    detail_path = get_output_path('table_4_4_2_granger_detail', 'csv')
    df_all.to_csv(detail_path, index=False, encoding='utf-8-sig')
    log_experiment('4.4.5', f'明细表已保存: {detail_path}')

    summary_path = get_output_path('table_4_4_2_granger_causality', 'csv')
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    log_experiment('4.4.5', f'汇总表已保存: {summary_path}')

    pivot_5 = summary[summary['horizon_min'] == 5].sort_values('mean_F', ascending=False)
    pivot_30 = summary[summary['horizon_min'] == 30].sort_values('mean_F', ascending=False)

    print("\n" + "=" * 70)
    print("  Granger因果检验结果（5min预测步长，10只股票均值）")
    print("=" * 70)
    for _, row in pivot_5.head(15).iterrows():
        stars = '***' if row['mean_p'] < 0.001 else ('**' if row['mean_p'] < 0.01 else ('*' if row['mean_p'] < 0.05 else ''))
        sig_note = f"显著({int(row['n_sig_005'])}/{int(row['n_stocks'])}只)" if row['sig_rate'] >= 0.5 else f"不显著({int(row['n_sig_005'])}/{int(row['n_stocks'])}只)"
        print(f"  {row['feature_cn']:15s}  F={row['mean_F']:8.3f}  p={row['mean_p']:.5f}{stars:4s}  {sig_note}")

    sig_features = pivot_5[pivot_5['sig_rate'] >= 0.5]['feature'].tolist()
    print(f"\n  通过因果检验的特征 (>=50%股票在5min步长上显著): {len(sig_features)}")
    for f in sig_features:
        cn = FEATURE_NAMES_CN.get(f, f)
        print(f"    - {f} ({cn})")

    return df_all, summary


if __name__ == "__main__":
    set_seed()
    run_experiment()
