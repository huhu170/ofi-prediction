# -*- coding: utf-8 -*-
"""
从回测汇总数据计算各模型的年化夏普比率
方法A（交易模型）: 组合级 MDD-vol 估算, σ_daily ≈ avg_MDD / sqrt(πT/2), Sharpe = r_daily/σ * sqrt(252)
方法B（Buy & Hold）: 从个股日收益率的截面分布估算
"""
import numpy as np
import pandas as pd
from pathlib import Path

TABLES = Path(r"d:\paper project\outputs\ch4\tables")
T = 102  # trading days
MDD_VOL_DENOM = np.sqrt(np.pi * T / 2)  # ~12.67

models = [
    ("cnn_lstm",              "CNN-LSTM",             "DL"),
    ("lstm",                  "LSTM",                 "DL"),
    ("gru",                   "GRU",                  "DL"),
    ("transformer",           "Transformer",          "DL"),
    ("pv_transformer",        "PV-Transformer",       "DL"),
    ("multi_scale",           "Multi-Scale PVT",      "DL"),
    ("random_forest",         "Random Forest",        "ML"),
    ("logistic_regression",   "Logistic Regression",  "ML"),
    ("xgboost",               "XGBoost",              "ML"),
    ("arima",                 "ARIMA",                "Stat"),
    ("buyhold",               "Buy & Hold",           "Base"),
]

results = []
for fname, display, mtype in models:
    detail = TABLES / f"backtest_{fname}_detail.csv"
    if not detail.exists():
        print(f"  [SKIP] {display}")
        continue
    df = pd.read_csv(detail)
    if df.empty or 'return_pct' not in df.columns:
        continue

    avg_ret = df['return_pct'].mean() / 100.0
    avg_mdd = df['max_dd_pct'].mean() / 100.0

    # 组合日均收益
    port_daily_ret = (1 + avg_ret) ** (1.0 / T) - 1

    if fname == 'buyhold' or avg_mdd == 0:
        # B&H没有MDD数据，用截面波动率（不除sqrt(N)，因为是被动基准）
        stock_daily_rets = [(1 + r / 100.0) ** (1.0 / T) - 1 for r in df['return_pct']]
        port_daily_vol = np.std(stock_daily_rets, ddof=1)
    else:
        # MDD-vol relationship: σ_daily ≈ MDD / sqrt(πT/2)
        port_daily_vol = avg_mdd / MDD_VOL_DENOM

    if port_daily_vol > 0:
        sharpe = (port_daily_ret / port_daily_vol) * np.sqrt(252)
    else:
        sharpe = 0.0

    ann_ret = ((1 + avg_ret) ** (252 / T) - 1) * 100

    results.append({
        'model': display,
        'type': mtype,
        'avg_return_pct': round(avg_ret * 100, 2),
        'annualized_return_pct': round(ann_ret, 1),
        'avg_max_dd_pct': round(avg_mdd * 100, 2),
        'sharpe': round(sharpe, 2),
    })
    print(f"  {display:25s}  Sharpe={sharpe:+.2f}  ann_ret={ann_ret:+.1f}%  avg_mdd={avg_mdd*100:.1f}%")

print(f"\n{'='*70}")
print(f"  Portfolio Sharpe Summary (sorted)")
print(f"{'='*70}")
for r in sorted(results, key=lambda x: -x['sharpe']):
    print(f"  {r['model']:25s} [{r['type']:4s}]  Sharpe={r['sharpe']:+.2f}  ann={r['annualized_return_pct']:+.1f}%  mdd={r['avg_max_dd_pct']:.1f}%")

# Save
out = pd.DataFrame(results)
out_path = TABLES / 'sharpe_ratios.csv'
out.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"\nSaved to {out_path}")
