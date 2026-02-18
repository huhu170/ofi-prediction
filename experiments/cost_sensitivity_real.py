# -*- coding: utf-8 -*-
"""
真实交易成本敏感性分析
对 PV-Transformer / CNN-LSTM / Transformer 在不同成本下的回测表现
复用 backtest_single.py 的核心逻辑，只改 COST_RATE
"""

import sys, gc, io
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

import numpy as np
import pandas as pd
import torch

import backtest_single as bs

COST_LEVELS = [
    (0.0, '0%(基准)'),
    (0.0003, '0.03%'),
    (0.0005, '0.05%'),
    (0.001, '0.10%'),
    (0.0015, '0.15%'),
]

MODELS_TO_TEST = ['pv_transformer', 'cnn_lstm', 'transformer', 'lstm']

MODEL_MAP = {
    'lstm': ('lstm', 'lstm'),
    'cnn_lstm': ('cnn_lstm', 'cnn_lstm'),
    'transformer': ('transformer', 'transformer'),
    'pv_transformer': ('pv_transformer', 'pv_transformer'),
}


def run_backtest_with_cost(predictions, probs, prices_seq, cost_rate):
    """用指定成本运行回测（复用 bs 的回测逻辑但接受 cost 参数）"""
    n = len(predictions)
    equity = 1_000_000
    position = 0
    entry_price = 0
    equity_history = [equity]
    long_trades = short_trades = long_wins = short_wins = 0
    current_direction = 0
    long_confirm = short_confirm = 0

    for i in range(0, n, bs.DECISION_INTERVAL):
        pred = predictions[i]
        prob = probs[i] if probs is not None else 1.0
        price = prices_seq[i, 0]

        if position != 0 and entry_price > 0:
            pnl_pct = position * (price - entry_price) / entry_price
            if pnl_pct <= -bs.STOP_LOSS or pnl_pct >= bs.TAKE_PROFIT:
                equity *= (1 + pnl_pct - cost_rate)
                if current_direction == 1:
                    long_trades += 1
                    if pnl_pct > 0: long_wins += 1
                else:
                    short_trades += 1
                    if pnl_pct > 0: short_wins += 1
                position = entry_price = current_direction = 0
                long_confirm = short_confirm = 0
                equity_history.append(equity)
                continue

        if bs.NEUTRAL_CLOSE and pred == 1 and position != 0:
            pnl_pct = position * (price - entry_price) / entry_price
            if pnl_pct < -bs.STOP_LOSS:
                pnl_pct = -bs.STOP_LOSS
            equity *= (1 + pnl_pct - cost_rate)
            if current_direction == 1:
                long_trades += 1
                if pnl_pct > 0: long_wins += 1
            else:
                short_trades += 1
                if pnl_pct > 0: short_wins += 1
            position = entry_price = current_direction = 0
            long_confirm = short_confirm = 0
            equity_history.append(equity)
            continue

        if pred == 2:
            long_confirm += 1; short_confirm = 0
        elif pred == 0:
            short_confirm += 1; long_confirm = 0
        else:
            long_confirm = short_confirm = 0

        if position == 0:
            if long_confirm >= bs.CONFIRM_BARS and prob >= bs.PROB_THRESHOLD:
                position = 1; current_direction = 1; entry_price = price
                equity *= (1 - cost_rate); long_confirm = 0
            elif short_confirm >= bs.CONFIRM_BARS and prob >= bs.PROB_THRESHOLD:
                position = -1; current_direction = -1; entry_price = price
                equity *= (1 - cost_rate); short_confirm = 0
        elif position == 1 and short_confirm >= bs.CONFIRM_BARS and prob >= bs.PROB_THRESHOLD:
            pnl_pct = (price - entry_price) / entry_price
            equity *= (1 + pnl_pct - cost_rate)
            long_trades += 1
            if pnl_pct > 0: long_wins += 1
            position = -1; current_direction = -1; entry_price = price
            equity *= (1 - cost_rate); short_confirm = 0
        elif position == -1 and long_confirm >= bs.CONFIRM_BARS and prob >= bs.PROB_THRESHOLD:
            pnl_pct = -(price - entry_price) / entry_price
            equity *= (1 + pnl_pct - cost_rate)
            short_trades += 1
            if pnl_pct > 0: short_wins += 1
            position = 1; current_direction = 1; entry_price = price
            equity *= (1 - cost_rate); long_confirm = 0

        equity_history.append(equity)

    if position != 0 and entry_price > 0:
        final_price = prices_seq[-1, 0]
        pnl_pct = position * (final_price - entry_price) / entry_price
        equity *= (1 + pnl_pct - cost_rate)
        if current_direction == 1:
            long_trades += 1
            if pnl_pct > 0: long_wins += 1
        else:
            short_trades += 1
            if pnl_pct > 0: short_wins += 1

    total_return = (equity / 1_000_000 - 1) * 100
    total_trades = long_trades + short_trades
    total_wins = long_wins + short_wins
    equity_arr = np.array(equity_history)
    peak = np.maximum.accumulate(equity_arr)
    max_dd = np.max((peak - equity_arr) / peak) * 100

    return {
        'total_return': total_return,
        'max_drawdown': max_dd,
        'total_trades': total_trades,
        'total_winrate': total_wins / max(total_trades, 1) * 100,
    }


def main():
    TABLES_DIR = Path(__file__).parent.parent / 'outputs' / 'ch4' / 'tables'
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_name in MODELS_TO_TEST:
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        dir_name, model_type = MODEL_MAP[model_name]

        for code, name in bs.STOCK_LIST:
            code_str = code.replace('.', '_')
            ext = 'pkl' if model_type == 'sklearn' else 'pt'
            model_path = bs.MODELS_DIR / dir_name / f"model_{code_str}_1M.{ext}"

            if not model_path.exists():
                print(f"  [{code}] SKIP - model not found")
                continue

            print(f"  [{code}] {name}...", end=' ', flush=True)

            try:
                df, start_date, end_date = bs.fetch_data(code, n_samples=bs.N_SAMPLES)
                if df is None:
                    print("NO DATA"); continue
                df = bs.compute_features(df)
                X, prices = bs.prepare_sequences(df)
                del df; gc.collect()
                if len(X) < 5000:
                    print("TOO FEW"); continue

                model, device = bs.load_model(model_path, model_type)
                preds, probs = bs.predict_batch(model, X, model_type, device)
                del model; gc.collect()
                if bs.DEVICE == 'cuda':
                    torch.cuda.empty_cache()

                for cost_rate, cost_label in COST_LEVELS:
                    result = run_backtest_with_cost(preds, probs, prices, cost_rate)
                    all_results.append({
                        'model': model_name,
                        'stock': code,
                        'cost_label': cost_label,
                        'cost_rate': cost_rate,
                        'return_pct': round(result['total_return'], 2),
                        'max_dd_pct': round(result['max_drawdown'], 2),
                        'total_trades': result['total_trades'],
                        'total_winrate': round(result['total_winrate'], 1),
                    })

                del X, prices, preds, probs; gc.collect()
                print("OK")

            except Exception as e:
                print(f"ERROR: {e}")

    if all_results:
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(TABLES_DIR / 'cost_sensitivity_real_detail.csv',
                       index=False, encoding='utf-8-sig')

        summary = df_all.groupby(['model', 'cost_label', 'cost_rate']).agg(
            avg_return=('return_pct', 'mean'),
            avg_dd=('max_dd_pct', 'mean'),
            avg_trades=('total_trades', 'mean'),
            avg_winrate=('total_winrate', 'mean'),
            n_stocks=('stock', 'count'),
        ).reset_index()

        summary.to_csv(TABLES_DIR / 'cost_sensitivity_real_summary.csv',
                       index=False, encoding='utf-8-sig')

        print("\n\n" + "="*70)
        print("  COST SENSITIVITY SUMMARY")
        print("="*70)
        for model in MODELS_TO_TEST:
            m = summary[summary['model'] == model]
            print(f"\n  {model}:")
            for _, row in m.iterrows():
                print(f"    {row['cost_label']:>8s}: return={row['avg_return']:+.2f}%, "
                      f"DD={row['avg_dd']:.1f}%, trades={row['avg_trades']:.0f}, "
                      f"winrate={row['avg_winrate']:.1f}%")

        print(f"\nSaved to: {TABLES_DIR / 'cost_sensitivity_real_summary.csv'}")


if __name__ == '__main__':
    main()
