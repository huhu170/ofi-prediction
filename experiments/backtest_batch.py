# -*- coding: utf-8 -*-
"""
批量回测脚本 - 逐个调用子进程避免内存问题
用法: python backtest_batch.py lstm
"""

import subprocess
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TABLES_DIR = PROJECT_ROOT / 'outputs' / 'ch4' / 'tables'

STOCK_LIST = [
    ('HK.00700', '腾讯'),
    ('HK.00005', '汇丰'),
    ('HK.09988', '阿里'),
    ('HK.01810', '小米'),
    ('HK.00939', '建行'),
    ('HK.01299', '友邦'),
    ('HK.00941', '中移动'),
    ('HK.03690', '美团'),
    ('HK.01211', '比亚迪'),
    ('HK.00388', '港交所'),
]

MODEL_LIST = [
    'buyhold', 'lstm', 'gru', 'cnn_lstm', 'transformer',
    'pv_transformer', 'multi_scale',
    'logistic_regression', 'random_forest', 'xgboost'
]

def run_single(model_name, code):
    """运行单只股票回测"""
    cmd = [sys.executable, 'experiments/backtest_single.py', model_name, code]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=str(PROJECT_ROOT))
        # 解析输出获取结果
        output = result.stdout
        for line in output.split('\n'):
            if 'Return:' in line:
                # 解析: Return: -3.85%, DD: 4.12%, WinRate: 40.0%, Trades: 40
                parts = line.strip().split(',')
                ret = float(parts[0].split(':')[1].strip().replace('%', ''))
                dd = float(parts[1].split(':')[1].strip().replace('%', ''))
                wr = float(parts[2].split(':')[1].strip().replace('%', ''))
                trades = int(parts[3].split(':')[1].strip())
                return {'return': ret, 'dd': dd, 'winrate': wr, 'trades': trades}
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python backtest_batch.py <model_name|all>")
        print(f"Models: {', '.join(MODEL_LIST)}")
        return
    
    model_arg = sys.argv[1].lower()
    
    if model_arg == 'all':
        models_to_run = MODEL_LIST
    else:
        models_to_run = [model_arg]
    
    all_summaries = []
    
    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")
        
        results = []
        for code, name in STOCK_LIST:
            print(f"  [{code}] {name}...", end=' ', flush=True)
            r = run_single(model_name, code)
            if r:
                print(f"Return: {r['return']:.2f}%, Trades: {r['trades']}")
                results.append({
                    'stock': code,
                    'name': name,
                    'return_pct': r['return'],
                    'max_dd_pct': r['dd'],
                    'win_rate_pct': r['winrate'],
                    'n_trades': r['trades'],
                })
            else:
                print("SKIP")
        
        if results:
            df = pd.DataFrame(results)
            avg_ret = df['return_pct'].mean()
            avg_dd = df['max_dd_pct'].mean()
            avg_wr = df['win_rate_pct'].mean()
            avg_tr = df['n_trades'].mean()
            
            print(f"\n  Summary: Avg Return={avg_ret:.2f}%, MaxDD={avg_dd:.2f}%, WinRate={avg_wr:.1f}%, Trades={avg_tr:.1f}")
            
            # 保存明细
            df.to_csv(TABLES_DIR / f"backtest_{model_name}_detail.csv", index=False, encoding='utf-8-sig')
            
            all_summaries.append({
                'model': model_name,
                'n_stocks': len(results),
                'avg_return_pct': round(avg_ret, 2),
                'avg_max_dd_pct': round(avg_dd, 2),
                'avg_win_rate_pct': round(avg_wr, 1),
                'avg_n_trades': round(avg_tr, 1),
            })
    
    # 保存汇总
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_file = TABLES_DIR / 'backtest_all_models.csv'
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"\n\nAll results saved to: {summary_file}")
        print("\n" + "="*60)
        print("  Final Summary")
        print("="*60)
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
