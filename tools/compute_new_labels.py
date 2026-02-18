"""计算自适应alpha下各K线类型的标签分布（用于更新论文表格）"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import os

DATA_DIR = 'data/processed'
stocks = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('HK_')])

PERCENTILE = 33

for ktype in ['1M', '5M', '60M', 'DAY']:
    print(f"\n=== {ktype} (percentile={PERCENTILE}) ===")
    all_down, all_neutral, all_up, all_n = 0, 0, 0, 0
    alphas = []
    
    for stock in stocks:
        path = f'{DATA_DIR}/{stock}/kline_features_{ktype}.parquet'
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        if 'future_return_5' not in df.columns:
            continue
        
        fr = df['future_return_5'].dropna().values
        alpha = np.percentile(np.abs(fr), PERCENTILE)
        alphas.append(alpha)
        
        n = len(fr)
        n_up = (fr > alpha).sum()
        n_down = (fr < -alpha).sum()
        n_neutral = n - n_up - n_down
        
        all_down += n_down
        all_neutral += n_neutral
        all_up += n_up
        all_n += n
    
    avg_alpha = np.mean(alphas)
    print(f"  avg alpha = {avg_alpha:.6f}")
    print(f"  down  = {all_down/all_n*100:.1f}%")
    print(f"  neutral = {all_neutral/all_n*100:.1f}%")
    print(f"  up    = {all_up/all_n*100:.1f}%")

# 也输出旧alpha=0.002的对比
print("\n\n=== 旧 alpha=0.002 对比 ===")
for ktype in ['1M', '5M', '60M', 'DAY']:
    all_down, all_neutral, all_up, all_n = 0, 0, 0, 0
    for stock in stocks:
        path = f'{DATA_DIR}/{stock}/kline_features_{ktype}.parquet'
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        if 'label_5' not in df.columns:
            continue
        labels = df['label_5'].values
        n = len(labels)
        all_down += (labels == -1).sum()
        all_neutral += (labels == 0).sum()
        all_up += (labels == 1).sum()
        all_n += n
    print(f"  {ktype}: down={all_down/all_n*100:.1f}%, neutral={all_neutral/all_n*100:.1f}%, up={all_up/all_n*100:.1f}%")
