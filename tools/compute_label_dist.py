"""Compute fixed vs adaptive threshold label distributions for all 10 stocks."""
import pandas as pd
import numpy as np
from pathlib import Path

stocks = [
    ('HK.00700', '腾讯控股', '科技'),
    ('HK.00005', '汇丰控股', '金融'),
    ('HK.09988', '阿里巴巴', '科技'),
    ('HK.01810', '小米集团', '科技'),
    ('HK.00939', '建设银行', '金融'),
    ('HK.01299', '友邦保险', '金融'),
    ('HK.00941', '中国移动', '通信'),
    ('HK.03690', '美团', '科技'),
    ('HK.01211', '比亚迪', '汽车'),
    ('HK.00388', '香港交易所', '金融'),
]

fixed_alpha = 0.002

for code, name, sector in stocks:
    code_dir = Path('data/processed') / code.replace('.', '_')
    fpath = code_dir / 'kline_features_1M.parquet'
    if not fpath.exists():
        print(f'MISSING: {fpath}')
        continue
    df = pd.read_parquet(fpath)
    ret = df['return_1'].dropna().values

    n = len(ret)
    down_f = (ret < -fixed_alpha).sum() / n * 100
    up_f = (ret > fixed_alpha).sum() / n * 100
    mid_f = 100 - down_f - up_f

    abs_ret = np.abs(ret)
    adaptive_alpha = np.percentile(abs_ret, 33)
    down_a = (ret < -adaptive_alpha).sum() / n * 100
    up_a = (ret > adaptive_alpha).sum() / n * 100
    mid_a = 100 - down_a - up_a

    print(f"{code}|{name}|{sector}|{ret.std()*100:.4f}|"
          f"{down_f:.1f}|{mid_f:.1f}|{up_f:.1f}|"
          f"{adaptive_alpha:.5f}|{down_a:.1f}|{mid_a:.1f}|{up_a:.1f}")
