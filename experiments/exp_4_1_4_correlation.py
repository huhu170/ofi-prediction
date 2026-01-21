"""
实验 4.1.4: 特征与收益率相关性检验

对应论文:
- 表 4.1-4: 核心特征与同期收益率的Pearson/Spearman相关系数

输出:
- table_4_1_4_correlation.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from exp_config import *
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

def compute_correlations(df: pd.DataFrame, target: str = 'return_1') -> pd.DataFrame:
    """计算特征与目标变量的相关系数"""
    results = []
    
    features = [f for f in ALL_FEATURES if f != target and f in df.columns]
    
    for feat in features:
        data = df[[feat, target]].dropna()
        if len(data) < 100:
            continue
        
        x, y = data[feat].values, data[target].values
        
        # Pearson
        pearson_r, pearson_p = scipy_stats.pearsonr(x, y)
        
        # Spearman
        spearman_r, spearman_p = scipy_stats.spearmanr(x, y)
        
        results.append({
            '特征': FEATURE_NAMES_CN.get(feat, feat),
            '特征代码': feat,
            'Pearson r': f'{pearson_r:.4f}',
            'Pearson p值': f'{pearson_p:.4e}' if pearson_p < 0.001 else f'{pearson_p:.4f}',
            'Spearman ρ': f'{spearman_r:.4f}',
            'Spearman p值': f'{spearman_p:.4e}' if spearman_p < 0.001 else f'{spearman_p:.4f}',
            '显著性': '***' if pearson_p < 0.001 else ('**' if pearson_p < 0.01 else ('*' if pearson_p < 0.05 else '')),
        })
    
    return pd.DataFrame(results)

def run_experiment():
    """运行实验"""
    log_experiment('4.1.4', '开始相关性检验')
    
    # 合并数据
    all_data = []
    for code, name, sector in STOCK_LIST:
        code_dir = DATA_PROCESSED / code.replace('.', '_')
        file_path = code_dir / f"kline_features_1M.parquet"
        
        if file_path.exists():
            df = pd.read_parquet(file_path)
            all_data.append(df)
    
    if not all_data:
        log_experiment('4.1.4', '[DEMO] 使用模拟数据')
        np.random.seed(42)
        n = 10000
        df_all = pd.DataFrame({
            'ti': np.random.normal(0, 1000, n),
            'return_1': np.random.normal(0, 0.001, n),
            'relative_volume': np.random.lognormal(0, 0.5, n),
            'rsi': np.random.uniform(20, 80, n),
            'atr_pct': np.random.exponential(0.5, n),
            'pv_corr': np.random.uniform(-1, 1, n),
            'ti_zscore': np.random.normal(0, 1, n),
            'return_zscore': np.random.normal(0, 1, n),
        })
        # 添加相关性
        df_all['return_1'] = df_all['return_1'] + 0.0001 * df_all['ti'] / 1000
    else:
        df_all = pd.concat(all_data, ignore_index=True)
    
    # 计算相关性
    df_corr = compute_correlations(df_all)
    
    # 按Pearson r绝对值排序
    df_corr['abs_r'] = df_corr['Pearson r'].apply(lambda x: abs(float(x)))
    df_corr = df_corr.sort_values('abs_r', ascending=False).drop('abs_r', axis=1)
    
    # 保存
    output_path = get_output_path('table_4_1_4_correlation', 'csv')
    df_corr.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    log_experiment('4.1.4', f'结果已保存: {output_path}')
    
    print("\n" + "="*70)
    print("  表 4.1-4: 核心特征与同期收益率相关系数")
    print("="*70)
    print(df_corr.to_string(index=False))
    
    return df_corr


if __name__ == "__main__":
    set_seed()
    run_experiment()
