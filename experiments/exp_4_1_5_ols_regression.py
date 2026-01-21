"""
实验 4.1.5: OLS回归分析

对应论文:
- 表 4.1-5: 特征对同期收益的线性回归结果

输出:
- table_4_1_5_ols_regression.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from exp_config import *
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

def run_ols_regression(df: pd.DataFrame, features: list, target: str = 'return_1'):
    """运行OLS回归"""
    results = []
    
    # 准备数据
    valid_features = [f for f in features if f in df.columns and f != target]
    data = df[[target] + valid_features].dropna()
    
    if len(data) < 100:
        return pd.DataFrame()
    
    y = data[target].values
    
    # 单变量回归
    for feat in valid_features:
        X = data[feat].values
        
        # 添加常数项
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        try:
            # OLS估计
            beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            y_pred = X_with_const @ beta
            residuals = y - y_pred
            
            # R²
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - y.mean())**2)
            r_squared = 1 - ss_res / ss_tot
            
            # t统计量
            n = len(y)
            mse = ss_res / (n - 2)
            var_beta = mse * np.linalg.inv(X_with_const.T @ X_with_const)
            se_beta = np.sqrt(np.diag(var_beta))
            t_stat = beta[1] / se_beta[1]
            p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), n - 2))
            
            results.append({
                '特征': FEATURE_NAMES_CN.get(feat, feat),
                '特征代码': feat,
                '系数β': f'{beta[1]:.6f}',
                't统计量': f'{t_stat:.2f}',
                'p值': f'{p_value:.4e}' if p_value < 0.001 else f'{p_value:.4f}',
                'R²': f'{r_squared:.4f}',
                '显著性': '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else '')),
            })
        except:
            continue
    
    return pd.DataFrame(results)

def run_experiment():
    """运行实验"""
    log_experiment('4.1.5', '开始OLS回归分析')
    
    # 合并数据
    all_data = []
    for code, name, sector in STOCK_LIST:
        code_dir = DATA_PROCESSED / code.replace('.', '_')
        file_path = code_dir / f"kline_features_1M.parquet"
        
        if file_path.exists():
            df = pd.read_parquet(file_path)
            all_data.append(df)
    
    if not all_data:
        log_experiment('4.1.5', '[DEMO] 使用模拟数据')
        np.random.seed(42)
        n = 10000
        
        ti = np.random.normal(0, 1000, n)
        rv = np.random.lognormal(0, 0.5, n)
        rsi = np.random.uniform(20, 80, n)
        
        df_all = pd.DataFrame({
            'ti': ti,
            'relative_volume': rv,
            'rsi': rsi,
            'ti_zscore': (ti - ti.mean()) / ti.std(),
            'atr_pct': np.random.exponential(0.5, n),
            'pv_corr': np.random.uniform(-1, 1, n),
            'return_1': 0.0001 * ti / 1000 + 0.00005 * (rv - 1) + np.random.normal(0, 0.001, n),
        })
    else:
        df_all = pd.concat(all_data, ignore_index=True)
    
    # 运行回归
    df_results = run_ols_regression(df_all, ALL_FEATURES)
    
    # 按R²排序
    df_results['R2_val'] = df_results['R²'].apply(lambda x: float(x))
    df_results = df_results.sort_values('R2_val', ascending=False).drop('R2_val', axis=1)
    
    # 保存
    output_path = get_output_path('table_4_1_5_ols_regression', 'csv')
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    log_experiment('4.1.5', f'结果已保存: {output_path}')
    
    print("\n" + "="*70)
    print("  表 4.1-5: 特征对同期收益的线性回归结果")
    print("="*70)
    print(df_results.to_string(index=False))
    
    return df_results


if __name__ == "__main__":
    set_seed()
    run_experiment()
