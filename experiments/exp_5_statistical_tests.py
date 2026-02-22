"""
统计显著性检验与补充分析
- Wilcoxon符号秩检验：深度学习 vs 传统ML
- 逐股胜负统计
- Top-K准确率（高置信度区域的预测精度）
"""

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "ch4" / "tables"

def load_per_stock_data():
    df = pd.read_csv(TABLES_DIR / "table_4_2_2_model_comparison.csv")
    return df

def wilcoxon_tests(df):
    stocks = df['股票代码'].unique()
    
    dl_models = ['CNN-LSTM', 'LSTM', 'GRU', 'Transformer', 'PV-Transformer']
    ml_models = ['XGBoost', 'RandomForest', 'LogisticRegression']
    
    print("=" * 70)
    print("Wilcoxon符号秩检验（单侧，H1: model_A > model_B）")
    print("=" * 70)
    
    comparisons = [
        ('CNN-LSTM', 'XGBoost', 'DL最优 vs ML最优'),
        ('CNN-LSTM', 'RandomForest', 'CNN-LSTM vs RF'),
        ('LSTM', 'XGBoost', 'LSTM vs XGBoost'),
        ('GRU', 'XGBoost', 'GRU vs XGBoost'),
        ('Transformer', 'XGBoost', 'Transformer vs XGBoost'),
    ]
    
    results = []
    
    for model_a, model_b, label in comparisons:
        for metric in ['AUC', 'F1-macro']:
            a_vals = []
            b_vals = []
            for stock in stocks:
                a_row = df[(df['股票代码'] == stock) & (df['模型'] == model_a)]
                b_row = df[(df['股票代码'] == stock) & (df['模型'] == model_b)]
                if len(a_row) > 0 and len(b_row) > 0:
                    a_vals.append(a_row[metric].values[0])
                    b_vals.append(b_row[metric].values[0])
            
            a_vals = np.array(a_vals)
            b_vals = np.array(b_vals)
            wins = np.sum(a_vals > b_vals)
            n = len(a_vals)
            
            try:
                stat, p = wilcoxon(a_vals, b_vals, alternative='greater')
            except Exception as e:
                stat, p = np.nan, np.nan
            
            result = {
                'comparison': label,
                'metric': metric,
                'model_a_mean': f"{a_vals.mean():.4f}",
                'model_b_mean': f"{b_vals.mean():.4f}",
                'wins': f"{wins}/{n}",
                'W_stat': f"{stat:.1f}" if not np.isnan(stat) else "N/A",
                'p_value': f"{p:.4f}" if not np.isnan(p) else "N/A",
                'significant': "Yes" if (not np.isnan(p) and p < 0.05) else "No"
            }
            results.append(result)
            
            sig_mark = "***" if (not np.isnan(p) and p < 0.01) else ("**" if (not np.isnan(p) and p < 0.05) else "")
            print(f"\n{label} [{metric}]:")
            print(f"  {model_a}: {a_vals.mean():.4f} ± {a_vals.std():.4f}")
            print(f"  {model_b}: {b_vals.mean():.4f} ± {b_vals.std():.4f}")
            print(f"  胜负: {model_a}胜 {wins}/{n} 只股票")
            print(f"  Wilcoxon W={stat:.1f}, p={p:.4f} {sig_mark}")
    
    print("\n" + "=" * 70)
    print("DL整体 vs ML整体（每只股票取DL最优AUC vs ML最优AUC）")
    print("=" * 70)
    
    dl_best_auc = []
    ml_best_auc = []
    for stock in stocks:
        dl_aucs = df[(df['股票代码'] == stock) & (df['模型'].isin(dl_models))]['AUC'].values
        ml_aucs = df[(df['股票代码'] == stock) & (df['模型'].isin(ml_models))]['AUC'].values
        if len(dl_aucs) > 0 and len(ml_aucs) > 0:
            dl_best_auc.append(dl_aucs.max())
            ml_best_auc.append(ml_aucs.max())
    
    dl_best_auc = np.array(dl_best_auc)
    ml_best_auc = np.array(ml_best_auc)
    wins = np.sum(dl_best_auc > ml_best_auc)
    
    try:
        stat, p = wilcoxon(dl_best_auc, ml_best_auc, alternative='greater')
    except Exception as e:
        stat, p = np.nan, np.nan
    
    print(f"  DL最优AUC: {dl_best_auc.mean():.4f} ± {dl_best_auc.std():.4f}")
    print(f"  ML最优AUC: {ml_best_auc.mean():.4f} ± {ml_best_auc.std():.4f}")
    print(f"  DL胜: {wins}/{len(dl_best_auc)} 只股票")
    print(f"  Wilcoxon W={stat:.1f}, p={p:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "table_5_wilcoxon_tests.csv", index=False, encoding='utf-8-sig')
    print(f"\n结果已保存至 {TABLES_DIR / 'table_5_wilcoxon_tests.csv'}")
    
    return results_df


def win_count_summary(df):
    stocks = df['股票代码'].unique()
    models = df['模型'].unique()
    
    print("\n" + "=" * 70)
    print("逐股胜负统计（AUC维度）")
    print("=" * 70)
    
    dl_models = ['CNN-LSTM', 'LSTM', 'GRU', 'Transformer', 'PV-Transformer']
    ml_models = ['XGBoost', 'RandomForest', 'LogisticRegression']
    
    print(f"\n{'对比':30s} {'DL胜':>6s} {'ML胜':>6s} {'平':>4s}")
    print("-" * 50)
    
    for dl in dl_models:
        for ml in ml_models:
            dl_wins = 0
            ml_wins = 0
            ties = 0
            for stock in stocks:
                dl_row = df[(df['股票代码'] == stock) & (df['模型'] == dl)]
                ml_row = df[(df['股票代码'] == stock) & (df['模型'] == ml)]
                if len(dl_row) > 0 and len(ml_row) > 0:
                    dl_auc = dl_row['AUC'].values[0]
                    ml_auc = ml_row['AUC'].values[0]
                    if dl_auc > ml_auc:
                        dl_wins += 1
                    elif ml_auc > dl_auc:
                        ml_wins += 1
                    else:
                        ties += 1
            print(f"{dl:15s} vs {ml:12s} {dl_wins:6d} {ml_wins:6d} {ties:4d}")


if __name__ == "__main__":
    df = load_per_stock_data()
    wilcoxon_tests(df)
    win_count_summary(df)
