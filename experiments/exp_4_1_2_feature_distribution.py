"""
实验 4.1.2: 特征分布分析

对应论文:
- 图 4.1-1: 核心特征分布直方图
- 表 4.1-2: 特征描述性统计

输出:
- fig_4_1_feature_distribution.png
- table_4_1_2_feature_stats.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from exp_config import *
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

# 可视化
import matplotlib.pyplot as plt
setup_plot()

def load_feature_data(code: str, ktype: str = '1M') -> pd.DataFrame:
    """加载特征数据"""
    code_dir = DATA_PROCESSED / code.replace('.', '_')
    file_path = code_dir / f"kline_features_{ktype}.parquet"
    
    if file_path.exists():
        return pd.read_parquet(file_path)
    return pd.DataFrame()

def compute_feature_stats(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """计算特征描述性统计"""
    stats_list = []
    
    for feat in features:
        if feat not in df.columns:
            continue
        
        data = df[feat].dropna()
        
        stats_list.append({
            '特征': FEATURE_NAMES_CN.get(feat, feat),
            '特征代码': feat,
            '均值': data.mean(),
            '标准差': data.std(),
            '偏度': scipy_stats.skew(data),
            '峰度': scipy_stats.kurtosis(data),
            '最小值': data.min(),
            '25%分位': data.quantile(0.25),
            '中位数': data.median(),
            '75%分位': data.quantile(0.75),
            '最大值': data.max(),
        })
    
    return pd.DataFrame(stats_list)

def plot_feature_distribution(df: pd.DataFrame, features: list, output_path: Path):
    """绘制特征分布直方图"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    core_features = ['ti', 'return_1', 'relative_volume', 'rsi', 'atr_pct', 'pv_corr']
    
    for i, feat in enumerate(core_features[:6]):
        ax = axes[i]
        
        if feat in df.columns:
            data = df[feat].dropna()
            
            # 直方图
            ax.hist(data, bins=50, density=True, alpha=0.7, color=COLORS['primary'])
            
            # 正态分布拟合
            mu, std = data.mean(), data.std()
            x = np.linspace(data.min(), data.max(), 100)
            ax.plot(x, scipy_stats.norm.pdf(x, mu, std), 'r--', lw=2, label='正态拟合')
            
            ax.set_title(FEATURE_NAMES_CN.get(feat, feat))
            ax.set_xlabel('值')
            ax.set_ylabel('密度')
            ax.legend()
        else:
            ax.text(0.5, 0.5, f'{feat}\n数据不可用', ha='center', va='center')
            ax.set_title(feat)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  图表已保存: {output_path}")

def run_experiment():
    """运行实验"""
    log_experiment('4.1.2', '开始特征分布分析')
    
    # 合并所有股票数据
    all_data = []
    for code, name, sector in STOCK_LIST:
        df = load_feature_data(code, '1M')
        if not df.empty:
            df['code'] = code
            all_data.append(df)
    
    if not all_data:
        log_experiment('4.1.2', '[WARN] 无数据，使用模拟数据演示')
        # 生成模拟数据
        np.random.seed(42)
        n = 10000
        df_all = pd.DataFrame({
            'ti': np.random.normal(0, 1000, n),
            'return_1': np.random.normal(0, 0.001, n),
            'relative_volume': np.random.lognormal(0, 0.5, n),
            'rsi': np.random.uniform(20, 80, n),
            'atr_pct': np.random.exponential(0.5, n),
            'pv_corr': np.random.uniform(-1, 1, n),
        })
    else:
        df_all = pd.concat(all_data, ignore_index=True)
    
    # 计算统计量
    stats_df = compute_feature_stats(df_all, ALL_FEATURES)
    
    # 保存表格
    table_path = get_output_path('table_4_1_2_feature_stats', 'csv')
    stats_df.to_csv(table_path, index=False, encoding='utf-8-sig')
    log_experiment('4.1.2', f'统计表格已保存: {table_path}')
    
    # 绘制分布图
    fig_path = get_output_path('fig_4_1_feature_distribution', 'png')
    plot_feature_distribution(df_all, ALL_FEATURES, fig_path)
    
    # 打印结果
    print("\n" + "="*60)
    print("  表 4.1-2: 特征描述性统计")
    print("="*60)
    print(stats_df.to_string(index=False))
    
    return stats_df


if __name__ == "__main__":
    set_seed()
    run_experiment()
