"""
实验 4.1.1: 样本描述性统计汇总

对应论文:
- 表 4.1-1: 样本描述性统计汇总

输出:
- table_4_1_sample_stats.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from exp_config import *
import pandas as pd
import numpy as np

def load_kline_data(code: str, ktype: str) -> pd.DataFrame:
    """加载清洗后的K线数据"""
    code_dir = DATA_PROCESSED / code.replace('.', '_')
    file_path = code_dir / f"kline_cleaned_{ktype}.parquet"
    
    if file_path.exists():
        return pd.read_parquet(file_path)
    return pd.DataFrame()

def compute_sample_stats(code: str, name: str) -> dict:
    """计算单只股票的样本统计"""
    stats = {
        '股票代码': code,
        '股票名称': name,
    }
    
    for ktype in KLINE_TYPES:
        df = load_kline_data(code, ktype)
        
        if df.empty:
            stats[f'{ktype}_样本量'] = 0
            stats[f'{ktype}_缺失率'] = 'N/A'
            stats[f'{ktype}_起始日期'] = 'N/A'
            stats[f'{ktype}_结束日期'] = 'N/A'
        else:
            stats[f'{ktype}_样本量'] = len(df)
            
            # 计算缺失率（基于时间连续性）
            if 'ts' in df.columns:
                df['ts'] = pd.to_datetime(df['ts'])
                # 对于分钟K线，检查时间间隔
                if ktype == '1M':
                    expected_interval = pd.Timedelta(minutes=1)
                elif ktype == '5M':
                    expected_interval = pd.Timedelta(minutes=5)
                elif ktype == '60M':
                    expected_interval = pd.Timedelta(hours=1)
                else:
                    expected_interval = pd.Timedelta(days=1)
                
                time_diff = df['ts'].diff()
                # 简化缺失率计算
                missing_rate = (df.isna().sum().sum()) / (len(df) * len(df.columns)) * 100
                stats[f'{ktype}_缺失率'] = f'{missing_rate:.2f}%'
                
                stats[f'{ktype}_起始日期'] = df['ts'].min().strftime('%Y-%m-%d')
                stats[f'{ktype}_结束日期'] = df['ts'].max().strftime('%Y-%m-%d')
            else:
                stats[f'{ktype}_缺失率'] = 'N/A'
                stats[f'{ktype}_起始日期'] = 'N/A'
                stats[f'{ktype}_结束日期'] = 'N/A'
    
    return stats

def run_experiment():
    """运行实验"""
    log_experiment('4.1.1', '开始样本描述性统计')
    
    results = []
    
    for code, name, sector in STOCK_LIST:
        log_experiment('4.1.1', f'处理 {code} {name}')
        stats = compute_sample_stats(code, name)
        stats['行业'] = sector
        results.append(stats)
    
    # 汇总为DataFrame
    df_results = pd.DataFrame(results)
    
    # 保存结果
    output_path = get_output_path('table_4_1_sample_stats', 'csv')
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    log_experiment('4.1.1', f'结果已保存: {output_path}')
    
    # 打印汇总
    print("\n" + "="*60)
    print("  表 4.1-1: 样本描述性统计汇总")
    print("="*60)
    print(df_results.to_string(index=False))
    
    return df_results


if __name__ == "__main__":
    set_seed()
    run_experiment()
