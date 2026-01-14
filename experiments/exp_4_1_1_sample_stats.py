#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 4.1.1: 样本描述性统计汇总
==============================
对应论文: 表4-1 样本描述性统计汇总

输出:
- tables/table_4_1_sample_stats.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime
from exp_config import (
    PROCESSED_DATA_DIR, RESULTS_DIR, 
    save_table, print_section, load_feature_data
)

def calculate_sample_stats():
    """计算样本描述性统计"""
    print_section("实验 4.1.1: 样本描述性统计")
    
    # 加载数据
    print("[1] 加载特征数据...")
    try:
        df = load_feature_data()
    except FileNotFoundError as e:
        print(f"  警告: {e}")
        print("  使用模拟数据进行演示...")
        df = generate_demo_data()
    
    print(f"  数据形状: {df.shape}")
    print(f"  时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    
    # 计算统计量
    print("\n[2] 计算描述性统计...")
    
    # 按股票代码分组统计
    stats_list = []
    
    if 'code' in df.columns:
        codes = df['code'].unique()
    else:
        codes = ['ALL']
        df['code'] = 'ALL'
    
    for code in codes:
        sub_df = df[df['code'] == code]
        
        # 计算交易日数
        if 'timestamp' in sub_df.columns:
            sub_df['date'] = pd.to_datetime(sub_df['timestamp']).dt.date
            trading_days = sub_df['date'].nunique()
        else:
            trading_days = len(sub_df) // 2340  # 假设每天约2340个10秒窗口
        
        # 计算统计量
        stats = {
            '股票代码': code,
            '样本期': f"{sub_df['timestamp'].min().strftime('%Y-%m-%d') if 'timestamp' in sub_df.columns else 'N/A'} ~ {sub_df['timestamp'].max().strftime('%Y-%m-%d') if 'timestamp' in sub_df.columns else 'N/A'}",
            '有效交易日': trading_days,
            '10秒窗口数': len(sub_df),
            '日均窗口数': len(sub_df) // max(trading_days, 1),
            '缺失率(%)': sub_df.isnull().sum().sum() / (len(sub_df) * len(sub_df.columns)) * 100,
        }
        
        # 如果有OFI列，计算更新频次
        if 'ofi_l1' in sub_df.columns:
            stats['OFI非零比例(%)'] = (sub_df['ofi_l1'] != 0).mean() * 100
        
        stats_list.append(stats)
    
    # 汇总统计
    result_df = pd.DataFrame(stats_list)
    
    # 添加汇总行
    total_stats = {
        '股票代码': '汇总',
        '样本期': f"{df['timestamp'].min().strftime('%Y-%m-%d')} ~ {df['timestamp'].max().strftime('%Y-%m-%d')}" if 'timestamp' in df.columns else 'N/A',
        '有效交易日': result_df['有效交易日'].sum() if len(result_df) > 1 else result_df['有效交易日'].iloc[0],
        '10秒窗口数': len(df),
        '日均窗口数': result_df['日均窗口数'].mean(),
        '缺失率(%)': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
    }
    if 'OFI非零比例(%)' in result_df.columns:
        total_stats['OFI非零比例(%)'] = result_df['OFI非零比例(%)'].mean()
    
    result_df = pd.concat([result_df, pd.DataFrame([total_stats])], ignore_index=True)
    
    # 打印结果
    print("\n[3] 统计结果:")
    print(result_df.to_string(index=False))
    
    # 保存结果
    print("\n[4] 保存结果...")
    save_table(result_df, 'table_4_1_sample_stats')
    
    return result_df

def generate_demo_data():
    """生成演示数据"""
    np.random.seed(42)
    n_samples = 10000
    
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='10s')
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'code': 'DEMO',
        'mid_price': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
        'ofi_l1': np.random.randn(n_samples) * 100,
        'ofi_l5': np.random.randn(n_samples) * 150,
        'smart_ofi': np.random.randn(n_samples) * 80,
    })
    
    return df

if __name__ == '__main__':
    result = calculate_sample_stats()
    print("\n✓ 实验 4.1.1 完成")
