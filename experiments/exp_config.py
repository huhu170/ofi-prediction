#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验配置文件
============
论文第四章所有实验的共享配置

版本: v1.0
日期: 2026-01-14
"""

import sys
from pathlib import Path

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'experiment_results'

# 确保结果目录存在
RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / 'tables').mkdir(exist_ok=True)
(RESULTS_DIR / 'figures').mkdir(exist_ok=True)

# 添加scripts到路径
sys.path.insert(0, str(SCRIPTS_DIR))

# ============================================================
# 数据路径
# ============================================================
PROCESSED_DATA_DIR = DATA_DIR / 'processed' / 'combined'
DATASET_DIR = PROCESSED_DATA_DIR / 'dataset_T100_k20'

# ============================================================
# 实验参数
# ============================================================

# 预测步长配置
PREDICTION_HORIZONS = [20, 50, 100]  # k值

# 标签阈值配置 (波动率自适应系数)
LABEL_THRESHOLD_ALPHA = 0.3  # 基准阈值系数

# 滑动窗口配置
SEQUENCE_LENGTH = 100  # T=100

# 模型列表
TRADITIONAL_MODELS = ['arima']  # 传统时序模型
ML_MODELS = ['logistic', 'rf', 'xgboost']  # 机器学习模型
BASELINE_MODELS = TRADITIONAL_MODELS + ML_MODELS  # 全部基准模型
DEEP_MODELS = ['lstm', 'gru', 'deeplob', 'transformer', 'smart_trans']
ALL_MODELS = BASELINE_MODELS + DEEP_MODELS  # 9个模型

# OFI特征变体
OFI_VARIANTS = ['ofi_l1', 'ofi_l5_exp', 'ofi_l10_exp', 'ofi_pca', 'smart_ofi']

# ============================================================
# 可视化配置
# ============================================================
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 图表样式
FIGURE_DPI = 150
FIGURE_SIZE_SMALL = (8, 6)
FIGURE_SIZE_MEDIUM = (10, 8)
FIGURE_SIZE_LARGE = (12, 10)

# 颜色方案
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'info': '#17A2B8',
}

MODEL_COLORS = {
    'arima': '#808080',
    'logistic': '#A0A0A0',
    'random_forest': '#90EE90',
    'xgboost': '#32CD32',
    'lstm': '#87CEEB',
    'gru': '#4169E1',
    'deeplob': '#9370DB',
    'transformer': '#FF6347',
    'smart_trans': '#DC143C',
}

# ============================================================
# 工具函数
# ============================================================

def save_table(df, name, format='csv'):
    """保存表格到结果目录"""
    path = RESULTS_DIR / 'tables' / f'{name}.{format}'
    if format == 'csv':
        df.to_csv(path, index=True, encoding='utf-8-sig')
    elif format == 'latex':
        df.to_latex(path)
    print(f"[保存] 表格: {path}")
    return path

def save_figure(fig, name, format='png'):
    """保存图表到结果目录"""
    path = RESULTS_DIR / 'figures' / f'{name}.{format}'
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    print(f"[保存] 图表: {path}")
    return path

def load_feature_data():
    """加载特征数据"""
    import pandas as pd
    parquet_files = list(PROCESSED_DATA_DIR.glob('features_*.parquet'))
    if not parquet_files:
        raise FileNotFoundError(f"未找到特征文件: {PROCESSED_DATA_DIR}")
    
    dfs = []
    for f in parquet_files:
        df = pd.read_parquet(f)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True).sort_values('timestamp')

def load_dataset():
    """加载训练数据集"""
    import numpy as np
    
    X_train = np.load(DATASET_DIR / 'X_train.npy')
    y_train = np.load(DATASET_DIR / 'y_train.npy')
    X_val = np.load(DATASET_DIR / 'X_val.npy')
    y_val = np.load(DATASET_DIR / 'y_val.npy')
    X_test = np.load(DATASET_DIR / 'X_test.npy')
    y_test = np.load(DATASET_DIR / 'y_test.npy')
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }

def print_section(title):
    """打印分隔线"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

# ============================================================
# 运行入口
# ============================================================
if __name__ == '__main__':
    print_section("实验配置检查")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据目录: {DATA_DIR}")
    print(f"模型目录: {MODELS_DIR}")
    print(f"结果目录: {RESULTS_DIR}")
    print(f"\n数据集目录: {DATASET_DIR}")
    print(f"  存在: {DATASET_DIR.exists()}")
