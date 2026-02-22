"""
实验 4.4.1: 双模型SHAP归因对比（XGBoost TreeSHAP + CNN-LSTM DeepSHAP）

对应论文:
- 表 4.4-1: 双模型SHAP特征重要性对比
- 图 4.4-1: 特征重要性对比图

使用真正的SHAP值（而非gain），对传统ML最优模型（XGBoost）
与深度学习最优模型（CNN-LSTM）进行特征归因对比。

输出:
- table_4_4_1_feature_importance.csv
- fig_4_4_1_feature_importance.png
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from exp_config import *
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
setup_plot()

# ============================================================
# 导入模型类
# ============================================================
import importlib.util

MODELS_IMPORTED = False
try:
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    try:
        spec = importlib.util.spec_from_file_location(
            "kline_model_trainer",
            PROJECT_ROOT / "scripts" / "08_model_trainer.py"
        )
        trainer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer_module)
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr

    CNNLSTMBaseline = trainer_module.CNNLSTMBaseline
    SklearnModelWrapper = trainer_module.SklearnModelWrapper
    create_model = trainer_module.create_model
    import __main__
    __main__.SklearnModelWrapper = SklearnModelWrapper
    MODELS_IMPORTED = True
    print("[OK] 模型类导入成功")
except Exception as e:
    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr
    print(f"[ERROR] 模型导入失败: {e}")

FEATURE_COLS = [
    'kline_position', 'range_pct', 'return_1', 'return_5', 'return_20',
    'return_60', 'return_zscore', 'atr_pct', 'volatility_20', 'ti',
    'ti_5', 'ti_60', 'ti_zscore', 'relative_volume', 'volume_change',
    'pv_corr', 'rsi', 'bb_position', 'macd_dif', 'macd_dea', 'macd', 'market_regime'
]
SEQ_LEN = 60
N_FEATURES = len(FEATURE_COLS)


# ============================================================
# 数据加载
# ============================================================

def load_dataset(code: str):
    dataset_path = DATA_PROCESSED.parent / 'datasets' / f"dataset_{code.replace('.', '_')}_1M.pkl"
    if dataset_path.exists():
        with open(dataset_path, 'rb') as f:
            return pickle.load(f)
    return None


def get_test_data(dataset) -> np.ndarray:
    """从数据集中提取测试集X，返回 (N, seq_len, n_features)"""
    ds = dataset['test']
    if hasattr(ds, 'X'):
        X = ds.X
    else:
        X = ds[0]
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    return np.array(X, dtype=np.float32)


# ============================================================
# XGBoost TreeSHAP
# ============================================================

def compute_xgboost_shap(code: str = 'HK.00700') -> pd.DataFrame:
    """使用TreeExplainer计算XGBoost的真正SHAP值"""
    import shap

    model_path = MODELS_DIR / 'xgboost' / f'model_{code.replace(".", "_")}_1M.pkl'
    if not model_path.exists():
        print(f"  [WARN] XGBoost模型不存在: {model_path}")
        return None

    with open(model_path, 'rb') as f:
        wrapper = pickle.load(f)
    xgb_model = wrapper.model if hasattr(wrapper, 'model') else wrapper

    dataset = load_dataset(code)
    if dataset is None:
        print(f"  [WARN] 数据集不存在: {code}")
        return None

    X_test = get_test_data(dataset)
    X_flat = X_test.reshape(X_test.shape[0], -1)

    bg_idx = np.random.choice(X_flat.shape[0], min(500, X_flat.shape[0]), replace=False)
    background = X_flat[bg_idx]

    print(f"  [INFO] XGBoost TreeSHAP: {X_flat.shape[0]} 样本, {X_flat.shape[1]} 特征维度")
    explainer = shap.TreeExplainer(xgb_model, background)

    sample_idx = np.random.choice(X_flat.shape[0], min(2000, X_flat.shape[0]), replace=False)
    X_sample = X_flat[sample_idx]

    shap_values = explainer.shap_values(X_sample)

    shap_arr = np.array(shap_values)
    if shap_arr.ndim == 3:
        shap_abs = np.abs(shap_arr).mean(axis=2).mean(axis=0)
    elif isinstance(shap_values, list):
        shap_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
    else:
        shap_abs = np.abs(shap_arr).mean(axis=0)

    print(f"  [DEBUG] shap聚合后维度: {shap_abs.shape}, 期望: {SEQ_LEN * N_FEATURES}")

    if shap_abs.shape[0] == SEQ_LEN * N_FEATURES:
        shap_matrix = shap_abs.reshape(SEQ_LEN, N_FEATURES)
        feature_shap = shap_matrix.sum(axis=0)
    elif shap_abs.shape[0] > N_FEATURES:
        n_steps = shap_abs.shape[0] // N_FEATURES
        feature_shap = shap_abs[:n_steps * N_FEATURES].reshape(n_steps, N_FEATURES).sum(axis=0)
    else:
        feature_shap = shap_abs[:N_FEATURES]

    total = feature_shap.sum()
    if total > 0:
        feature_shap = feature_shap / total

    return pd.DataFrame({
        '特征代码': FEATURE_COLS,
        'XGBoost_SHAP': feature_shap,
    })


# ============================================================
# CNN-LSTM DeepSHAP
# ============================================================

def compute_cnn_lstm_shap(code: str = 'HK.00700') -> pd.DataFrame:
    """使用DeepExplainer计算CNN-LSTM的SHAP值"""
    import shap

    if not MODELS_IMPORTED:
        print("  [ERROR] 模型类未导入")
        return None

    model_path = MODELS_DIR / 'cnn_lstm' / f'model_{code.replace(".", "_")}_1M.pt'
    if not model_path.exists():
        print(f"  [WARN] CNN-LSTM模型不存在: {model_path}")
        return None

    device = torch.device('cpu')
    model = create_model('cnn_lstm', input_dim=N_FEATURES, seq_len=SEQ_LEN)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    dataset = load_dataset(code)
    if dataset is None:
        return None

    X_test = get_test_data(dataset)

    bg_size = min(200, X_test.shape[0])
    bg_idx = np.random.choice(X_test.shape[0], bg_size, replace=False)
    background = torch.FloatTensor(X_test[bg_idx])

    sample_size = min(500, X_test.shape[0])
    sample_idx = np.random.choice(X_test.shape[0], sample_size, replace=False)
    X_sample = torch.FloatTensor(X_test[sample_idx])

    print(f"  [INFO] CNN-LSTM GradientExplainer: background={bg_size}, samples={sample_size}")

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(X_sample)

    shap_arr = np.array(shap_values)
    print(f"  [DEBUG] CNN-LSTM SHAP原始维度: {shap_arr.shape}")
    # shape: (samples, seq_len, features, classes) → 对类别取均值，对时间步求和，对样本取均值
    if shap_arr.ndim == 4:
        feature_shap = np.abs(shap_arr).mean(axis=3).mean(axis=0).sum(axis=0)
    elif shap_arr.ndim == 3:
        feature_shap = np.abs(shap_arr).mean(axis=0).sum(axis=0)
    else:
        feature_shap = np.abs(shap_arr).mean(axis=0)

    total = feature_shap.sum()
    if total > 0:
        feature_shap = feature_shap / total

    return pd.DataFrame({
        '特征代码': FEATURE_COLS,
        'CNN_LSTM_SHAP': feature_shap,
    })


# ============================================================
# 合并与可视化
# ============================================================

def merge_and_rank(df_xgb: pd.DataFrame, df_cnn: pd.DataFrame) -> pd.DataFrame:
    df = df_xgb.merge(df_cnn, on='特征代码')
    df['特征名称'] = df['特征代码'].map(FEATURE_NAMES_CN)
    df['特征类别'] = df['特征代码'].map({
        'kline_position': 'K线形态', 'range_pct': 'K线形态',
        'return_1': '价格动量', 'return_5': '价格动量', 'return_20': '价格动量',
        'return_60': '价格动量', 'return_zscore': '价格动量',
        'atr_pct': '波动率', 'volatility_20': '波动率',
        'ti': '成交不平衡', 'ti_5': '成交不平衡', 'ti_60': '成交不平衡', 'ti_zscore': '成交不平衡',
        'relative_volume': '成交量', 'volume_change': '成交量', 'pv_corr': '成交量',
        'rsi': '技术指标', 'bb_position': '技术指标',
        'macd_dif': '技术指标', 'macd_dea': '技术指标', 'macd': '技术指标',
        'market_regime': '市场状态',
    })
    df['平均SHAP'] = (df['XGBoost_SHAP'] + df['CNN_LSTM_SHAP']) / 2
    df = df.sort_values('平均SHAP', ascending=False).reset_index(drop=True)
    df['排名'] = df.index + 1
    return df[['排名', '特征代码', '特征名称', 'XGBoost_SHAP', 'CNN_LSTM_SHAP', '平均SHAP', '特征类别']]


def plot_dual_shap(df: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(12, 9))

    df_top = df.head(15).copy().iloc[::-1]
    y_pos = np.arange(len(df_top))
    bar_height = 0.35

    bars1 = ax.barh(y_pos + bar_height/2, df_top['XGBoost_SHAP'], bar_height,
                    label='XGBoost (TreeSHAP)', color='#1f77b4', alpha=0.85)
    bars2 = ax.barh(y_pos - bar_height/2, df_top['CNN_LSTM_SHAP'], bar_height,
                    label='CNN-LSTM (DeepSHAP)', color='#ff7f0e', alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_top['特征名称'])
    ax.set_xlabel('归一化SHAP值')
    ax.set_title('图 4.4-1: 双模型SHAP特征重要性对比')
    ax.legend(loc='lower right')

    for bar in bars1:
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height()/2,
                f'{w:.3f}', va='center', fontsize=8)
    for bar in bars2:
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height()/2,
                f'{w:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {output_path}")


# ============================================================
# 主函数
# ============================================================

def run_experiment():
    log_experiment('4.4.1', '开始双模型SHAP归因分析')
    set_seed()

    code = 'HK.00700'

    print("\n[Step 1] XGBoost TreeSHAP...")
    df_xgb = compute_xgboost_shap(code)
    if df_xgb is None:
        print("[ERROR] XGBoost SHAP失败")
        return

    print("\n[Step 2] CNN-LSTM DeepSHAP...")
    df_cnn = compute_cnn_lstm_shap(code)
    if df_cnn is None:
        print("[ERROR] CNN-LSTM SHAP失败")
        return

    print("\n[Step 3] 合并排序...")
    df_merged = merge_and_rank(df_xgb, df_cnn)

    table_path = get_output_path('table_4_4_1_feature_importance', 'csv')
    df_merged.to_csv(table_path, index=False, encoding='utf-8-sig')
    log_experiment('4.4.1', f'表格已保存: {table_path}')

    fig_path = get_output_path('fig_4_4_1_feature_importance', 'png')
    plot_dual_shap(df_merged, fig_path)

    print("\n" + "=" * 70)
    print("  双模型SHAP特征重要性对比（Top 15）")
    print("=" * 70)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 120)
    print(df_merged.head(15).to_string(index=False))

    xgb_rank = df_merged.set_index('特征代码')['排名']
    cnn_rank = df_merged.set_index('特征代码')
    cnn_rank['CNN排名'] = cnn_rank['CNN_LSTM_SHAP'].rank(ascending=False).astype(int)
    spearman_corr = df_merged['XGBoost_SHAP'].corr(df_merged['CNN_LSTM_SHAP'], method='spearman')
    print(f"\n  Spearman秩相关系数: {spearman_corr:.3f}")

    ti_features = ['ti', 'ti_5', 'ti_60', 'ti_zscore']
    ti_xgb = df_merged[df_merged['特征代码'].isin(ti_features)]['XGBoost_SHAP'].sum()
    ti_cnn = df_merged[df_merged['特征代码'].isin(ti_features)]['CNN_LSTM_SHAP'].sum()
    print(f"  TI类特征合计: XGBoost={ti_xgb:.1%}, CNN-LSTM={ti_cnn:.1%}")

    return df_merged


if __name__ == "__main__":
    run_experiment()
