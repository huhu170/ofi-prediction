"""
实验 4.4.1b: 多股票SHAP归因分析（跨标的鲁棒性检验）

扩展exp_4_4_1的单股票SHAP分析至全部10只股票，检验特征重要性排序
在不同标的间的一致性。

输出:
- table_4_4_1b_shap_multi_stock.csv  (逐股特征级SHAP)
- table_4_4_1b_shap_category_summary.csv  (逐股类别级SHAP汇总)
- table_4_4_1b_shap_cross_stock_consistency.csv  (跨标的一致性指标)
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
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 导入模型类（复用现有脚本的导入逻辑）
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
    try:
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
    except:
        pass
    print(f"[ERROR] 模型导入失败: {e}")

FEATURE_COLS = [
    'kline_position', 'range_pct', 'return_1', 'return_5', 'return_20',
    'return_60', 'return_zscore', 'atr_pct', 'volatility_20', 'ti',
    'ti_5', 'ti_60', 'ti_zscore', 'relative_volume', 'volume_change',
    'pv_corr', 'rsi', 'bb_position', 'macd_dif', 'macd_dea', 'macd', 'market_regime'
]
SEQ_LEN = 60
N_FEATURES = len(FEATURE_COLS)

CATEGORY_MAP = {
    'kline_position': 'K线形态', 'range_pct': 'K线形态',
    'return_1': '价格动量', 'return_5': '价格动量', 'return_20': '价格动量',
    'return_60': '价格动量', 'return_zscore': '价格动量',
    'atr_pct': '波动率', 'volatility_20': '波动率',
    'ti': '成交不平衡', 'ti_5': '成交不平衡', 'ti_60': '成交不平衡', 'ti_zscore': '成交不平衡',
    'relative_volume': '成交量', 'volume_change': '成交量', 'pv_corr': '成交量',
    'rsi': '技术指标', 'bb_position': '技术指标',
    'macd_dif': '技术指标', 'macd_dea': '技术指标', 'macd': '技术指标',
    'market_regime': '市场状态',
}

CATEGORIES_ORDER = ['成交不平衡', '价格动量', '技术指标', '成交量', '波动率', 'K线形态', '市场状态']


def load_dataset(code: str):
    dataset_path = DATA_PROCESSED.parent / 'datasets' / f"dataset_{code.replace('.', '_')}_1M.pkl"
    if dataset_path.exists():
        with open(dataset_path, 'rb') as f:
            return pickle.load(f)
    return None


def get_test_data(dataset) -> np.ndarray:
    ds = dataset['test']
    if hasattr(ds, 'X'):
        X = ds.X
    else:
        X = ds[0]
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    return np.array(X, dtype=np.float32)


# ============================================================
# XGBoost TreeSHAP（单股票）
# ============================================================

def compute_xgboost_shap(code: str) -> np.ndarray:
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
        return None

    X_test = get_test_data(dataset)
    X_flat = X_test.reshape(X_test.shape[0], -1)

    bg_idx = np.random.choice(X_flat.shape[0], min(500, X_flat.shape[0]), replace=False)
    background = X_flat[bg_idx]

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

    return feature_shap


# ============================================================
# CNN-LSTM DeepSHAP（单股票）
# ============================================================

def compute_cnn_lstm_shap(code: str) -> np.ndarray:
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

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(X_sample)

    shap_arr = np.array(shap_values)
    if shap_arr.ndim == 4:
        feature_shap = np.abs(shap_arr).mean(axis=3).mean(axis=0).sum(axis=0)
    elif shap_arr.ndim == 3:
        feature_shap = np.abs(shap_arr).mean(axis=0).sum(axis=0)
    else:
        feature_shap = np.abs(shap_arr).mean(axis=0)

    total = feature_shap.sum()
    if total > 0:
        feature_shap = feature_shap / total

    return feature_shap


# ============================================================
# 多股票分析主流程
# ============================================================

def run_multi_stock_shap():
    set_seed()

    all_results = []
    stock_names = {code: name for code, name, _ in STOCK_LIST}

    for i, (code, name, sector) in enumerate(STOCK_LIST):
        print(f"\n{'='*60}")
        print(f"[{i+1}/10] {name} ({code}) - {sector}")
        print(f"{'='*60}")

        print(f"  XGBoost TreeSHAP...")
        xgb_shap = compute_xgboost_shap(code)
        if xgb_shap is None:
            print(f"  [SKIP] XGBoost SHAP失败")
            continue

        print(f"  CNN-LSTM DeepSHAP...")
        cnn_shap = compute_cnn_lstm_shap(code)
        if cnn_shap is None:
            print(f"  [SKIP] CNN-LSTM SHAP失败")
            continue

        for j, feat in enumerate(FEATURE_COLS):
            all_results.append({
                '股票代码': code,
                '股票名称': name,
                '行业': sector,
                '特征': feat,
                '特征类别': CATEGORY_MAP[feat],
                'XGBoost_SHAP': xgb_shap[j],
                'CNN_LSTM_SHAP': cnn_shap[j],
                '平均SHAP': (xgb_shap[j] + cnn_shap[j]) / 2,
            })

        spearman = pd.Series(xgb_shap).corr(pd.Series(cnn_shap), method='spearman')
        print(f"  双模型Spearman秩相关: {spearman:.3f}")

    if not all_results:
        print("[ERROR] 没有成功计算任何股票的SHAP值")
        return

    df_all = pd.DataFrame(all_results)

    # --- 输出1: 逐股特征级SHAP ---
    path1 = get_output_path('table_4_4_1b_shap_multi_stock', 'csv')
    df_all.to_csv(path1, index=False, encoding='utf-8-sig')
    print(f"\n逐股特征级SHAP已保存: {path1}")

    # --- 输出2: 逐股类别级汇总 ---
    df_cat = df_all.groupby(['股票代码', '股票名称', '行业', '特征类别']).agg(
        XGBoost_SHAP=('XGBoost_SHAP', 'sum'),
        CNN_LSTM_SHAP=('CNN_LSTM_SHAP', 'sum'),
        平均SHAP=('平均SHAP', 'sum'),
    ).reset_index()

    path2 = get_output_path('table_4_4_1b_shap_category_summary', 'csv')
    df_cat.to_csv(path2, index=False, encoding='utf-8-sig')
    print(f"逐股类别级汇总已保存: {path2}")

    # --- 输出3: 跨标的一致性分析 ---
    print(f"\n{'='*70}")
    print("跨标的一致性分析")
    print(f"{'='*70}")

    # 3a: 每只股票的Top-1类别
    print("\n各股票Top-1特征类别（按平均SHAP）:")
    consistency_rows = []
    for code, name, sector in STOCK_LIST:
        stock_cat = df_cat[df_cat['股票代码'] == code].sort_values('平均SHAP', ascending=False)
        if len(stock_cat) == 0:
            continue
        top1 = stock_cat.iloc[0]
        top2 = stock_cat.iloc[1] if len(stock_cat) > 1 else None

        ti_shap = stock_cat[stock_cat['特征类别'] == '成交不平衡']['平均SHAP'].values
        ti_val = ti_shap[0] if len(ti_shap) > 0 else 0

        consistency_rows.append({
            '股票代码': code,
            '股票名称': name,
            '行业': sector,
            'Top1类别': top1['特征类别'],
            'Top1_SHAP': top1['平均SHAP'],
            'Top2类别': top2['特征类别'] if top2 is not None else '',
            'Top2_SHAP': top2['平均SHAP'] if top2 is not None else 0,
            'TI类合计SHAP': ti_val,
        })
        print(f"  {name:8s}: Top1={top1['特征类别']}({top1['平均SHAP']:.1%}), "
              f"Top2={top2['特征类别'] if top2 is not None else '-'}({top2['平均SHAP']:.1%}), "
              f"TI类合计={ti_val:.1%}")

    df_consistency = pd.DataFrame(consistency_rows)

    # 3b: TI类在所有股票中的排名统计
    ti_top1_count = sum(1 for r in consistency_rows if r['Top1类别'] == '成交不平衡')
    ti_top2_count = sum(1 for r in consistency_rows if r['Top1类别'] == '成交不平衡' or r['Top2类别'] == '成交不平衡')
    print(f"\n成交不平衡（TI）类排名Top-1的股票数: {ti_top1_count}/{len(consistency_rows)}")
    print(f"成交不平衡（TI）类排名Top-2的股票数: {ti_top2_count}/{len(consistency_rows)}")

    # 3c: 跨股票的特征排名Spearman相关矩阵
    stocks_with_data = df_all['股票代码'].unique()
    if len(stocks_with_data) >= 2:
        pivot = df_all.pivot_table(index='特征', columns='股票代码', values='平均SHAP')
        corr_matrix = pivot.corr(method='spearman')
        mean_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        min_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min()
        max_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()

        print(f"\n跨股票特征排名Spearman相关:")
        print(f"  均值: {mean_corr:.3f}")
        print(f"  范围: [{min_corr:.3f}, {max_corr:.3f}]")

        df_consistency_extra = pd.DataFrame([{
            '指标': '跨股票Spearman秩相关均值',
            '值': f'{mean_corr:.3f}',
        }, {
            '指标': '跨股票Spearman秩相关范围',
            '值': f'[{min_corr:.3f}, {max_corr:.3f}]',
        }, {
            '指标': 'TI类Top-1股票数',
            '值': f'{ti_top1_count}/{len(consistency_rows)}',
        }, {
            '指标': 'TI类Top-2股票数',
            '值': f'{ti_top2_count}/{len(consistency_rows)}',
        }])

        path3 = get_output_path('table_4_4_1b_shap_cross_stock_consistency', 'csv')
        df_consistency.to_csv(path3, index=False, encoding='utf-8-sig')
        df_consistency_extra.to_csv(
            path3, mode='a', index=False, encoding='utf-8-sig',
            header=True
        )
        print(f"\n跨标的一致性指标已保存: {path3}")

    # 3d: 类别级跨股票均值±标准差
    print(f"\n{'='*70}")
    print("特征类别SHAP跨股票汇总（均值±标准差）")
    print(f"{'='*70}")
    cat_summary = df_cat.groupby('特征类别').agg(
        XGB_mean=('XGBoost_SHAP', 'mean'),
        XGB_std=('XGBoost_SHAP', 'std'),
        CNN_mean=('CNN_LSTM_SHAP', 'mean'),
        CNN_std=('CNN_LSTM_SHAP', 'std'),
        AVG_mean=('平均SHAP', 'mean'),
        AVG_std=('平均SHAP', 'std'),
    ).reindex(CATEGORIES_ORDER)

    for cat in CATEGORIES_ORDER:
        if cat in cat_summary.index:
            row = cat_summary.loc[cat]
            print(f"  {cat:8s}: XGB={row['XGB_mean']:.1%}±{row['XGB_std']:.1%}  "
                  f"CNN={row['CNN_mean']:.1%}±{row['CNN_std']:.1%}  "
                  f"平均={row['AVG_mean']:.1%}±{row['AVG_std']:.1%}")

    print(f"\n{'='*70}")
    print("多股票SHAP分析完成")
    print(f"{'='*70}")

    return df_all, df_cat, df_consistency


if __name__ == "__main__":
    run_multi_stock_shap()
