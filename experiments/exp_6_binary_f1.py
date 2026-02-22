"""
实验6: 事后二分类F1分析

方法：加载三分类模型的预测概率，去掉argmax预测为"中性"(class=1)的样本，
仅对预测为"涨"(class=2)或"跌"(class=0)的样本计算二分类F1。

标签映射: 0=下跌, 1=中性, 2=上涨
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
from sklearn.metrics import f1_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

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

FEATURE_COLS = ALL_FEATURES
SEQ_LEN = 60
N_FEATURES = len(FEATURE_COLS)

MODEL_CONFIGS = [
    ('cnn_lstm', 'CNN-LSTM', 'pt', True),
    ('lstm', 'LSTM', 'pt', True),
    ('gru', 'GRU', 'pt', True),
    ('transformer', 'Transformer', 'pt', True),
    ('xgboost', 'XGBoost', 'pkl', False),
    ('random_forest', 'RandomForest', 'pkl', False),
    ('logistic_regression', 'LogisticRegression', 'pkl', False),
]


def load_dataset(code: str):
    path = DATA_PROCESSED.parent / 'datasets' / f"dataset_{code.replace('.', '_')}_1M.pkl"
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def get_test_xy(dataset):
    ds = dataset['test']
    if hasattr(ds, 'X'):
        X, y = ds.X, ds.y
    else:
        X, y = ds[0], ds[1]
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if isinstance(y, torch.Tensor):
        y = y.numpy()
    y = np.array(y, dtype=np.float32)
    if y.min() < 0:
        y = y + 1
    return np.array(X, dtype=np.float32), y.astype(np.int64)


def predict_deep(model_name, code, X_test):
    model_path = MODELS_DIR / model_name / f'model_{code.replace(".", "_")}_1M.pt'
    if not model_path.exists():
        return None

    device = torch.device('cpu')
    model = create_model(model_name, input_dim=N_FEATURES, seq_len=SEQ_LEN)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    X_tensor = torch.FloatTensor(X_test)
    all_probs = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.numpy())

    return np.concatenate(all_probs, axis=0)


def predict_sklearn(model_name, code, X_test):
    model_path = MODELS_DIR / model_name / f'model_{code.replace(".", "_")}_1M.pkl'
    if not model_path.exists():
        return None

    with open(model_path, 'rb') as f:
        wrapper = pickle.load(f)
    model = wrapper.model if hasattr(wrapper, 'model') else wrapper

    X_flat = X_test.reshape(X_test.shape[0], -1)
    probs = model.predict_proba(X_flat)
    return probs


def binary_analysis(y_true, probs, model_label):
    """去掉argmax预测为中性(1)的样本，计算二分类指标"""
    preds_3class = np.argmax(probs, axis=1)
    non_neutral_mask = preds_3class != 1

    n_total = len(y_true)
    n_non_neutral = non_neutral_mask.sum()
    coverage = n_non_neutral / n_total

    if n_non_neutral == 0:
        return None

    y_filtered = y_true[non_neutral_mask]
    preds_filtered = preds_3class[non_neutral_mask]

    binary_true = (y_filtered == 2).astype(int)
    binary_pred = (preds_filtered == 2).astype(int)

    f1_bin = f1_score(binary_true, binary_pred, average='binary', zero_division=0)
    acc_bin = accuracy_score(binary_true, binary_pred)

    true_up = (y_filtered == 2).sum()
    true_down = (y_filtered == 0).sum()
    true_neutral = (y_filtered == 1).sum()

    return {
        'model': model_label,
        'total_samples': n_total,
        'non_neutral_preds': int(n_non_neutral),
        'coverage': coverage,
        'binary_f1': f1_bin,
        'binary_accuracy': acc_bin,
        'true_up_in_filtered': int(true_up),
        'true_down_in_filtered': int(true_down),
        'true_neutral_in_filtered': int(true_neutral),
    }


def confidence_stratified_analysis(y_true, probs, model_label, topk_ratios=[0.1, 0.2, 0.3]):
    """
    置信度分层分析：按预测概率最大值降序排列，
    只评估Top-K%最有把握的预测的准确率和F1。
    """
    preds = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)

    sorted_idx = np.argsort(-max_probs)

    n = len(y_true)
    results = []

    full_f1 = f1_score(y_true, preds, average='macro')
    full_acc = accuracy_score(y_true, preds)
    results.append({
        'model': model_label,
        'topk': '100%（全量）',
        'n_samples': n,
        'accuracy': full_acc,
        'f1_macro': full_f1,
        'avg_confidence': float(max_probs.mean()),
    })

    for ratio in topk_ratios:
        k = int(n * ratio)
        if k == 0:
            continue
        top_idx = sorted_idx[:k]
        y_top = y_true[top_idx]
        preds_top = preds[top_idx]
        conf_top = max_probs[top_idx]

        acc = accuracy_score(y_top, preds_top)
        f1 = f1_score(y_top, preds_top, average='macro', zero_division=0)

        results.append({
            'model': model_label,
            'topk': f'Top-{int(ratio*100)}%',
            'n_samples': k,
            'accuracy': acc,
            'f1_macro': f1,
            'avg_confidence': float(conf_top.mean()),
        })

    margin = np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]
    margin_sorted_idx = np.argsort(-margin)

    for ratio in topk_ratios:
        k = int(n * ratio)
        if k == 0:
            continue
        top_idx = margin_sorted_idx[:k]
        y_top = y_true[top_idx]
        preds_top = preds[top_idx]

        acc = accuracy_score(y_top, preds_top)
        f1 = f1_score(y_top, preds_top, average='macro', zero_division=0)

        results.append({
            'model': model_label,
            'topk': f'Top-{int(ratio*100)}%（margin）',
            'n_samples': k,
            'accuracy': acc,
            'f1_macro': f1,
            'avg_confidence': float(margin[top_idx].mean()),
        })

    return results


def run_single_stock(code='HK.00700'):
    stock_name = {c: n for c, n, _ in STOCK_LIST}.get(code, code)
    print(f"\n{'='*70}")
    print(f"置信度分层分析: {stock_name} ({code})")
    print(f"{'='*70}")

    dataset = load_dataset(code)
    if dataset is None:
        print(f"[ERROR] 数据集不存在: {code}")
        return

    X_test, y_test = get_test_xy(dataset)
    print(f"测试集: {len(y_test)} 样本")
    print(f"标签分布: 下跌={int((y_test==0).sum())}({(y_test==0).mean()*100:.1f}%), "
          f"中性={int((y_test==1).sum())}({(y_test==1).mean()*100:.1f}%), "
          f"上涨={int((y_test==2).sum())}({(y_test==2).mean()*100:.1f}%)")

    all_conf_results = []

    for model_name, model_label, ext, is_deep in MODEL_CONFIGS:
        print(f"\n  [{model_label}]...", end=' ')
        try:
            if is_deep:
                probs = predict_deep(model_name, code, X_test)
            else:
                probs = predict_sklearn(model_name, code, X_test)

            if probs is None:
                print("模型文件不存在")
                continue

            conf_results = confidence_stratified_analysis(
                y_test, probs, model_label, topk_ratios=[0.10, 0.20, 0.30]
            )
            all_conf_results.extend(conf_results)

            full = [r for r in conf_results if r['topk'] == '100%（全量）'][0]
            top10 = [r for r in conf_results if r['topk'] == 'Top-10%'][0]
            top20 = [r for r in conf_results if r['topk'] == 'Top-20%'][0]
            top30 = [r for r in conf_results if r['topk'] == 'Top-30%'][0]

            print(f"全量Acc={full['accuracy']:.3f}/F1={full['f1_macro']:.3f} | "
                  f"Top10%Acc={top10['accuracy']:.3f}/F1={top10['f1_macro']:.3f} | "
                  f"Top20%Acc={top20['accuracy']:.3f}/F1={top20['f1_macro']:.3f} | "
                  f"Top30%Acc={top30['accuracy']:.3f}/F1={top30['f1_macro']:.3f}")

        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()

    if all_conf_results:
        df_conf = pd.DataFrame(all_conf_results)

        print(f"\n{'='*70}")
        print("置信度分层结果汇总")
        print(f"{'='*70}")
        pivot = df_conf[~df_conf['topk'].str.contains('margin')].pivot_table(
            index='model', columns='topk',
            values=['accuracy', 'f1_macro'], aggfunc='first'
        )
        print(pivot.to_string())

        out_path = get_output_path('table_6_confidence_stratified', 'csv')
        df_conf.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存: {out_path}")

    return all_conf_results


def compute_random_baseline_f1(y_true, n_sim=1000):
    """计算实际标签分布下的随机分类器F1-macro期望值"""
    classes, counts = np.unique(y_true, return_counts=True)
    priors = counts / counts.sum()
    n = len(y_true)
    f1s = []
    rng = np.random.RandomState(42)
    for _ in range(n_sim):
        preds = rng.choice(classes, size=n, p=priors)
        f1s.append(f1_score(y_true, preds, average='macro', zero_division=0))
    return float(np.mean(f1s))


def run_all_stocks():
    """全部10只股票的置信度分层分析"""
    set_seed()

    all_stock_results = []
    baseline_rows = []

    for i, (code, name, sector) in enumerate(STOCK_LIST):
        print(f"\n{'='*70}")
        print(f"[{i+1}/10] {name} ({code}) - {sector}")
        print(f"{'='*70}")

        dataset = load_dataset(code)
        if dataset is None:
            print(f"[SKIP] 数据集不存在")
            continue

        X_test, y_test = get_test_xy(dataset)
        n = len(y_test)
        dist = {0: (y_test==0).sum(), 1: (y_test==1).sum(), 2: (y_test==2).sum()}
        print(f"测试集: {n} 样本  下跌={dist[0]}({dist[0]/n*100:.1f}%) "
              f"中性={dist[1]}({dist[1]/n*100:.1f}%) 上涨={dist[2]}({dist[2]/n*100:.1f}%)")

        rand_f1 = compute_random_baseline_f1(y_test)
        print(f"随机基线F1-macro: {rand_f1:.3f}")
        baseline_rows.append({
            '股票代码': code, '股票名称': name, '行业': sector,
            '下跌%': dist[0]/n, '中性%': dist[1]/n, '上涨%': dist[2]/n,
            '随机基线F1': rand_f1,
        })

        for model_name, model_label, ext, is_deep in MODEL_CONFIGS:
            print(f"  [{model_label}]...", end=' ')
            try:
                if is_deep:
                    probs = predict_deep(model_name, code, X_test)
                else:
                    probs = predict_sklearn(model_name, code, X_test)

                if probs is None:
                    print("跳过")
                    continue

                conf_results = confidence_stratified_analysis(
                    y_test, probs, model_label, topk_ratios=[0.10, 0.20, 0.30]
                )
                for r in conf_results:
                    r['股票代码'] = code
                    r['股票名称'] = name
                    r['随机基线F1'] = rand_f1
                all_stock_results.extend(conf_results)

                full = [r for r in conf_results if r['topk'] == '100%（全量）'][0]
                top10 = [r for r in conf_results if r['topk'] == 'Top-10%'][0]
                beat = '>' if full['f1_macro'] > rand_f1 else '<='
                beat10 = '>' if top10['f1_macro'] > rand_f1 else '<='
                print(f"全量F1={full['f1_macro']:.3f}{beat}rand({rand_f1:.3f}) | "
                      f"Top10%F1={top10['f1_macro']:.3f}{beat10}rand")

            except Exception as e:
                print(f"错误: {e}")

    if not all_stock_results:
        return

    df_all = pd.DataFrame(all_stock_results)
    df_base = pd.DataFrame(baseline_rows)

    # 保存详细数据
    df_all.to_csv(get_output_path('table_6_confidence_all_stocks', 'csv'),
                  index=False, encoding='utf-8-sig')
    df_base.to_csv(get_output_path('table_6_random_baselines', 'csv'),
                   index=False, encoding='utf-8-sig')

    # 汇总: 每个模型在各topk层级的跨股票均值
    print(f"\n{'='*70}")
    print("跨10只股票汇总（F1-macro均值±标准差）")
    print(f"{'='*70}")

    non_margin = df_all[~df_all['topk'].str.contains('margin')]
    summary = non_margin.groupby(['model', 'topk']).agg(
        f1_mean=('f1_macro', 'mean'),
        f1_std=('f1_macro', 'std'),
        acc_mean=('accuracy', 'mean'),
    ).reset_index()

    avg_rand = df_base['随机基线F1'].mean()
    print(f"\n平均随机基线F1-macro: {avg_rand:.3f}")
    print(f"（若均衡分布则为0.333，实际分布下为{avg_rand:.3f}）\n")

    for topk in ['100%（全量）', 'Top-10%', 'Top-20%', 'Top-30%']:
        sub = summary[summary['topk'] == topk].sort_values('f1_mean', ascending=False)
        print(f"\n--- {topk} ---")
        for _, row in sub.iterrows():
            beat = '***' if row['f1_mean'] > avg_rand else ''
            print(f"  {row['model']:20s}: F1={row['f1_mean']:.3f}±{row['f1_std']:.3f}  "
                  f"Acc={row['acc_mean']:.3f} {beat}")

    # 逐模型: Top-10% vs 全量的提升
    print(f"\n{'='*70}")
    print("Top-10% vs 全量 F1提升（跨股票均值）")
    print(f"{'='*70}")
    for model_label in ['CNN-LSTM', 'LSTM', 'GRU', 'Transformer', 'XGBoost',
                         'RandomForest', 'LogisticRegression']:
        full_vals = non_margin[(non_margin['model'] == model_label) &
                               (non_margin['topk'] == '100%（全量）')]['f1_macro']
        top10_vals = non_margin[(non_margin['model'] == model_label) &
                                 (non_margin['topk'] == 'Top-10%')]['f1_macro']
        if len(full_vals) > 0 and len(top10_vals) > 0:
            full_m = full_vals.mean()
            top10_m = top10_vals.mean()
            lift = (top10_m - full_m) / full_m * 100 if full_m > 0 else 0
            print(f"  {model_label:20s}: 全量={full_m:.3f} → Top-10%={top10_m:.3f}  "
                  f"提升{lift:+.1f}%")

    summary.to_csv(get_output_path('table_6_confidence_summary', 'csv'),
                   index=False, encoding='utf-8-sig')
    print(f"\n汇总结果已保存")


if __name__ == "__main__":
    run_all_stocks()
