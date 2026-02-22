"""
实验 4.2.2: 全模型性能评估（含AUC计算）

对应论文:
- 表 4.2-1 / 4.2-2: 模型性能汇总（所有9种模型 + AUC）

输出:
- table_4_2_2_model_comparison.csv （每只股票×每个模型的详细指标）
- table_4_2_2_model_summary.csv   （按模型汇总的均值±标准差）

修改说明: 加载已训练模型，在测试集上计算 Accuracy / F1 / AUC
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from exp_config import *
import pandas as pd
import numpy as np
import pickle
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ============================================================
# 安全导入 13b_kline_model_trainer（防止 exec_module 关闭 stdout）
# ============================================================
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "kline_model_trainer",
    PROJECT_ROOT / "scripts" / "08_model_trainer.py"
)
trainer_module = importlib.util.module_from_spec(spec)

MODELS_IMPORTED = False
try:
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    try:
        spec.loader.exec_module(trainer_module)
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr

    LSTMBaseline = trainer_module.LSTMBaseline
    GRUBaseline = trainer_module.GRUBaseline
    CNNLSTMBaseline = trainer_module.CNNLSTMBaseline
    TransformerBaseline = trainer_module.TransformerBaseline
    PVTransformer = trainer_module.PVTransformer
    MultiScalePVTransformer = trainer_module.MultiScalePVTransformer
    SklearnModelWrapper = trainer_module.SklearnModelWrapper
    create_model = trainer_module.create_model
    PRICE_RELATED = trainer_module.PRICE_RELATED
    VOLUME_RELATED = trainer_module.VOLUME_RELATED
    is_multi_scale_dataset = trainer_module.is_multi_scale_dataset
    create_multi_scale_dataloaders = trainer_module.create_multi_scale_dataloaders
    MODELS_IMPORTED = True
    print("[OK] 模型类导入成功")
except Exception as e:
    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr
    print(f"[ERROR] 模型导入失败: {e}")

MODEL_NAME_MAP = {
    'lstm': 'LSTM',
    'gru': 'GRU',
    'cnn_lstm': 'CNN-LSTM',
    'transformer': 'Transformer',
    'pv_transformer': 'PV-Transformer',
    'multi_scale': 'PV-Transformer+LSF',
    'logistic_regression': 'LogisticRegression',
    'random_forest': 'RandomForest',
    'xgboost': 'XGBoost',
}


# ============================================================
# 数据加载
# ============================================================

def load_dataset(code: str, ktype: str = '1M') -> Optional[Dict]:
    dataset_path = DATA_PROCESSED.parent / 'datasets' / f"dataset_{code.replace('.', '_')}_{ktype}.pkl"
    if dataset_path.exists():
        with open(dataset_path, 'rb') as f:
            return pickle.load(f)
    return None


def load_multi_scale_dataset(code: str) -> Optional[Dict]:
    dataset_path = DATA_PROCESSED.parent / 'datasets' / f"dataset_{code.replace('.', '_')}_multi_scale.pkl"
    if dataset_path.exists():
        with open(dataset_path, 'rb') as f:
            return pickle.load(f)
    return None


def create_test_dataloader(dataset: Dict, batch_size: int = 64) -> Optional[DataLoader]:
    if 'test' not in dataset:
        return None

    ds = dataset['test']
    if hasattr(ds, 'X'):
        X, y = ds.X, ds.y
    else:
        X, y = ds

    if isinstance(y, torch.Tensor):
        y_np = y.numpy()
    else:
        y_np = np.array(y)

    if y_np.min() == -1:
        y_np = y_np + 1

    tensor_ds = TensorDataset(
        torch.FloatTensor(X) if not isinstance(X, torch.Tensor) else X,
        torch.LongTensor(y_np)
    )
    return DataLoader(tensor_ds, batch_size=batch_size, shuffle=False)


# ============================================================
# 模型加载
# ============================================================

def load_pytorch_model(model_path: Path, device: torch.device):
    if not model_path.exists():
        return None
    return torch.load(model_path, map_location=device)


def load_sklearn_model(model_path: Path):
    if not model_path.exists():
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def recreate_model(model_name: str, input_dim: int, seq_len: int, **kwargs):
    if not MODELS_IMPORTED:
        return None
    try:
        return create_model(model_name, input_dim=input_dim, seq_len=seq_len, **kwargs)
    except Exception as e:
        print(f"  [ERROR] 创建模型 {model_name} 失败: {e}")
        return None


# ============================================================
# 评估函数
# ============================================================

def compute_metrics(all_labels: np.ndarray, all_preds: np.ndarray,
                    all_probs: np.ndarray) -> Dict:
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
    }
    try:
        if len(np.unique(all_labels)) > 2:
            metrics['auc'] = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        else:
            metrics['auc'] = roc_auc_score(all_labels, all_probs[:, 1])
    except Exception as e:
        print(f"    [WARN] AUC计算失败: {e}")
        metrics['auc'] = np.nan
    return metrics


def evaluate_single_scale_model(model: nn.Module, dataloader: DataLoader,
                                device: torch.device) -> Dict:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    price_dim = len(PRICE_RELATED)

    with torch.no_grad():
        for batch in dataloader:
            X, y = batch[0].to(device), batch[1].to(device)

            if hasattr(model, 'model_name') and 'pv' in model.model_name:
                logits = model(X[:, :, :price_dim], X[:, :, price_dim:])
            else:
                logits = model(X)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return compute_metrics(np.array(all_labels), np.array(all_preds),
                           np.array(all_probs))


def evaluate_multi_scale_model(model: nn.Module, dataloader: DataLoader,
                               device: torch.device) -> Dict:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    price_dim = len(PRICE_RELATED)

    with torch.no_grad():
        for batch in dataloader:
            scale_data_raw = batch[0]
            y = batch[1].to(device)
            scale_data = {}
            for scale in model.scale_names:
                if scale in scale_data_raw:
                    x = scale_data_raw[scale].to(device)
                    scale_data[scale] = (x[:, :, :price_dim], x[:, :, price_dim:])

            logits = model(scale_data)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return compute_metrics(np.array(all_labels), np.array(all_preds),
                           np.array(all_probs))


def evaluate_sklearn(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    if len(X_test.shape) == 3:
        X_flat = X_test.reshape(X_test.shape[0], -1)
    else:
        X_flat = X_test

    X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)
    X_flat = np.clip(X_flat, -1e6, 1e6)

    y_pred = model.predict(X_flat)

    all_probs = None
    try:
        all_probs = model.predict_proba(X_flat)
    except:
        pass

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
    }

    if all_probs is not None:
        try:
            if len(np.unique(y_test)) > 2:
                metrics['auc'] = roc_auc_score(y_test, all_probs, multi_class='ovr')
            else:
                metrics['auc'] = roc_auc_score(y_test, all_probs[:, 1])
        except Exception as e:
            print(f"    [WARN] sklearn AUC失败: {e}")
            metrics['auc'] = np.nan
    else:
        metrics['auc'] = np.nan

    return metrics


# ============================================================
# 主实验
# ============================================================

def run_experiment():
    if not MODELS_IMPORTED:
        print("[FATAL] 模型类未导入，无法运行")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    results = []

    single_scale_models = ['lstm', 'gru', 'cnn_lstm', 'transformer', 'pv_transformer']
    sklearn_models = ['logistic_regression', 'random_forest', 'xgboost']

    for code, name, sector in STOCK_LIST:
        code_str = code.replace('.', '_')
        print(f"\n{'='*60}")
        print(f"  {code} {name} ({sector})")
        print(f"{'='*60}")

        # --- 单尺度深度学习模型 ---
        dataset_1m = load_dataset(code)
        if dataset_1m is not None:
            test_loader = create_test_dataloader(dataset_1m)
            if test_loader is not None:
                sample_X, _ = next(iter(test_loader))
                seq_len, input_dim = sample_X.shape[1], sample_X.shape[2]

                for mname in single_scale_models:
                    model_path = MODELS_DIR / mname / f"model_{code_str}_1M.pt"
                    if not model_path.exists():
                        print(f"  [SKIP] {mname} 模型文件不存在")
                        continue

                    try:
                        checkpoint = load_pytorch_model(model_path, device)
                        model = recreate_model(mname, input_dim, seq_len)
                        if model is None:
                            continue
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model = model.to(device)

                        metrics = evaluate_single_scale_model(model, test_loader, device)
                        results.append({
                            '股票代码': code, '股票名称': name,
                            '模型': MODEL_NAME_MAP[mname],
                            'Accuracy': metrics['accuracy'],
                            'F1-macro': metrics['f1_macro'],
                            'F1-weighted': metrics['f1_weighted'],
                            'AUC': metrics['auc'],
                        })
                        print(f"  [OK] {mname}: Acc={metrics['accuracy']:.4f}  "
                              f"F1={metrics['f1_macro']:.4f}  AUC={metrics['auc']:.4f}")
                    except Exception as e:
                        print(f"  [ERROR] {mname}: {e}")

                # --- sklearn ---
                test_data = dataset_1m['test']
                if hasattr(test_data, 'X'):
                    X_test, y_test = test_data.X, test_data.y
                else:
                    X_test, y_test = test_data
                if hasattr(y_test, 'numpy'):
                    y_test = y_test.numpy()
                y_test = np.array(y_test)
                if y_test.min() == -1:
                    y_test = y_test + 1

                for mname in sklearn_models:
                    model_path = MODELS_DIR / mname / f"model_{code_str}_1M.pkl"
                    if not model_path.exists():
                        print(f"  [SKIP] {mname} 模型文件不存在")
                        continue

                    try:
                        model = load_sklearn_model(model_path)
                        metrics = evaluate_sklearn(model, X_test, y_test)
                        results.append({
                            '股票代码': code, '股票名称': name,
                            '模型': MODEL_NAME_MAP[mname],
                            'Accuracy': metrics['accuracy'],
                            'F1-macro': metrics['f1_macro'],
                            'F1-weighted': metrics['f1_weighted'],
                            'AUC': metrics['auc'],
                        })
                        print(f"  [OK] {mname}: Acc={metrics['accuracy']:.4f}  "
                              f"F1={metrics['f1_macro']:.4f}  AUC={metrics['auc']:.4f}")
                    except Exception as e:
                        print(f"  [ERROR] {mname}: {e}")
            else:
                print(f"  [SKIP] 1M数据集无test split")

            del dataset_1m
        else:
            print(f"  [SKIP] 1M数据集不存在")

        # --- multi_scale (PV-Transformer+LSF) ---
        ms_dataset = load_multi_scale_dataset(code)
        if ms_dataset is not None:
            try:
                _, _, ms_test_loader = create_multi_scale_dataloaders(ms_dataset, batch_size=64)
                if ms_test_loader is not None:
                    ms_model_path = MODELS_DIR / 'multi_scale' / f"model_{code_str}_1M.pt"
                    if ms_model_path.exists():
                        checkpoint = load_pytorch_model(ms_model_path, device)
                        price_dim = len(PRICE_RELATED)
                        volume_dim = len(VOLUME_RELATED)
                        model = MultiScalePVTransformer(
                            price_dim, volume_dim,
                            scale_seq_lens={'1M': 60, '5M': 24, '60M': 12},
                            num_classes=3
                        )
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model = model.to(device)

                        metrics = evaluate_multi_scale_model(model, ms_test_loader, device)
                        results.append({
                            '股票代码': code, '股票名称': name,
                            '模型': 'PV-Transformer+LSF',
                            'Accuracy': metrics['accuracy'],
                            'F1-macro': metrics['f1_macro'],
                            'F1-weighted': metrics['f1_weighted'],
                            'AUC': metrics['auc'],
                        })
                        print(f"  [OK] multi_scale: Acc={metrics['accuracy']:.4f}  "
                              f"F1={metrics['f1_macro']:.4f}  AUC={metrics['auc']:.4f}")
                    else:
                        print(f"  [SKIP] multi_scale 模型文件不存在")
            except Exception as e:
                print(f"  [ERROR] multi_scale: {e}")

            del ms_dataset
        else:
            print(f"  [SKIP] multi_scale 数据集不存在")

    if not results:
        print("\n[FATAL] 没有成功评估任何模型")
        return None

    # --- 保存详细结果 ---
    df = pd.DataFrame(results)
    detail_path = get_output_path('table_4_2_2_model_comparison', 'csv')
    df.to_csv(detail_path, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存: {detail_path}")

    # --- 汇总统计 ---
    summary_rows = []
    for model_label in MODEL_NAME_MAP.values():
        sub = df[df['模型'] == model_label]
        if sub.empty:
            continue
        summary_rows.append({
            '模型': model_label,
            'Acc_mean': sub['Accuracy'].mean(),
            'Acc_std': sub['Accuracy'].std(),
            'F1_mean': sub['F1-macro'].mean(),
            'F1_std': sub['F1-macro'].std(),
            'AUC_mean': sub['AUC'].mean(),
            'AUC_std': sub['AUC'].std(),
            'n_stocks': len(sub),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary.sort_values('F1_mean', ascending=False)

    summary_path = get_output_path('table_4_2_2_model_summary', 'csv')
    df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"汇总结果已保存: {summary_path}")

    print("\n" + "=" * 70)
    print("  模型性能汇总（均值 ± 标准差）")
    print("=" * 70)
    for _, row in df_summary.iterrows():
        print(f"  {row['模型']:24s}  Acc={row['Acc_mean']:.4f}±{row['Acc_std']:.4f}  "
              f"F1={row['F1_mean']:.4f}±{row['F1_std']:.4f}  "
              f"AUC={row['AUC_mean']:.4f}±{row['AUC_std']:.4f}  "
              f"(n={int(row['n_stocks'])})")

    return df_summary


def main():
    set_seed()
    return run_experiment()


if __name__ == "__main__":
    main()
