"""
实验 4.2.2: 深度学习模型性能评估

对应论文:
- 表 4.2-2: 深度学习模型性能汇总（LSTM、GRU、CNN-LSTM、Transformer、PV-Transformer）

输出:
- table_4_2_2_deep_models.csv

修改说明: 使用已训练好的模型进行评估，而不是重新训练
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from exp_config import *
import pandas as pd
import numpy as np
import pickle
from typing import Dict, Tuple, Optional

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 评估指标
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# 添加scripts路径以导入模型
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

# 导入13b的模型类（用于正确加载模型权重）
# 需要重命名导入，因为文件名以数字开头
import importlib.util
spec = importlib.util.spec_from_file_location("kline_model_trainer", PROJECT_ROOT / "scripts" / "13b_kline_model_trainer.py")
trainer_module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(trainer_module)
    LSTMBaseline = trainer_module.LSTMBaseline
    GRUBaseline = trainer_module.GRUBaseline
    CNNLSTMBaseline = trainer_module.CNNLSTMBaseline
    TransformerBaseline = trainer_module.TransformerBaseline
    PVTransformer = trainer_module.PVTransformer
    SklearnModelWrapper = trainer_module.SklearnModelWrapper
    create_model = trainer_module.create_model
    PRICE_RELATED = trainer_module.PRICE_RELATED
    VOLUME_RELATED = trainer_module.VOLUME_RELATED
    MODELS_IMPORTED = True
except Exception as e:
    print(f"Warning: Could not import models from 13b: {e}")
    MODELS_IMPORTED = False

# 模型名称映射（脚本名 -> 论文名）
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


def load_dataset(code: str, ktype: str = '1M') -> Optional[Dict]:
    """加载数据集"""
    dataset_path = DATA_PROCESSED.parent / 'datasets' / f"dataset_{code.replace('.', '_')}_{ktype}.pkl"
    
    if dataset_path.exists():
        with open(dataset_path, 'rb') as f:
            return pickle.load(f)
    return None


def create_test_dataloader(dataset: Dict, batch_size: int = 64) -> Optional[DataLoader]:
    """只创建测试集DataLoader"""
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
    
    # 确保标签从0开始
    if y_np.min() == -1:
        y_np = y_np + 1
    
    tensor_ds = TensorDataset(
        torch.FloatTensor(X) if not isinstance(X, torch.Tensor) else X,
        torch.LongTensor(y_np)
    )
    
    return DataLoader(tensor_ds, batch_size=batch_size, shuffle=False)


def load_pytorch_model(model_path: Path, device: torch.device):
    """加载PyTorch模型"""
    if not model_path.exists():
        return None
    
    checkpoint = torch.load(model_path, map_location=device)
    return checkpoint


def load_sklearn_model(model_path: Path):
    """加载sklearn模型"""
    if not model_path.exists():
        return None
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def evaluate_pytorch_model(model: nn.Module, dataloader: DataLoader, device: torch.device, 
                           model_name: str = '') -> Dict:
    """评估PyTorch模型"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    # 特征维度信息
    price_dim = 14  # PRICE_RELATED 特征数
    
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch[0].to(device), batch[1].to(device)
            
            # 根据模型类型处理输入
            if hasattr(model, 'model_name') and 'pv' in model.model_name:
                # PV-Transformer需要分离价格和成交量特征
                price_x = X[:, :, :price_dim]
                volume_x = X[:, :, price_dim:]
                logits = model(price_x, volume_x)
            else:
                logits = model(X)
            
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
    }
    
    # AUC（多分类）
    try:
        if len(np.unique(all_labels)) > 2:
            metrics['auc'] = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        else:
            metrics['auc'] = roc_auc_score(all_labels, all_probs[:, 1])
    except:
        metrics['auc'] = 0.5
    
    return metrics


def evaluate_sklearn_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """评估sklearn模型"""
    # 展平序列数据 (N, T, F) -> (N, T*F)
    if len(X_test.shape) == 3:
        X_flat = X_test.reshape(X_test.shape[0], -1)
    else:
        X_flat = X_test
    
    y_pred = model.predict(X_flat)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
    }
    
    # AUC
    try:
        y_proba = model.predict_proba(X_flat)
        if len(np.unique(y_test)) > 2:
            metrics['auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr')
        else:
            metrics['auc'] = roc_auc_score(y_test, y_proba[:, 1])
    except:
        metrics['auc'] = 0.5
    
    return metrics


def recreate_model(model_name: str, input_dim: int, seq_len: int):
    """重新创建模型结构以加载权重"""
    if not MODELS_IMPORTED:
        print(f"  [ERROR] 模型类未导入")
        return None
    
    try:
        return create_model(model_name, input_dim=input_dim, seq_len=seq_len)
    except Exception as e:
        print(f"  [ERROR] 创建模型失败: {e}")
        return None


def run_experiment():
    """运行实验"""
    log_experiment('4.2.2', '开始模型性能评估（使用已训练模型）')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_experiment('4.2.2', f'使用设备: {device}')
    
    results = []
    
    # 模型列表（PyTorch和sklearn）
    pytorch_models = ['lstm', 'gru', 'cnn_lstm', 'transformer', 'pv_transformer']
    sklearn_models = ['logistic_regression', 'random_forest', 'xgboost']
    
    # 只评估有数据的股票
    codes_to_eval = []
    for code, name, sector in STOCK_LIST:
        dataset = load_dataset(code)
        if dataset is not None:
            codes_to_eval.append((code, name, sector))
    
    if not codes_to_eval:
        log_experiment('4.2.2', '[ERROR] 没有找到任何数据集')
        return None
    
    log_experiment('4.2.2', f'找到 {len(codes_to_eval)} 只股票的数据集')
    
    for code, name, sector in codes_to_eval:
        log_experiment('4.2.2', f'评估 {code} {name}')
        
        dataset = load_dataset(code)
        test_loader = create_test_dataloader(dataset)
        
        if test_loader is None:
            continue
        
        # 获取维度
        sample_X, sample_y = next(iter(test_loader))
        seq_len, input_dim = sample_X.shape[1], sample_X.shape[2]
        
        # 获取测试数据（用于sklearn模型）
        test_data = dataset['test']
        if hasattr(test_data, 'X'):
            X_test, y_test = test_data.X, test_data.y
        else:
            X_test, y_test = test_data
        if hasattr(y_test, 'numpy'):
            y_test = y_test.numpy()
        y_test = np.array(y_test)
        if y_test.min() == -1:
            y_test = y_test + 1
        
        code_str = code.replace('.', '_')
        
        # 评估PyTorch模型
        for model_name in pytorch_models:
            model_path = MODELS_DIR / model_name / f"model_{code_str}_1M.pt"
            
            if not model_path.exists():
                log_experiment('4.2.2', f'  [SKIP] {model_name} 模型不存在')
                continue
            
            log_experiment('4.2.2', f'  评估 {model_name}...')
            
            # 加载模型
            checkpoint = load_pytorch_model(model_path, device)
            if checkpoint is None:
                continue
            
            # 重建模型结构
            model = recreate_model(model_name, input_dim, seq_len)
            if model is None:
                continue
            
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(device)
                
                metrics = evaluate_pytorch_model(model, test_loader, device, model_name)
                
                results.append({
                    '股票代码': code,
                    '股票名称': name,
                    '模型': MODEL_NAME_MAP.get(model_name, model_name),
                    'Accuracy': metrics['accuracy'],
                    'F1-macro': metrics['f1_macro'],
                    'F1-weighted': metrics['f1_weighted'],
                    'AUC': metrics['auc'],
                })
            except Exception as e:
                log_experiment('4.2.2', f'    [ERROR] {e}')
        
        # 评估sklearn模型
        for model_name in sklearn_models:
            model_path = MODELS_DIR / model_name / f"model_{code_str}_1M.pkl"
            
            if not model_path.exists():
                log_experiment('4.2.2', f'  [SKIP] {model_name} 模型不存在')
                continue
            
            log_experiment('4.2.2', f'  评估 {model_name}...')
            
            model = load_sklearn_model(model_path)
            if model is None:
                continue
            
            try:
                metrics = evaluate_sklearn_model(model, X_test, y_test)
                
                results.append({
                    '股票代码': code,
                    '股票名称': name,
                    '模型': MODEL_NAME_MAP.get(model_name, model_name),
                    'Accuracy': metrics['accuracy'],
                    'F1-macro': metrics['f1_macro'],
                    'F1-weighted': metrics['f1_weighted'],
                    'AUC': metrics['auc'],
                })
            except Exception as e:
                log_experiment('4.2.2', f'    [ERROR] {e}')
    
    if not results:
        log_experiment('4.2.2', '[ERROR] 没有成功评估任何模型')
        return None
    
    # 汇总结果
    df_results = pd.DataFrame(results)
    
    # 格式化数值
    for col in ['Accuracy', 'F1-macro', 'F1-weighted', 'AUC']:
        df_results[col] = df_results[col].apply(lambda x: f"{x:.4f}")
    
    # 保存
    output_path = get_output_path('table_4_2_2_model_comparison', 'csv')
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    log_experiment('4.2.2', f'结果已保存: {output_path}')
    
    print("\n" + "="*70)
    print("  表 4.2-2: 模型性能汇总")
    print("="*70)
    print(df_results.to_string(index=False))
    
    # 生成汇总统计
    print("\n" + "="*70)
    print("  模型平均性能")
    print("="*70)
    
    # 转回数值计算平均
    df_numeric = df_results.copy()
    for col in ['Accuracy', 'F1-macro', 'F1-weighted', 'AUC']:
        df_numeric[col] = df_numeric[col].astype(float)
    
    summary = df_numeric.groupby('模型')[['Accuracy', 'F1-macro', 'AUC']].mean()
    summary = summary.sort_values('F1-macro', ascending=False)
    print(summary.round(4).to_string())
    
    return df_results


def main():
    """主函数"""
    set_seed()
    return run_experiment()


if __name__ == "__main__":
    main()
