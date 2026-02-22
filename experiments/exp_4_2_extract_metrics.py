# -*- coding: utf-8 -*-
"""
实验 4.2: 从已训练模型中提取性能指标并汇总

从所有checkpoint中提取保存的验证/测试指标，生成完整的表4.2-1

输出:
- tables/table_4_2_1_model_comparison.csv: 完整模型性能对比表
- figures/fig_4_2_1_model_comparison.png: 模型性能可视化图
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from exp_config import *
import pandas as pd
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt

# 导入sklearn模型包装器（用于正确反序列化pickle文件）
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "kline_model_trainer", 
        PROJECT_ROOT / "scripts" / "08_model_trainer.py"
    )
    trainer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer_module)
    SklearnModelWrapper = trainer_module.SklearnModelWrapper
    SKLEARN_WRAPPER_IMPORTED = True
except Exception as e:
    print(f"[WARN] Could not import SklearnModelWrapper: {e}")
    SKLEARN_WRAPPER_IMPORTED = False

# 模型名称映射
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

# 模型类型
PYTORCH_MODELS = ['lstm', 'gru', 'cnn_lstm', 'transformer', 'pv_transformer', 'multi_scale']
SKLEARN_MODELS = ['logistic_regression', 'random_forest', 'xgboost']


def extract_pytorch_metrics(model_path: Path) -> dict:
    """从PyTorch模型checkpoint中提取指标"""
    try:
        ckpt = torch.load(model_path, map_location='cpu')
        history = ckpt.get('history', {})
        
        # 检查训练是否正常完成（train_loss不为nan，且epochs > 1）
        train_loss = history.get('train_loss', [])
        epochs_trained = len(train_loss)
        
        # 如果train_loss为空或包含nan，标记为无效训练
        is_valid_training = True
        has_nan_loss = False
        
        if epochs_trained == 0:
            is_valid_training = False
        elif train_loss:
            # 检查是否有nan loss
            for loss in train_loss:
                if loss is not None and np.isnan(loss):
                    has_nan_loss = True
                    break
            if has_nan_loss:
                is_valid_training = False
        
        # 提取最佳指标
        metrics = {
            'best_val_f1': ckpt.get('best_val_f1', None),
            'final_val_acc': history.get('val_acc', [None])[-1] if history.get('val_acc') else None,
            'final_val_f1': history.get('val_f1', [None])[-1] if history.get('val_f1') else None,
            'final_train_loss': history.get('train_loss', [None])[-1] if history.get('train_loss') else None,
            'epochs_trained': epochs_trained,
            'is_valid_training': is_valid_training,
        }
        
        # 如果有测试指标
        if 'test_metrics' in ckpt:
            test = ckpt['test_metrics']
            metrics['test_acc'] = test.get('accuracy', None)
            metrics['test_f1'] = test.get('f1_macro', None)
            metrics['test_auc'] = test.get('auc', None)
        
        return metrics
    except Exception as e:
        print(f"  [ERROR] {model_path}: {e}")
        return {}


def extract_sklearn_metrics(model_path: Path) -> dict:
    """从sklearn模型文件中提取指标"""
    if not SKLEARN_WRAPPER_IMPORTED:
        print(f"  [SKIP] sklearn wrapper not imported")
        return {}
    
    try:
        # 需要将SklearnModelWrapper注入到__main__模块的命名空间
        import __main__
        __main__.SklearnModelWrapper = SklearnModelWrapper
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        # 检查是否是SklearnModelWrapper对象
        if hasattr(data, 'model') and hasattr(data, 'model_type'):
            # 这是SklearnModelWrapper，没有直接保存指标
            # 需要标记为需要重新评估（但模型本身是有效的）
            return {'model_loaded': True}
        elif isinstance(data, dict):
            metrics = data.get('metrics', {})
            return {
                'test_acc': metrics.get('accuracy', None),
                'test_f1': metrics.get('f1_macro', None),
                'test_auc': metrics.get('auc', None),
            }
        else:
            return {'model_loaded': True}
    except Exception as e:
        print(f"  [ERROR] {model_path}: {e}")
        return {}


def run_extraction():
    """从所有模型中提取指标"""
    print("=" * 70)
    print("  从已训练模型中提取性能指标")
    print("=" * 70)
    
    results = []
    
    # 遍历所有股票
    for code, name, sector in STOCK_LIST:
        code_str = code.replace('.', '_')
        print(f"\n处理 {code} {name} ({sector})...")
        
        # PyTorch模型
        for model_name in PYTORCH_MODELS:
            model_path = MODELS_DIR / model_name / f"model_{code_str}_1M.pt"
            
            if not model_path.exists():
                continue
            
            metrics = extract_pytorch_metrics(model_path)
            
            if metrics:
                # 检查训练是否有效（特别是multi_scale模型可能因NaN loss导致训练失败）
                is_valid = metrics.get('is_valid_training', True)
                epochs = metrics.get('epochs_trained', 0)
                
                if not is_valid:
                    print(f"  [SKIP] {model_name}: 训练异常 (epochs={epochs}, 含nan_loss)")
                    continue
                
                results.append({
                    '股票代码': code,
                    '股票名称': name,
                    '行业': sector,
                    '模型': MODEL_NAME_MAP.get(model_name, model_name),
                    'Accuracy': metrics.get('final_val_acc') or metrics.get('test_acc'),
                    'F1-macro': metrics.get('best_val_f1') or metrics.get('final_val_f1') or metrics.get('test_f1'),
                    'AUC': metrics.get('test_auc', 0.5),
                    '数据来源': 'val' if metrics.get('final_val_acc') else 'test',
                })
                print(f"  [OK] {model_name}: F1={results[-1]['F1-macro']:.4f}" if results[-1]['F1-macro'] else f"  [OK] {model_name}")
        
        # sklearn模型
        for model_name in SKLEARN_MODELS:
            model_path = MODELS_DIR / model_name / f"model_{code_str}_1M.pkl"
            
            if not model_path.exists():
                continue
            
            metrics = extract_sklearn_metrics(model_path)
            
            if metrics and metrics.get('test_acc'):
                results.append({
                    '股票代码': code,
                    '股票名称': name,
                    '行业': sector,
                    '模型': MODEL_NAME_MAP.get(model_name, model_name),
                    'Accuracy': metrics.get('test_acc'),
                    'F1-macro': metrics.get('test_f1'),
                    'AUC': metrics.get('test_auc', 0.5),
                    '数据来源': 'test',
                })
                print(f"  [OK] {model_name}: Acc={results[-1]['Accuracy']:.4f}" if results[-1]['Accuracy'] else f"  [OK] {model_name}")
            else:
                # sklearn模型没有保存指标，标记为需要重新评估
                results.append({
                    '股票代码': code,
                    '股票名称': name,
                    '行业': sector,
                    '模型': MODEL_NAME_MAP.get(model_name, model_name),
                    'Accuracy': None,
                    'F1-macro': None,
                    'AUC': None,
                    '数据来源': 'need_eval',
                })
                print(f"  ? {model_name}: 需要重新评估")
    
    if not results:
        print("\n[ERROR] 未找到任何模型文件")
        return None
    
    df = pd.DataFrame(results)
    
    print(f"\n总计提取: {len(df)} 条记录")
    print(f"有效指标: {df['F1-macro'].notna().sum()} 条")
    
    return df


def calculate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """计算各模型的汇总统计"""
    # 只使用有效数据
    df_valid = df[df['F1-macro'].notna()].copy()
    
    if df_valid.empty:
        return None
    
    # 按模型分组计算均值和标准差
    summary = df_valid.groupby('模型').agg({
        'Accuracy': ['mean', 'std', 'count'],
        'F1-macro': ['mean', 'std'],
        'AUC': ['mean', 'std'],
    }).round(4)
    
    # 展平列名
    summary.columns = ['Acc_mean', 'Acc_std', 'n_stocks', 'F1_mean', 'F1_std', 'AUC_mean', 'AUC_std']
    summary = summary.reset_index()
    
    # 按F1排序
    summary = summary.sort_values('F1_mean', ascending=False)
    
    return summary


def create_comparison_figure(summary: pd.DataFrame):
    """创建模型性能对比图"""
    setup_plot()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 模型顺序（按F1排序）
    models = summary['模型'].tolist()
    
    # 颜色：突出本研究模型
    colors = []
    for m in models:
        if 'PV-Transformer' in m:
            colors.append('#d62728')  # 红色 - 本研究模型
        elif m in ['Transformer', 'LSTM', 'GRU', 'CNN-LSTM']:
            colors.append('#1f77b4')  # 蓝色 - 深度学习基线
        else:
            colors.append('#7f7f7f')  # 灰色 - 传统ML基线
    
    # 图1: F1-macro对比
    ax1 = axes[0]
    bars1 = ax1.barh(models, summary['F1_mean'], xerr=summary['F1_std'], 
                     color=colors, capsize=3, alpha=0.8)
    ax1.set_xlabel('F1-macro', fontsize=12)
    ax1.set_title('(a) 模型F1-macro对比', fontsize=14)
    ax1.set_xlim(0, 0.5)
    
    # 添加数值标签
    for bar, val in zip(bars1, summary['F1_mean']):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=10)
    
    # 图2: Accuracy对比
    ax2 = axes[1]
    bars2 = ax2.barh(models, summary['Acc_mean'], xerr=summary['Acc_std'],
                     color=colors, capsize=3, alpha=0.8)
    ax2.set_xlabel('Accuracy', fontsize=12)
    ax2.set_title('(b) 模型准确率对比', fontsize=14)
    ax2.set_xlim(0, 1.0)
    
    for bar, val in zip(bars2, summary['Acc_mean']):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', label='本研究模型'),
        Patch(facecolor='#1f77b4', label='深度学习基线'),
        Patch(facecolor='#7f7f7f', label='传统ML基线'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=11)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # 保存
    output_path = FIGURES_DIR / 'fig_4_2_1_model_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存: {output_path}")
    
    plt.close()


def load_sklearn_eval_metrics():
    """从sklearn评估结果文件中加载指标"""
    eval_path = TABLES_DIR / 'table_4_2_sklearn_eval.csv'
    
    if not eval_path.exists():
        print(f"[WARN] sklearn评估文件不存在: {eval_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(eval_path)
    
    # 转换为统一格式
    df['数据来源'] = 'db_eval'
    
    print(f"\n从sklearn评估文件加载指标: {len(df)} 条")
    return df


def main():
    """主函数"""
    set_seed()
    
    # 1. 提取深度学习模型指标
    df_all = run_extraction()
    if df_all is None:
        return
    
    # 2. 加载sklearn模型评估指标
    df_sklearn = load_sklearn_eval_metrics()
    
    # 3. 合并数据（只保留有效的指标）
    df_dl = df_all[df_all['F1-macro'].notna()].copy()  # 深度学习模型的有效数据
    
    if not df_sklearn.empty:
        # 确保列一致
        common_cols = ['股票代码', '股票名称', '行业', '模型', 'Accuracy', 'F1-macro', 'AUC', '数据来源']
        df_dl = df_dl[common_cols]
        df_sklearn = df_sklearn[common_cols]
        df_combined = pd.concat([df_dl, df_sklearn], ignore_index=True)
    else:
        df_combined = df_dl
    
    # 4. 保存详细结果
    detail_path = TABLES_DIR / 'table_4_2_1_model_comparison_detail.csv'
    df_combined.to_csv(detail_path, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存: {detail_path}")
    
    # 5. 计算汇总统计
    summary = calculate_summary(df_combined)
    
    if summary is not None:
        # 保存汇总表
        summary_path = TABLES_DIR / 'table_4_2_1_model_comparison.csv'
        summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"汇总表已保存: {summary_path}")
        
        # 打印汇总表
        print("\n" + "=" * 70)
        print("  表 4.2-1: 模型性能汇总（所有股票平均）")
        print("=" * 70)
        print(summary.to_string(index=False))
        
        # 6. 生成可视化
        create_comparison_figure(summary)
    
    print("\n" + "=" * 70)
    print("  实验完成")
    print("=" * 70)
    
    # 打印数据完整性说明
    print("\n数据完整性说明:")
    for model in summary['模型'].tolist():
        n = int(summary[summary['模型'] == model]['n_stocks'].values[0])
        if n < 10:
            print(f"  * {model}: 仅有 {n} 只股票的数据")
    
    print("\n注意: 部分模型数据不完整的原因:")
    print("  - sklearn模型: pickle文件未保存评估指标，需要数据库重新评估")
    print("  - PV-Transformer+LSF: 部分股票训练时出现NaN loss，已排除异常数据")


if __name__ == "__main__":
    main()
