"""
SHAP特征归因分析模块
分析OFI预测模型的特征重要性和决策机制

功能:
1. 计算SHAP值
2. 特征重要性排序
3. 单样本归因分析
4. 特征交互效应
5. 生成可视化图表

使用方法:
    python 16_shap_analysis.py --model transformer --samples 100
    python 16_shap_analysis.py --model smart_trans --analyze-extreme
"""

import os
import sys
import io
import json
import argparse
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# 解决Windows编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 加载环境变量
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".apikey.env"
load_dotenv(env_path, override=True)

import numpy as np
import pandas as pd

# PyTorch
import torch
import torch.nn.functional as F

# SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[WARN] SHAP未安装，请运行: pip install shap")

# 可视化
try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ============================================================
# 配置
# ============================================================

# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 特征名称（与数据集对齐）
FEATURE_NAMES = [
    'spread_bps', 'return_pct',
    'ofi_l1', 'ofi_l5', 'ofi_l10', 'smart_ofi',
    'ofi_ma_10', 'ofi_std_10', 'ofi_zscore',
    'smart_ofi_ma_10', 'smart_ofi_std_10', 'smart_ofi_zscore',
    'return_ma_10', 'return_std_10',
    'bid_depth_5', 'ask_depth_5', 'depth_imbalance_5',
    'bid_depth_10', 'ask_depth_10', 'depth_imbalance_10',
    'buy_volume', 'sell_volume', 'trade_count', 'trade_imbalance',
    'corr_stock_index'
]

# 特征中文名
FEATURE_NAMES_CN = {
    'spread_bps': '买卖价差(bps)',
    'return_pct': '收益率(%)',
    'ofi_l1': 'OFI-L1',
    'ofi_l5': 'OFI-L5',
    'ofi_l10': 'OFI-L10',
    'smart_ofi': 'Smart-OFI',
    'ofi_ma_10': 'OFI均值',
    'ofi_std_10': 'OFI标准差',
    'ofi_zscore': 'OFI Z-score',
    'smart_ofi_ma_10': 'Smart-OFI均值',
    'smart_ofi_std_10': 'Smart-OFI标准差',
    'smart_ofi_zscore': 'Smart-OFI Z-score',
    'return_ma_10': '收益率均值',
    'return_std_10': '收益率标准差',
    'bid_depth_5': '买盘深度(5档)',
    'ask_depth_5': '卖盘深度(5档)',
    'depth_imbalance_5': '深度不平衡(5档)',
    'bid_depth_10': '买盘深度(10档)',
    'ask_depth_10': '卖盘深度(10档)',
    'depth_imbalance_10': '深度不平衡(10档)',
    'buy_volume': '买方成交量',
    'sell_volume': '卖方成交量',
    'trade_count': '成交笔数',
    'trade_imbalance': '成交不平衡',
    'corr_stock_index': '个股-指数相关性'
}

# 数据路径
DATA_PROCESSED = Path(os.getenv("DATA_PROCESSED", "data/processed"))
MODEL_DIR = Path("models")


# ============================================================
# 模型包装器
# ============================================================

class ModelWrapper:
    """
    模型包装器，用于SHAP分析
    
    SHAP需要一个可调用对象，输入为2D数组，输出为预测概率
    """
    
    def __init__(self, model, seq_len: int = 100, num_features: int = 25):
        self.model = model
        self.seq_len = seq_len
        self.num_features = num_features
        self.model.eval()
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        预测函数
        
        Args:
            X: 输入数组 (N, seq_len * num_features)
            
        Returns:
            概率数组 (N, num_classes)
        """
        # 重塑为3D
        X_3d = X.reshape(-1, self.seq_len, self.num_features)
        
        # 转为tensor
        X_tensor = torch.FloatTensor(X_3d).to(DEVICE)
        
        # 预测
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        
        return probs


class SequenceModelWrapper:
    """
    序列模型包装器，保持时间维度
    
    用于分析时间步的重要性
    """
    
    def __init__(self, model, target_class: int = 2):
        self.model = model
        self.target_class = target_class  # 2=上涨
        self.model.eval()
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: 输入数组 (N, seq_len, num_features)
            
        Returns:
            目标类别的概率 (N,)
        """
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        
        return probs[:, self.target_class]


# ============================================================
# SHAP分析器
# ============================================================

class SHAPAnalyzer:
    """
    SHAP特征归因分析器
    """
    
    def __init__(self, model, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names or FEATURE_NAMES
        
        self.explainer = None
        self.shap_values = None
        self.background_data = None
    
    def fit(self, X_background: np.ndarray, n_background: int = 100):
        """
        初始化SHAP解释器
        
        Args:
            X_background: 背景数据 (N, seq_len, num_features)
            n_background: 使用的背景样本数
        """
        if not HAS_SHAP:
            raise ImportError("SHAP未安装")
        
        # 选择背景样本
        if len(X_background) > n_background:
            indices = np.random.choice(len(X_background), n_background, replace=False)
            X_background = X_background[indices]
        
        self.background_data = X_background
        
        # 创建模型包装器
        seq_len, num_features = X_background.shape[1], X_background.shape[2]
        model_wrapper = ModelWrapper(self.model, seq_len, num_features)
        
        # 展平背景数据
        X_flat = X_background.reshape(len(X_background), -1)
        
        # 创建SHAP解释器（使用KernelExplainer，适用于任何模型）
        print("  创建SHAP解释器...")
        self.explainer = shap.KernelExplainer(model_wrapper, X_flat[:50])  # 使用较少的背景样本加速
        
        print(f"  背景样本: {len(X_flat)} 个")
    
    def explain(self, X: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """
        计算SHAP值
        
        Args:
            X: 待解释样本 (N, seq_len, num_features)
            n_samples: 解释的样本数
            
        Returns:
            SHAP值数组
        """
        if self.explainer is None:
            raise ValueError("请先调用fit()初始化解释器")
        
        # 选择样本
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
        
        # 展平
        X_flat = X.reshape(len(X), -1)
        
        print(f"  计算SHAP值... ({len(X_flat)} 样本)")
        self.shap_values = self.explainer.shap_values(X_flat)
        
        return self.shap_values
    
    def get_feature_importance(self, class_idx: int = 2) -> pd.DataFrame:
        """
        获取特征重要性排序
        
        Args:
            class_idx: 目标类别索引 (0=下跌, 1=平稳, 2=上涨)
            
        Returns:
            特征重要性DataFrame
        """
        if self.shap_values is None:
            raise ValueError("请先调用explain()计算SHAP值")
        
        # 获取目标类别的SHAP值
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[class_idx]
        else:
            shap_vals = self.shap_values
        
        # 重塑为 (N, seq_len, num_features)
        seq_len = self.background_data.shape[1]
        num_features = self.background_data.shape[2]
        shap_vals_3d = shap_vals.reshape(-1, seq_len, num_features)
        
        # 对时间维度取平均，得到每个特征的平均重要性
        # |SHAP|的平均值
        feature_importance = np.abs(shap_vals_3d).mean(axis=(0, 1))
        
        # 创建DataFrame
        df = pd.DataFrame({
            'feature': self.feature_names[:num_features],
            'feature_cn': [FEATURE_NAMES_CN.get(f, f) for f in self.feature_names[:num_features]],
            'importance': feature_importance,
            'importance_pct': feature_importance / feature_importance.sum() * 100
        })
        
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return df
    
    def get_temporal_importance(self, class_idx: int = 2) -> np.ndarray:
        """
        获取时间步重要性
        
        Returns:
            各时间步的平均SHAP值 (seq_len,)
        """
        if self.shap_values is None:
            raise ValueError("请先调用explain()计算SHAP值")
        
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[class_idx]
        else:
            shap_vals = self.shap_values
        
        seq_len = self.background_data.shape[1]
        num_features = self.background_data.shape[2]
        shap_vals_3d = shap_vals.reshape(-1, seq_len, num_features)
        
        # 对特征维度取平均
        temporal_importance = np.abs(shap_vals_3d).mean(axis=(0, 2))
        
        return temporal_importance


# ============================================================
# 简化版SHAP分析（基于梯度）
# ============================================================

class GradientSHAPAnalyzer:
    """
    基于梯度的SHAP分析器
    
    对于神经网络模型，使用GradientExplainer更高效
    """
    
    def __init__(self, model, feature_names: List[str] = None, num_features: int = None):
        self.model = model
        self.num_features = num_features  # 会在分析时动态确定
        # 如果提供了特征名，使用它；否则使用默认的或动态生成
        self._custom_feature_names = feature_names
        self.model.eval()
    
    def _get_feature_names(self, num_features: int) -> List[str]:
        """获取特征名称，支持动态数量"""
        if self._custom_feature_names is not None:
            if len(self._custom_feature_names) >= num_features:
                return self._custom_feature_names[:num_features]
            else:
                # 扩展特征名
                extended = self._custom_feature_names + [f'feature_{i}' for i in range(len(self._custom_feature_names), num_features)]
                return extended
        elif num_features <= len(FEATURE_NAMES):
            return FEATURE_NAMES[:num_features]
        else:
            # 动态生成特征名
            return FEATURE_NAMES + [f'feature_{i}' for i in range(len(FEATURE_NAMES), num_features)]
    
    def compute_gradients(self, X: np.ndarray, target_class: int = 2) -> np.ndarray:
        """
        计算输入梯度
        
        Args:
            X: 输入数据 (N, seq_len, num_features)
            target_class: 目标类别
            
        Returns:
            梯度数组 (N, seq_len, num_features)
        """
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        X_tensor.requires_grad = True
        
        # 前向传播
        logits = self.model(X_tensor)
        probs = F.softmax(logits, dim=1)
        
        # 目标类别的概率
        target_probs = probs[:, target_class]
        
        # 反向传播
        self.model.zero_grad()
        target_probs.sum().backward()
        
        # 获取梯度
        gradients = X_tensor.grad.cpu().numpy()
        
        return gradients
    
    def compute_integrated_gradients(
        self, 
        X: np.ndarray, 
        baseline: np.ndarray = None,
        target_class: int = 2,
        steps: int = 50
    ) -> np.ndarray:
        """
        计算积分梯度（Integrated Gradients）
        
        IG是一种更稳定的归因方法
        
        Args:
            X: 输入数据 (N, seq_len, num_features)
            baseline: 基线（默认为0）
            target_class: 目标类别
            steps: 积分步数
            
        Returns:
            积分梯度 (N, seq_len, num_features)
        """
        if baseline is None:
            baseline = np.zeros_like(X)
        
        # 创建插值路径
        alphas = np.linspace(0, 1, steps)
        
        integrated_grads = np.zeros_like(X)
        
        for alpha in alphas:
            X_interp = baseline + alpha * (X - baseline)
            grads = self.compute_gradients(X_interp, target_class)
            integrated_grads += grads
        
        # 平均并乘以输入差值
        integrated_grads = (X - baseline) * integrated_grads / steps
        
        return integrated_grads
    
    def get_feature_importance(self, X: np.ndarray, target_class: int = 2) -> pd.DataFrame:
        """
        获取特征重要性
        """
        # 计算积分梯度
        ig = self.compute_integrated_gradients(X, target_class=target_class)
        
        # 对时间和样本维度取平均
        feature_importance = np.abs(ig).mean(axis=(0, 1))
        
        num_features = len(feature_importance)
        feature_names = self._get_feature_names(num_features)
        
        df = pd.DataFrame({
            'feature': feature_names,
            'feature_cn': [FEATURE_NAMES_CN.get(f, f) for f in feature_names],
            'importance': feature_importance,
            'importance_pct': feature_importance / feature_importance.sum() * 100
        })
        
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return df


# ============================================================
# 可视化
# ============================================================

def plot_feature_importance(
    importance_df: pd.DataFrame, 
    title: str = "特征重要性排序",
    save_path: Path = None,
    top_n: int = 15
):
    """绘制特征重要性条形图"""
    if not HAS_PLOT:
        print("  [WARN] matplotlib未安装")
        return
    
    df = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(df)))[::-1]
    
    bars = ax.barh(range(len(df)), df['importance_pct'], color=colors)
    
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['feature_cn'])
    ax.invert_yaxis()
    
    ax.set_xlabel('重要性占比 (%)')
    ax.set_title(title)
    
    # 添加数值标签
    for i, (bar, pct) in enumerate(zip(bars, df['importance_pct'])):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  图表已保存: {save_path}")
    
    plt.close()


def plot_temporal_importance(
    temporal_importance: np.ndarray,
    title: str = "时间步重要性",
    save_path: Path = None
):
    """绘制时间步重要性图"""
    if not HAS_PLOT:
        return
    
    seq_len = len(temporal_importance)
    fig, ax = plt.subplots(figsize=(12, 4))
    
    x = np.arange(seq_len)
    ax.fill_between(x, 0, temporal_importance, alpha=0.3)
    ax.plot(x, temporal_importance, linewidth=1)
    
    # 动态设置x轴标签
    ax.set_xlabel(f'时间步 (t-{seq_len} 到 t)')
    ax.set_ylabel('平均 |SHAP|')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # 标注最近的时间步更重要
    ax.axvline(x=seq_len-1, color='red', linestyle='--', alpha=0.5, label='当前时刻')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  图表已保存: {save_path}")
    
    plt.close()


def plot_class_comparison(
    importance_by_class: Dict[str, pd.DataFrame],
    save_path: Path = None,
    top_n: int = 10
):
    """绘制不同预测类别的特征重要性对比"""
    if not HAS_PLOT:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    class_names = ['下跌', '平稳', '上涨']
    colors = ['#ff6b6b', '#ffd93d', '#6bcb77']
    
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        ax = axes[i]
        
        if str(i) in importance_by_class:
            df = importance_by_class[str(i)].head(top_n)
            
            ax.barh(range(len(df)), df['importance_pct'], color=color, alpha=0.7)
            ax.set_yticks(range(len(df)))
            ax.set_yticklabels(df['feature_cn'])
            ax.invert_yaxis()
            ax.set_xlabel('重要性 (%)')
            ax.set_title(f'预测{class_name}时的特征重要性')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  图表已保存: {save_path}")
    
    plt.close()


# ============================================================
# 主流程
# ============================================================

def load_model_for_shap(model_name: str, input_dim: int = 25, seq_len: int = 100):
    """加载模型"""
    sys.path.insert(0, str(Path(__file__).parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_trainer", Path(__file__).parent / "13_model_trainer.py")
    model_trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_trainer)
    create_model = model_trainer.create_model
    
    model_path = MODEL_DIR / model_name / 'model.pt'
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model = create_model(model_name, input_dim, seq_len)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    return model


def run_shap_analysis(
    model_name: str,
    data_dir: Path,
    n_samples: int = 100,
    output_dir: Path = None,
    use_gradient: bool = True
):
    """
    运行SHAP分析
    
    Args:
        model_name: 模型名称
        data_dir: 数据目录
        n_samples: 分析的样本数
        output_dir: 输出目录
        use_gradient: 是否使用梯度方法（更快）
    """
    print(f"\n{'='*60}")
    print(f"  SHAP分析: {model_name.upper()}")
    print("="*60)
    
    # 加载数据
    print("\n[1] 加载数据...")
    dataset_dir = data_dir / 'dataset_T100_k20'
    
    X_test = np.load(dataset_dir / 'X_test.npy')
    y_test = np.load(dataset_dir / 'y_test.npy')
    
    print(f"  测试集: {len(X_test)} 样本")
    
    # 加载模型
    print("\n[2] 加载模型...")
    model = load_model_for_shap(model_name, input_dim=X_test.shape[2], seq_len=X_test.shape[1])
    
    # 选择分析样本
    if len(X_test) > n_samples:
        indices = np.random.choice(len(X_test), n_samples, replace=False)
        X_analysis = X_test[indices]
        y_analysis = y_test[indices]
    else:
        X_analysis = X_test
        y_analysis = y_test
    
    # SHAP分析
    print("\n[3] 计算特征重要性...")
    
    if use_gradient:
        analyzer = GradientSHAPAnalyzer(model)
        
        # 智能检测标签格式
        y_min, y_max = y_analysis.min(), y_analysis.max()
        if y_min == -1 and y_max == 1:
            # 标签格式: -1, 0, 1
            label_map = {0: -1, 1: 0, 2: 1}  # class_idx -> label
        elif y_min == 0 and y_max == 2:
            # 标签格式: 0, 1, 2
            label_map = {0: 0, 1: 1, 2: 2}
        else:
            print(f"  [WARN] 非预期标签范围 [{y_min}, {y_max}]，假设为0,1,2格式")
            label_map = {0: 0, 1: 1, 2: 2}
        
        # 分类别计算
        importance_by_class = {}
        for class_idx, class_name in enumerate(['下跌', '平稳', '上涨']):
            print(f"  计算 {class_name} 类的特征重要性...")
            target_label = label_map[class_idx]
            mask = y_analysis == target_label
            if mask.sum() > 0:
                X_class = X_analysis[mask][:min(50, mask.sum())]
                importance_df = analyzer.get_feature_importance(X_class, target_class=class_idx)
                importance_by_class[str(class_idx)] = importance_df
        
        # 整体特征重要性
        overall_importance = analyzer.get_feature_importance(X_analysis[:100], target_class=2)
        
    else:
        if not HAS_SHAP:
            print("  [ERROR] SHAP未安装，使用梯度方法")
            return
        
        analyzer = SHAPAnalyzer(model)
        analyzer.fit(X_test[:200])
        analyzer.explain(X_analysis, n_samples=n_samples)
        
        overall_importance = analyzer.get_feature_importance(class_idx=2)
        importance_by_class = {
            '2': analyzer.get_feature_importance(class_idx=2),
            '1': analyzer.get_feature_importance(class_idx=1),
            '0': analyzer.get_feature_importance(class_idx=0)
        }
    
    # 打印结果
    print("\n" + "="*60)
    print(f"  {model_name.upper()} 特征重要性排序 (预测上涨)")
    print("="*60)
    print(overall_importance.head(15).to_string(index=False))
    
    # 保存结果
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存特征重要性
        overall_importance.to_csv(output_dir / f'feature_importance_{model_name}.csv', index=False)
        
        # 绘制图表
        plot_feature_importance(
            overall_importance,
            title=f'{model_name.upper()} 特征重要性排序',
            save_path=output_dir / f'shap_importance_{model_name}.png'
        )
        
        # 分类对比图
        if len(importance_by_class) == 3:
            plot_class_comparison(
                importance_by_class,
                save_path=output_dir / f'shap_class_comparison_{model_name}.png'
            )
        
        print(f"\n  结果已保存: {output_dir}")
    
    return overall_importance, importance_by_class


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='SHAP特征归因分析')
    parser.add_argument('--model', type=str, default='transformer', help='模型名称')
    parser.add_argument('--data', type=str, default='data/processed/combined', help='数据目录')
    parser.add_argument('--samples', type=int, default=100, help='分析样本数')
    parser.add_argument('--output', type=str, default='shap_results', help='输出目录')
    parser.add_argument('--use-shap', action='store_true', help='使用SHAP库（较慢）')
    
    args = parser.parse_args()
    
    print("="*60)
    print("  OFI论文 - SHAP特征归因分析")
    print("="*60)
    print(f"  模型: {args.model}")
    print(f"  样本数: {args.samples}")
    print(f"  方法: {'SHAP库' if args.use_shap else '积分梯度'}")
    
    try:
        run_shap_analysis(
            model_name=args.model,
            data_dir=Path(args.data),
            n_samples=args.samples,
            output_dir=Path(args.output),
            use_gradient=not args.use_shap
        )
        print("\n[DONE] SHAP分析完成！")
        
    except Exception as e:
        print(f"\n[ERROR] 分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
