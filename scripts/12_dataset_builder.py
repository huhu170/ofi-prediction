"""
数据集构建模块
将特征数据转换为深度学习模型所需的滑动窗口序列格式

功能:
1. 滑动窗口切片（T=100步作为输入序列）
2. 训练/验证/测试集划分（时序划分，避免数据泄露）
3. 特征归一化（Z-score标准化）
4. 生成 PyTorch Dataset 和 DataLoader

使用方法:
    python 12_dataset_builder.py --code HK.00700 --seq-len 100 --horizon 20
    python 12_dataset_builder.py --input data/processed/HK_00700/features_20260114.parquet
"""

import os
import sys
import io
import argparse
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union

# 解决Windows编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 加载环境变量
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".apikey.env"
load_dotenv(env_path, override=True)

import numpy as np
import pandas as pd

# PyTorch（可选，用于生成DataLoader）
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch未安装，将只生成NumPy数组格式")

# ============================================================
# 配置
# ============================================================

# 序列参数
DEFAULT_SEQ_LEN = 100       # 输入序列长度（100步 = 约16.7分钟）
DEFAULT_HORIZON = 20        # 预测步长（默认k=20）

# 数据划分比例（时序划分）
TRAIN_RATIO = 0.7           # 训练集70%
VAL_RATIO = 0.15            # 验证集15%
TEST_RATIO = 0.15           # 测试集15%

# 特征列配置（与11_feature_calculator.py对齐）
FEATURE_COLS = [
    # 价格特征
    'spread_bps', 'return_pct',
    # OFI聚合特征
    'ofi_l1', 'ofi_l5', 'ofi_l10', 'smart_ofi',
    # 分档OFI特征（各档独立OFI，用于SHAP分析）
    'ofi_level_1', 'ofi_level_2', 'ofi_level_3', 'ofi_level_4', 'ofi_level_5',
    'ofi_level_6', 'ofi_level_7', 'ofi_level_8', 'ofi_level_9', 'ofi_level_10',
    # OFI滚动统计
    'ofi_ma_10', 'ofi_std_10', 'ofi_zscore',
    'smart_ofi_ma_10', 'smart_ofi_std_10', 'smart_ofi_zscore',
    'return_ma_10', 'return_std_10',
    # 深度特征
    'bid_depth_5', 'ask_depth_5', 'depth_imbalance_5',
    'bid_depth_10', 'ask_depth_10', 'depth_imbalance_10',
    # 深度不平衡滚动统计
    'depth_imb_ma_10', 'depth_imb_std_10', 'depth_imb_zscore',
    # 成交特征
    'buy_volume', 'sell_volume', 'trade_count', 'trade_imbalance',
    # 成交不平衡滚动统计
    'trade_imb_ma_10', 'trade_imb_std_10', 'trade_imb_zscore',
    # 协方差
    'cov_stock_index', 'corr_stock_index',
    # 市场状态
    'market_regime'
]

# 数据路径
DATA_PROCESSED = Path(os.getenv("DATA_PROCESSED", "data/processed"))


# ============================================================
# 特征标准化器
# ============================================================

class FeatureScaler:
    """
    特征标准化器
    
    使用Z-score标准化: x_scaled = (x - mean) / std
    
    注意：只在训练集上计算统计量，防止数据泄露
    """
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, feature_names: List[str] = None):
        """
        在训练集上计算均值和标准差
        
        Args:
            X: 训练数据 (N, T, F) 或 (N, F)
            feature_names: 特征名称列表
        """
        # 展平时间维度（如果有）
        if X.ndim == 3:
            X_flat = X.reshape(-1, X.shape[-1])
        else:
            X_flat = X
        
        self.mean = np.nanmean(X_flat, axis=0)
        self.std = np.nanstd(X_flat, axis=0)
        
        # 防止除以零
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        
        self.feature_names = feature_names
        self.is_fitted = True
        
        print(f"  标准化器已拟合: {len(self.mean)} 个特征")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        应用标准化
        
        Args:
            X: 数据 (N, T, F) 或 (N, F)
            
        Returns:
            标准化后的数据
        """
        if not self.is_fitted:
            raise ValueError("Scaler未拟合，请先调用fit()")
        
        return (X - self.mean) / self.std
    
    def fit_transform(self, X: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
        """拟合并转换"""
        self.fit(X, feature_names)
        return self.transform(X)
    
    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """逆变换（恢复原始尺度）"""
        return X_scaled * self.std + self.mean
    
    def save(self, path: Path):
        """保存标准化参数"""
        params = {
            'mean': self.mean,
            'std': self.std,
            'feature_names': self.feature_names
        }
        with open(path, 'wb') as f:
            pickle.dump(params, f)
        print(f"  标准化参数已保存: {path}")
    
    def load(self, path: Path):
        """加载标准化参数"""
        with open(path, 'rb') as f:
            params = pickle.load(f)
        self.mean = params['mean']
        self.std = params['std']
        self.feature_names = params['feature_names']
        self.is_fitted = True
        print(f"  标准化参数已加载: {path}")


# ============================================================
# 滑动窗口生成器
# ============================================================

class SequenceGenerator:
    """
    滑动窗口序列生成器
    
    将时间序列数据转换为 (X, y) 对:
    - X: 过去T步的特征序列 (T, F)
    - y: 未来第k步的标签
    """
    
    def __init__(
        self, 
        seq_len: int = DEFAULT_SEQ_LEN,
        horizon: int = DEFAULT_HORIZON,
        step: int = 1
    ):
        """
        Args:
            seq_len: 输入序列长度（时间步数）
            horizon: 预测步长
            step: 滑动步长（默认1）
        """
        self.seq_len = seq_len
        self.horizon = horizon
        self.step = step
    
    def generate(
        self, 
        features: np.ndarray, 
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成滑动窗口序列
        
        Args:
            features: 特征数组 (N, F)
            labels: 标签数组 (N,)
            
        Returns:
            X: 特征序列 (num_samples, seq_len, num_features)
            y: 标签数组 (num_samples,)
        """
        n_samples = len(features)
        
        # 计算可生成的样本数
        # 需要至少 seq_len 个历史点 + horizon 个未来点
        max_start = n_samples - self.seq_len - self.horizon + 1
        
        if max_start <= 0:
            raise ValueError(
                f"数据不足: 需要至少 {self.seq_len + self.horizon} 个时间点，"
                f"但只有 {n_samples} 个"
            )
        
        # 生成索引
        indices = list(range(0, max_start, self.step))
        num_sequences = len(indices)
        
        # 预分配数组
        X = np.zeros((num_sequences, self.seq_len, features.shape[1]))
        y = np.zeros(num_sequences)
        
        for i, start_idx in enumerate(indices):
            # 输入序列: [start_idx, start_idx + seq_len)
            X[i] = features[start_idx : start_idx + self.seq_len]
            
            # 标签: 序列结束后第 horizon 步
            label_idx = start_idx + self.seq_len + self.horizon - 1
            y[i] = labels[label_idx] if label_idx < n_samples else np.nan
        
        # 移除无效样本（标签为NaN的）
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"  生成序列: {len(X):,} 个样本")
        print(f"    序列长度: T = {self.seq_len}")
        print(f"    预测步长: k = {self.horizon}")
        print(f"    特征维度: F = {features.shape[1]}")
        
        return X, y


# ============================================================
# 数据集划分
# ============================================================

class DataSplitter:
    """
    时序数据划分器
    
    采用时序划分（而非随机划分），防止未来信息泄露
    """
    
    def __init__(
        self,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        test_ratio: float = TEST_RATIO
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
    
    def split(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        按时序划分数据集
        
        Args:
            X: 特征数组 (N, T, F)
            y: 标签数组 (N,)
            
        Returns:
            {
                'train': (X_train, y_train),
                'val': (X_val, y_val),
                'test': (X_test, y_test)
            }
        """
        n = len(X)
        
        # 计算分割点
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        # 划分数据
        splits = {
            'train': (X[:train_end], y[:train_end]),
            'val': (X[train_end:val_end], y[train_end:val_end]),
            'test': (X[val_end:], y[val_end:])
        }
        
        print(f"\n  数据集划分（时序）:")
        print(f"    训练集: {len(splits['train'][0]):,} 样本 ({self.train_ratio*100:.0f}%)")
        print(f"    验证集: {len(splits['val'][0]):,} 样本 ({self.val_ratio*100:.0f}%)")
        print(f"    测试集: {len(splits['test'][0]):,} 样本 ({self.test_ratio*100:.0f}%)")
        
        # 统计标签分布
        for split_name, (_, y_split) in splits.items():
            if len(y_split) > 0:
                unique, counts = np.unique(y_split, return_counts=True)
                dist = {int(u): c/len(y_split)*100 for u, c in zip(unique, counts)}
                print(f"    {split_name} 标签分布: {dist}")
        
        return splits


# ============================================================
# PyTorch Dataset
# ============================================================

if HAS_TORCH:
    class OFIDataset(Dataset):
        """
        OFI特征数据集（PyTorch格式）
        """
        
        def __init__(
            self, 
            X: np.ndarray, 
            y: np.ndarray,
            transform=None
        ):
            """
            Args:
                X: 特征数组 (N, T, F)
                y: 标签数组 (N,)
                transform: 可选的数据增强
            """
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y.astype(np.int64) + 1)  # -1,0,1 -> 0,1,2
            self.transform = transform
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            x = self.X[idx]
            y = self.y[idx]
            
            if self.transform:
                x = self.transform(x)
            
            return x, y
    
    def create_dataloaders(
        splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
        batch_size: int = 64,
        num_workers: int = 0
    ) -> Dict[str, DataLoader]:
        """
        创建PyTorch DataLoader
        
        Args:
            splits: 划分后的数据集
            batch_size: 批大小
            num_workers: 数据加载线程数
            
        Returns:
            {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
        """
        loaders = {}
        
        for split_name, (X, y) in splits.items():
            dataset = OFIDataset(X, y)
            
            shuffle = (split_name == 'train')
            
            loaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True if torch.cuda.is_available() else False
            )
        
        return loaders


# ============================================================
# 主流程
# ============================================================

class DatasetBuilder:
    """
    数据集构建主流程
    """
    
    def __init__(
        self,
        seq_len: int = DEFAULT_SEQ_LEN,
        horizon: int = DEFAULT_HORIZON,
        feature_cols: List[str] = None
    ):
        self.seq_len = seq_len
        self.horizon = horizon
        self.feature_cols = feature_cols or FEATURE_COLS
        
        self.scaler = FeatureScaler()
        self.seq_gen = SequenceGenerator(seq_len, horizon)
        self.splitter = DataSplitter()
        
        self.stats = {}
    
    def load_features(self, file_paths: List[Path]) -> pd.DataFrame:
        """
        加载特征文件
        
        Args:
            file_paths: 特征文件路径列表
            
        Returns:
            合并后的DataFrame
        """
        dfs = []
        
        for path in file_paths:
            df = pd.read_parquet(path)
            dfs.append(df)
            print(f"  加载: {path.name} ({len(df):,} 行)")
        
        # 合并并按时间排序
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values('ts').reset_index(drop=True)
        
        print(f"  总计: {len(combined):,} 行")
        
        return combined
    
    def prepare_arrays(
        self, 
        df: pd.DataFrame,
        label_col: str = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备特征和标签数组
        
        Args:
            df: 特征DataFrame
            label_col: 标签列名（如 'label_20'）
            
        Returns:
            features: (N, F) 特征数组
            labels: (N,) 标签数组
            feature_names: 特征名称列表
        """
        if label_col is None:
            label_col = f'label_{self.horizon}'
        
        # 选择存在的特征列
        available_cols = [c for c in self.feature_cols if c in df.columns]
        missing_cols = [c for c in self.feature_cols if c not in df.columns]
        
        if missing_cols:
            print(f"  [WARN] 缺失特征列: {missing_cols}")
        
        print(f"  使用特征: {len(available_cols)} 个")
        
        # 提取数组
        features = df[available_cols].values.astype(np.float32)
        labels = df[label_col].values.astype(np.float32)
        
        # 处理NaN（用0填充特征，移除标签为NaN的行）
        features = np.nan_to_num(features, nan=0.0)
        
        valid_mask = ~np.isnan(labels)
        features = features[valid_mask]
        labels = labels[valid_mask]
        
        print(f"  有效样本: {len(features):,} 行")
        
        return features, labels, available_cols
    
    def build(
        self,
        file_paths: List[Path],
        label_col: str = None,
        normalize: bool = True
    ) -> Dict:
        """
        构建完整数据集
        
        Args:
            file_paths: 特征文件路径列表
            label_col: 标签列名
            normalize: 是否标准化
            
        Returns:
            {
                'splits': {'train': (X, y), 'val': (X, y), 'test': (X, y)},
                'scaler': FeatureScaler,
                'feature_names': List[str],
                'config': dict
            }
        """
        print("\n[1/4] 加载特征数据...")
        df = self.load_features(file_paths)
        
        print("\n[2/4] 准备特征数组...")
        features, labels, feature_names = self.prepare_arrays(df, label_col)
        
        print("\n[3/4] 生成滑动窗口序列...")
        X, y = self.seq_gen.generate(features, labels)
        
        print("\n[4/4] 划分数据集...")
        splits = self.splitter.split(X, y)
        
        # 标准化（只在训练集上fit）
        if normalize:
            print("\n[+] 标准化特征...")
            X_train, y_train = splits['train']
            self.scaler.fit(X_train, feature_names)
            
            # 应用到所有集合
            splits = {
                name: (self.scaler.transform(X), y)
                for name, (X, y) in splits.items()
            }
        
        # 保存统计信息
        self.stats = {
            'total_samples': len(X),
            'seq_len': self.seq_len,
            'horizon': self.horizon,
            'num_features': len(feature_names),
            'train_samples': len(splits['train'][0]),
            'val_samples': len(splits['val'][0]),
            'test_samples': len(splits['test'][0])
        }
        
        return {
            'splits': splits,
            'scaler': self.scaler,
            'feature_names': feature_names,
            'config': {
                'seq_len': self.seq_len,
                'horizon': self.horizon,
                'feature_cols': feature_names,
                'normalize': normalize
            }
        }
    
    def save(
        self, 
        result: Dict, 
        output_dir: Path,
        save_numpy: bool = True,
        save_torch: bool = True
    ):
        """
        保存数据集
        
        Args:
            result: build()的返回结果
            output_dir: 输出目录
            save_numpy: 是否保存NumPy格式
            save_torch: 是否保存PyTorch格式
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        splits = result['splits']
        
        # 保存NumPy格式
        if save_numpy:
            print("\n  保存NumPy格式...")
            for name, (X, y) in splits.items():
                np.save(output_dir / f'X_{name}.npy', X)
                np.save(output_dir / f'y_{name}.npy', y)
            print(f"    保存至: {output_dir}")
        
        # 保存PyTorch格式
        if save_torch and HAS_TORCH:
            print("\n  保存PyTorch格式...")
            for name, (X, y) in splits.items():
                dataset = OFIDataset(X, y)
                torch.save(dataset, output_dir / f'dataset_{name}.pt')
            print(f"    保存至: {output_dir}")
        
        # 保存Scaler
        result['scaler'].save(output_dir / 'scaler.pkl')
        
        # 保存配置
        config_path = output_dir / 'config.pkl'
        with open(config_path, 'wb') as f:
            pickle.dump(result['config'], f)
        print(f"  配置已保存: {config_path}")
    
    def print_stats(self):
        """打印统计信息"""
        print("\n" + "="*50)
        print("  数据集构建统计")
        print("="*50)
        for key, value in self.stats.items():
            print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
        print("="*50)


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='数据集构建模块')
    parser.add_argument('--input', type=str, nargs='+', help='输入特征文件路径')
    parser.add_argument('--code', type=str, help='股票代码')
    parser.add_argument('--days', type=int, default=30, help='使用最近N天数据')
    parser.add_argument('--seq-len', type=int, default=DEFAULT_SEQ_LEN, help='序列长度')
    parser.add_argument('--horizon', type=int, default=DEFAULT_HORIZON, help='预测步长')
    parser.add_argument('--batch-size', type=int, default=64, help='批大小')
    parser.add_argument('--output', type=str, help='输出目录')
    parser.add_argument('--no-normalize', action='store_true', help='不进行标准化')
    
    args = parser.parse_args()
    
    print("="*50)
    print("  OFI论文 - 数据集构建模块")
    print("="*50)
    print(f"  序列长度: T = {args.seq_len}")
    print(f"  预测步长: k = {args.horizon}")
    print(f"  批大小: {args.batch_size}")
    
    # 确定输入文件
    if args.input:
        input_files = [Path(p) for p in args.input]
    elif args.code:
        code_dir = DATA_PROCESSED / args.code.replace('.', '_')
        input_files = sorted(code_dir.glob('features_*.parquet'))[-args.days:]
    else:
        # 查找所有特征文件
        input_files = sorted(DATA_PROCESSED.glob('*/features_*.parquet'))[-args.days:]
    
    if not input_files:
        print("\n[ERROR] 未找到输入文件")
        return
    
    print(f"\n  找到 {len(input_files)} 个特征文件")
    
    # 构建数据集
    builder = DatasetBuilder(
        seq_len=args.seq_len,
        horizon=args.horizon
    )
    
    result = builder.build(
        input_files,
        normalize=not args.no_normalize
    )
    
    # 确定输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        code = args.code.replace('.', '_') if args.code else 'combined'
        output_dir = DATA_PROCESSED / code / f'dataset_T{args.seq_len}_k{args.horizon}'
    
    # 保存
    builder.save(result, output_dir)
    
    # 创建DataLoader示例
    if HAS_TORCH:
        print("\n[+] 创建PyTorch DataLoader...")
        loaders = create_dataloaders(result['splits'], batch_size=args.batch_size)
        
        for name, loader in loaders.items():
            print(f"    {name}: {len(loader)} batches")
        
        # 测试一个batch
        X_batch, y_batch = next(iter(loaders['train']))
        print(f"\n  示例batch:")
        print(f"    X shape: {X_batch.shape}")  # (batch, seq_len, features)
        print(f"    y shape: {y_batch.shape}")  # (batch,)
        print(f"    y values: {y_batch[:10].tolist()}")  # 0=下跌, 1=平稳, 2=上涨
    
    # 打印统计
    builder.print_stats()
    
    print("\n[DONE] 数据集构建完成！")


if __name__ == "__main__":
    main()
