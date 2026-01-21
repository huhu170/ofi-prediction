"""
K线数据集构建模块
将K线特征数据转换为深度学习模型所需的多尺度滑动窗口序列格式

功能:
1. 多尺度K线特征加载（1M/5M/60M/DAY）
2. 滑动窗口切片（按论文配置：1min=60步, 5min=24步, 60min=12步, 日K=20步）
3. 特征归一化（滚动窗口标准化，防止数据泄露）
4. 训练/验证/测试集划分（时序划分）
5. 生成PyTorch Dataset

使用方法:
    python 12b_kline_dataset_builder.py --code HK.00700 --horizon 5
    python 12b_kline_dataset_builder.py --code HK.00700 --multi-scale
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

# PyTorch
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch未安装，将只生成NumPy数组格式")

# ============================================================
# 配置（与论文第三章对齐）
# ============================================================

# 多尺度K线配置（论文表3.3-2）
KLINE_SCALES = {
    '1M': {
        'name': '1分钟',
        'input_len': 60,      # 输入长度：60根 = 1小时
        'horizon_steps': {    # 预测步长对应的K线数
            5: 5,             # 5分钟 = 5根1分钟K线
            15: 15,           # 15分钟 = 15根
            30: 30,           # 30分钟 = 30根
        }
    },
    '5M': {
        'name': '5分钟',
        'input_len': 24,      # 24根 = 2小时
        'horizon_steps': {
            5: 1,             # 5分钟 = 1根5分钟K线
            15: 3,            # 15分钟 = 3根
            30: 6,            # 30分钟 = 6根
        }
    },
    '60M': {
        'name': '60分钟',
        'input_len': 12,      # 12根 = 12小时
        'horizon_steps': {
            5: 1,             # 近似
            15: 1,
            30: 1,
        }
    },
    'DAY': {
        'name': '日K',
        'input_len': 20,      # 20根 = 1个月
        'horizon_steps': {
            5: 1,
            15: 1,
            30: 1,
        }
    },
}

# 默认预测步长（分钟）
DEFAULT_HORIZON = 5

# 数据划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# K线特征列（与11b_kline_feature_calculator.py对齐）
KLINE_FEATURE_COLS = [
    # 成交不平衡(TI)
    'ti', 'ti_5', 'ti_20', 'ti_zscore',
    # 收益率
    'return_1', 'return_5', 'return_20', 'return_zscore',
    # 成交量
    'relative_volume', 'volume_change', 'pv_corr',
    # 波动率
    'atr', 'atr_pct', 'range_pct', 'volatility_20',
    # 技术指标
    'rsi', 'macd_dif', 'macd_dea', 'macd', 'bb_position',
    # 市场状态
    'market_regime'
]

# 数据路径
DATA_PROCESSED = Path(os.getenv("DATA_PROCESSED", "data/processed"))


# ============================================================
# 特征标准化器（滚动窗口版本，防止数据泄露）
# ============================================================

class RollingScaler:
    """
    滚动窗口标准化器（论文第三章第五节）
    
    使用训练窗口的统计量进行标准化，严禁使用验证/测试期自身的统计量
    """
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, feature_names: List[str] = None):
        """在训练集上计算统计量"""
        if X.ndim == 3:
            X_flat = X.reshape(-1, X.shape[-1])
        else:
            X_flat = X
        
        self.mean = np.nanmean(X_flat, axis=0)
        self.std = np.nanstd(X_flat, axis=0)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        
        self.feature_names = feature_names
        self.is_fitted = True
        
        print(f"  标准化器已拟合: {len(self.mean)} 个特征")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """应用标准化"""
        if not self.is_fitted:
            raise ValueError("Scaler未拟合，请先调用fit()")
        return (X - self.mean) / self.std
    
    def fit_transform(self, X: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
        self.fit(X, feature_names)
        return self.transform(X)
    
    def save(self, path: Path):
        """保存标准化参数"""
        params = {
            'mean': self.mean,
            'std': self.std,
            'feature_names': self.feature_names
        }
        with open(path, 'wb') as f:
            pickle.dump(params, f)
    
    def load(self, path: Path):
        """加载标准化参数"""
        with open(path, 'rb') as f:
            params = pickle.load(f)
        self.mean = params['mean']
        self.std = params['std']
        self.feature_names = params['feature_names']
        self.is_fitted = True


# ============================================================
# 序列生成器
# ============================================================

class KlineSequenceGenerator:
    """K线滑动窗口序列生成器"""
    
    def __init__(self, seq_len: int, horizon_steps: int, step: int = 1):
        """
        Args:
            seq_len: 输入序列长度
            horizon_steps: 预测步长（K线根数）
            step: 滑动步长
        """
        self.seq_len = seq_len
        self.horizon_steps = horizon_steps
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
            X: (num_samples, seq_len, num_features)
            y: (num_samples,)
        """
        n_samples = len(features)
        max_start = n_samples - self.seq_len - self.horizon_steps + 1
        
        if max_start <= 0:
            raise ValueError(f"数据不足: 需要 {self.seq_len + self.horizon_steps} 点")
        
        indices = list(range(0, max_start, self.step))
        num_sequences = len(indices)
        
        X = np.zeros((num_sequences, self.seq_len, features.shape[1]))
        y = np.zeros(num_sequences)
        
        for i, start_idx in enumerate(indices):
            X[i] = features[start_idx : start_idx + self.seq_len]
            label_idx = start_idx + self.seq_len + self.horizon_steps - 1
            y[i] = labels[label_idx] if label_idx < n_samples else np.nan
        
        valid_mask = ~np.isnan(y)
        X, y = X[valid_mask], y[valid_mask]
        
        print(f"  生成序列: {len(X):,} 样本, T={self.seq_len}, F={features.shape[1]}")
        return X, y


# ============================================================
# 数据划分器
# ============================================================

class TimeSeriesSplitter:
    """时序数据划分器（防止未来信息泄露）"""
    
    def __init__(
        self,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        test_ratio: float = TEST_RATIO,
        gap: int = 0  # 训练/验证/测试之间的间隔
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.gap = gap
    
    def split(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        时序划分
        
        Returns:
            {'train': (X, y), 'val': (X, y), 'test': (X, y)}
        """
        n = len(X)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        return {
            'train': (X[:train_end], y[:train_end]),
            'val': (X[train_end + self.gap : val_end], y[train_end + self.gap : val_end]),
            'test': (X[val_end + self.gap:], y[val_end + self.gap:])
        }


# ============================================================
# PyTorch Dataset
# ============================================================

if HAS_TORCH:
    class KlineDataset(Dataset):
        """K线特征数据集"""
        
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y.astype(int))
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    class MultiScaleKlineDataset(Dataset):
        """
        多尺度K线数据集（用于LSF模块）
        
        返回: {
            '1M': (seq_len_1m, F),
            '5M': (seq_len_5m, F),
            '60M': (seq_len_60m, F),
            'DAY': (seq_len_day, F),
            'label': scalar
        }
        """
        
        def __init__(self, scale_data: Dict[str, np.ndarray], labels: np.ndarray):
            self.scale_data = {k: torch.FloatTensor(v) for k, v in scale_data.items()}
            self.labels = torch.LongTensor(labels.astype(int))
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            sample = {k: v[idx] for k, v in self.scale_data.items()}
            sample['label'] = self.labels[idx]
            return sample


# ============================================================
# 主数据集构建器
# ============================================================

class KlineDatasetBuilder:
    """K线数据集构建器"""
    
    def __init__(
        self,
        horizon_minutes: int = DEFAULT_HORIZON,
        feature_cols: List[str] = None
    ):
        self.horizon_minutes = horizon_minutes
        self.feature_cols = feature_cols or KLINE_FEATURE_COLS
        self.scaler = RollingScaler()
        self.stats = {}
    
    def load_features(self, code: str, ktype: str) -> Optional[pd.DataFrame]:
        """加载K线特征数据"""
        code_dir = DATA_PROCESSED / code.replace('.', '_')
        feature_path = code_dir / f"kline_features_{ktype}.parquet"
        
        if not feature_path.exists():
            print(f"  [WARN] 特征文件不存在: {feature_path}")
            return None
        
        df = pd.read_parquet(feature_path)
        print(f"  加载特征: {len(df):,} 条 ({ktype})")
        return df
    
    def build_single_scale(
        self,
        code: str,
        ktype: str = '1M'
    ) -> Optional[Dict]:
        """
        构建单尺度数据集
        
        Args:
            code: 股票代码
            ktype: K线类型
            
        Returns:
            {'train': Dataset, 'val': Dataset, 'test': Dataset, 'scaler': Scaler}
        """
        print(f"\n[1/4] 加载数据...")
        df = self.load_features(code, ktype)
        if df is None or df.empty:
            return None
        
        # 获取配置
        config = KLINE_SCALES[ktype]
        seq_len = config['input_len']
        horizon_steps = config['horizon_steps'].get(self.horizon_minutes, 1)
        
        # 准备特征和标签
        feature_cols = [c for c in self.feature_cols if c in df.columns]
        features = df[feature_cols].values
        
        label_col = f'label_{self.horizon_minutes}'
        if label_col not in df.columns:
            print(f"  [ERROR] 标签列不存在: {label_col}")
            return None
        labels = df[label_col].values
        
        print(f"[2/4] 生成序列...")
        generator = KlineSequenceGenerator(seq_len, horizon_steps)
        X, y = generator.generate(features, labels)
        
        print(f"[3/4] 划分数据集...")
        splitter = TimeSeriesSplitter()
        splits = splitter.split(X, y)
        
        print(f"[4/4] 标准化...")
        # 只在训练集上fit
        X_train_scaled = self.scaler.fit_transform(splits['train'][0], feature_cols)
        X_val_scaled = self.scaler.transform(splits['val'][0])
        X_test_scaled = self.scaler.transform(splits['test'][0])
        
        # 统计
        self.stats = {
            'ktype': ktype,
            'seq_len': seq_len,
            'n_features': len(feature_cols),
            'train_size': len(X_train_scaled),
            'val_size': len(X_val_scaled),
            'test_size': len(X_test_scaled),
        }
        
        if HAS_TORCH:
            return {
                'train': KlineDataset(X_train_scaled, splits['train'][1]),
                'val': KlineDataset(X_val_scaled, splits['val'][1]),
                'test': KlineDataset(X_test_scaled, splits['test'][1]),
                'scaler': self.scaler,
                'feature_names': feature_cols,
            }
        else:
            return {
                'train': (X_train_scaled, splits['train'][1]),
                'val': (X_val_scaled, splits['val'][1]),
                'test': (X_test_scaled, splits['test'][1]),
                'scaler': self.scaler,
                'feature_names': feature_cols,
            }
    
    def build_multi_scale(self, code: str) -> Optional[Dict]:
        """
        构建多尺度数据集（用于LSF模块）
        
        Returns:
            {'train': MultiScaleDataset, 'val': ..., 'test': ...}
        """
        print("\n构建多尺度数据集...")
        
        # 加载各尺度数据
        scale_features = {}
        min_len = float('inf')
        
        for ktype in ['1M', '5M', '60M', 'DAY']:
            df = self.load_features(code, ktype)
            if df is None:
                print(f"  [SKIP] {ktype} 数据缺失")
                continue
            
            config = KLINE_SCALES[ktype]
            feature_cols = [c for c in self.feature_cols if c in df.columns]
            
            # 生成序列
            seq_len = config['input_len']
            horizon_steps = config['horizon_steps'].get(self.horizon_minutes, 1)
            generator = KlineSequenceGenerator(seq_len, horizon_steps)
            
            features = df[feature_cols].values
            label_col = f'label_{self.horizon_minutes}'
            labels = df[label_col].values if label_col in df.columns else np.zeros(len(df))
            
            X, y = generator.generate(features, labels)
            scale_features[ktype] = {'X': X, 'y': y}
            min_len = min(min_len, len(X))
        
        if not scale_features:
            print("  [ERROR] 无可用数据")
            return None
        
        # 对齐样本数量（取最小值）
        for ktype in scale_features:
            scale_features[ktype]['X'] = scale_features[ktype]['X'][:min_len]
            scale_features[ktype]['y'] = scale_features[ktype]['y'][:min_len]
        
        # 使用1分钟数据的标签作为主标签
        main_labels = scale_features.get('1M', list(scale_features.values())[0])['y']
        
        # 划分
        splitter = TimeSeriesSplitter()
        n = min_len
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        
        result = {'train': {}, 'val': {}, 'test': {}}
        
        for ktype, data in scale_features.items():
            result['train'][ktype] = data['X'][:train_end]
            result['val'][ktype] = data['X'][train_end:val_end]
            result['test'][ktype] = data['X'][val_end:]
        
        result['train']['labels'] = main_labels[:train_end]
        result['val']['labels'] = main_labels[train_end:val_end]
        result['test']['labels'] = main_labels[val_end:]
        
        print(f"  多尺度数据集构建完成: {min_len} 样本")
        
        return result
    
    def print_stats(self):
        """打印统计信息"""
        print("\n" + "="*50)
        print("  数据集统计")
        print("="*50)
        for k, v in self.stats.items():
            print(f"  {k}: {v}")
        print("="*50)


# ============================================================
# 导出
# ============================================================

def export_dataset(
    dataset_dict: Dict,
    output_dir: Path,
    code: str,
    ktype: str
):
    """导出数据集"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为pickle
    output_path = output_dir / f"dataset_{code.replace('.', '_')}_{ktype}.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(dataset_dict, f)
    
    print(f"  [OK] 数据集已保存: {output_path}")
    
    # 保存标准化参数
    if 'scaler' in dataset_dict:
        scaler_path = output_dir / f"scaler_{code.replace('.', '_')}_{ktype}.pkl"
        dataset_dict['scaler'].save(scaler_path)


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='K线数据集构建')
    parser.add_argument('--code', type=str, default='HK.00700', help='股票代码')
    parser.add_argument('--ktype', type=str, default='1M', help='K线类型')
    parser.add_argument('--horizon', type=int, default=DEFAULT_HORIZON, 
                        help='预测步长（分钟）')
    parser.add_argument('--multi-scale', action='store_true', 
                        help='构建多尺度数据集')
    parser.add_argument('--output', type=str, default='data/datasets',
                        help='输出目录')
    
    args = parser.parse_args()
    
    print("="*50)
    print("  K线数据集构建")
    print("="*50)
    print(f"  股票代码: {args.code}")
    print(f"  预测步长: {args.horizon} 分钟")
    
    builder = KlineDatasetBuilder(horizon_minutes=args.horizon)
    output_dir = Path(args.output)
    
    if args.multi_scale:
        print("\n构建多尺度数据集...")
        result = builder.build_multi_scale(args.code)
        if result:
            export_dataset(result, output_dir, args.code, 'multi_scale')
    else:
        print(f"\n构建单尺度数据集 ({args.ktype})...")
        result = builder.build_single_scale(args.code, args.ktype)
        if result:
            export_dataset(result, output_dir, args.code, args.ktype)
            builder.print_stats()
    
    print("\n[DONE] 数据集构建完成！")


if __name__ == "__main__":
    main()
