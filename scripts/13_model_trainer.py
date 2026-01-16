"""
模型训练模块
包含所有基准模型和本研究模型的定义、训练与评估

模型体系:
1. 基准模型: Logistic Regression, XGBoost, Random Forest
2. 深度学习基准: LSTM, GRU
3. 专用架构: DeepLOB (Zhang et al. 2019)
4. 本研究模型: Transformer, Smart-Transformer (协方差加权)

使用方法:
    python 13_model_trainer.py --model lstm --data data/processed/HK_00700/dataset_T100_k20
    python 13_model_trainer.py --model all --epochs 50
    python 13_model_trainer.py --model transformer smart_trans --compare
"""

import os
import sys
import io
import json
import argparse
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from abc import ABC, abstractmethod

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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# sklearn（可选）
try:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
    from sklearn.preprocessing import label_binarize
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] sklearn未安装，部分评估指标和ML基准模型不可用")

# XGBoost（可选）
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# statsmodels（ARIMA）
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

HAS_ML = HAS_SKLEARN

# ============================================================
# 配置
# ============================================================

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型超参数
MODEL_CONFIG = {
    'lstm': {
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'bidirectional': False
    },
    'gru': {
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'bidirectional': False
    },
    'deeplob': {
        'conv_filters': [32, 32, 32],
        'lstm_hidden': 64,
        'dropout': 0.2
    },
    'transformer': {
        'd_model': 256,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 512,
        'dropout': 0.1
    },
    'smart_trans': {
        'd_model': 256,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'cov_gamma': 1.0  # 协方差权重系数
    }
}

# 训练超参数
TRAIN_CONFIG = {
    'batch_size': 64,
    'learning_rate': 1e-4,
    'max_epochs': 100,
    'early_stopping_patience': 10,
    'lr_scheduler_patience': 5,
    'lr_scheduler_factor': 0.5,
    'weight_decay': 1e-5
}

# 类别数
NUM_CLASSES = 3  # 下跌(0), 平稳(1), 上涨(2)

# 数据路径
DATA_PROCESSED = Path(os.getenv("DATA_PROCESSED", "data/processed"))
MODEL_DIR = Path("models")


# ============================================================
# 数据加载
# ============================================================

def load_dataset(data_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    加载数据集
    
    Args:
        data_dir: 数据集目录（包含X_train.npy等文件）
        
    Returns:
        {'train': (X, y), 'val': (X, y), 'test': (X, y)}
    """
    data_dir = Path(data_dir)
    
    splits = {}
    for split in ['train', 'val', 'test']:
        X = np.load(data_dir / f'X_{split}.npy')
        y = np.load(data_dir / f'y_{split}.npy')
        splits[split] = (X, y)
        print(f"  {split}: X={X.shape}, y={y.shape}")
    
    return splits


def create_dataloaders(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    batch_size: int = TRAIN_CONFIG['batch_size'],
    correlations: Dict[str, np.ndarray] = None
) -> Dict[str, DataLoader]:
    """
    创建PyTorch DataLoader
    
    Args:
        splits: 数据集字典 {'train': (X, y), ...}
        batch_size: 批大小
        correlations: 可选的相关系数字典，用于协方差加权训练
        
    Returns:
        DataLoader字典
    """
    loaders = {}
    
    for split, (X, y) in splits.items():
        X_tensor = torch.FloatTensor(X)
        
        # 智能标签转换: 自动检测标签范围
        y_min, y_max = y.min(), y.max()
        if y_min == -1 and y_max == 1:
            # 原始标签 -1,0,1 → 0,1,2 (CrossEntropyLoss需要从0开始)
            y_shifted = y.astype(np.int64) + 1
        elif y_min == 0 and y_max == 2:
            # 已经是 0,1,2 格式
            y_shifted = y.astype(np.int64)
        else:
            print(f"  [WARN] 非预期标签范围 [{y_min}, {y_max}]，尝试直接使用")
            y_shifted = y.astype(np.int64)
        
        y_tensor = torch.LongTensor(y_shifted)
        
        # 如果提供了相关系数，创建包含相关系数的Dataset
        if correlations is not None and split in correlations:
            corr_tensor = torch.FloatTensor(correlations[split])
            dataset = TensorDataset(X_tensor, y_tensor, corr_tensor)
        else:
            dataset = TensorDataset(X_tensor, y_tensor)
        
        shuffle = (split == 'train')
        
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    return loaders


# ============================================================
# 模型基类
# ============================================================

class BaseModel(ABC, nn.Module):
    """所有模型的抽象基类"""
    
    def __init__(self, input_dim: int, seq_len: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.model_name = "base"
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, seq_len, input_dim)
            
        Returns:
            logits: 输出张量 (batch, num_classes)
        """
        pass
    
    def get_num_params(self) -> int:
        """获取模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path: Path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'input_dim': self.input_dim,
            'seq_len': self.seq_len,
            'num_classes': self.num_classes
        }, path)
    
    def load(self, path: Path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=DEVICE)
        self.load_state_dict(checkpoint['model_state_dict'])


# ============================================================
# LSTM模型
# ============================================================

class LSTMModel(BaseModel):
    """
    2层堆叠LSTM模型
    
    架构:
    Input (batch, seq, features) 
    → LSTM (2层, hidden=128) 
    → 取最后时间步 
    → FC → Softmax
    """
    
    def __init__(
        self, 
        input_dim: int, 
        seq_len: int, 
        num_classes: int = NUM_CLASSES,
        config: dict = None
    ):
        super().__init__(input_dim, seq_len, num_classes)
        self.model_name = "lstm"
        
        cfg = config or MODEL_CONFIG['lstm']
        self.hidden_dim = cfg['hidden_dim']
        self.num_layers = cfg['num_layers']
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=cfg['dropout'] if self.num_layers > 1 else 0,
            bidirectional=cfg['bidirectional']
        )
        
        fc_input_dim = self.hidden_dim * (2 if cfg['bidirectional'] else 1)
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # (batch, hidden)
        logits = self.fc(last_output)
        return logits


# ============================================================
# GRU模型
# ============================================================

class GRUModel(BaseModel):
    """
    2层堆叠GRU模型
    
    与LSTM类似，但使用GRU单元（参数更少）
    """
    
    def __init__(
        self, 
        input_dim: int, 
        seq_len: int, 
        num_classes: int = NUM_CLASSES,
        config: dict = None
    ):
        super().__init__(input_dim, seq_len, num_classes)
        self.model_name = "gru"
        
        cfg = config or MODEL_CONFIG['gru']
        self.hidden_dim = cfg['hidden_dim']
        self.num_layers = cfg['num_layers']
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=cfg['dropout'] if self.num_layers > 1 else 0,
            bidirectional=cfg['bidirectional']
        )
        
        fc_input_dim = self.hidden_dim * (2 if cfg['bidirectional'] else 1)
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, h_n = self.gru(x)
        last_output = gru_out[:, -1, :]
        logits = self.fc(last_output)
        return logits


# ============================================================
# DeepLOB模型 (Zhang et al. 2019)
# ============================================================

class DeepLOBModel(BaseModel):
    """
    DeepLOB模型 (Zhang et al. 2019)
    
    架构:
    Input (batch, seq, features)
    → 3层1D卷积（跨特征维度）
    → LSTM
    → FC → Softmax
    
    参考: Zhang et al. "DeepLOB: Deep Convolutional Neural Networks 
          for Limit Order Books" (2019)
    """
    
    def __init__(
        self, 
        input_dim: int, 
        seq_len: int, 
        num_classes: int = NUM_CLASSES,
        config: dict = None
    ):
        super().__init__(input_dim, seq_len, num_classes)
        self.model_name = "deeplob"
        
        cfg = config or MODEL_CONFIG['deeplob']
        
        # 卷积层（1D卷积，沿时间维度）
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, cfg['conv_filters'][0], kernel_size=3, padding=1),
            nn.BatchNorm1d(cfg['conv_filters'][0]),
            nn.LeakyReLU(0.01),
            nn.MaxPool1d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(cfg['conv_filters'][0], cfg['conv_filters'][1], kernel_size=3, padding=1),
            nn.BatchNorm1d(cfg['conv_filters'][1]),
            nn.LeakyReLU(0.01),
            nn.MaxPool1d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(cfg['conv_filters'][1], cfg['conv_filters'][2], kernel_size=3, padding=1),
            nn.BatchNorm1d(cfg['conv_filters'][2]),
            nn.LeakyReLU(0.01)
        )
        
        # LSTM层（卷积后序列长度 = seq_len // 4，经过2次MaxPool1d(2)）
        self.lstm = nn.LSTM(
            input_size=cfg['conv_filters'][2],
            hidden_size=cfg['lstm_hidden'],
            num_layers=1,
            batch_first=True
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(cfg['lstm_hidden'], 32),
            nn.ReLU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features) → (batch, features, seq) for Conv1d
        x = x.permute(0, 2, 1)
        
        # 卷积
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # (batch, channels, seq) → (batch, seq, channels) for LSTM
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        # 分类
        logits = self.fc(last_output)
        return logits


# ============================================================
# Transformer模型
# ============================================================

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(BaseModel):
    """
    标准Transformer Encoder模型
    
    架构:
    Input (batch, seq, features)
    → Linear (features → d_model)
    → Positional Encoding
    → Transformer Encoder (4层)
    → 取CLS token或平均池化
    → FC → Softmax
    """
    
    def __init__(
        self, 
        input_dim: int, 
        seq_len: int, 
        num_classes: int = NUM_CLASSES,
        config: dict = None
    ):
        super().__init__(input_dim, seq_len, num_classes)
        self.model_name = "transformer"
        
        cfg = config or MODEL_CONFIG['transformer']
        self.d_model = cfg['d_model']
        
        # 输入嵌入
        self.input_embedding = nn.Linear(input_dim, self.d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=seq_len + 1, dropout=cfg['dropout'])
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=cfg['nhead'],
            dim_feedforward=cfg['dim_feedforward'],
            dropout=cfg['dropout'],
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg['num_layers'])
        
        # 分类头
        self.fc = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(self.d_model // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 输入嵌入
        x = self.input_embedding(x)  # (batch, seq, d_model)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq+1, d_model)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 取CLS token的输出
        cls_output = x[:, 0, :]  # (batch, d_model)
        
        # 分类
        logits = self.fc(cls_output)
        return logits


# ============================================================
# Smart-Transformer模型（本研究）
# ============================================================

class SmartTransformer(BaseModel):
    """
    Smart-Transformer模型（本研究创新）
    
    在标准Transformer基础上增加:
    1. 输入层融合Smart-OFI等质量过滤特征
    2. 训练阶段支持协方差加权损失
    
    架构与TransformerModel相同，但训练策略不同
    """
    
    def __init__(
        self, 
        input_dim: int, 
        seq_len: int, 
        num_classes: int = NUM_CLASSES,
        config: dict = None
    ):
        super().__init__(input_dim, seq_len, num_classes)
        self.model_name = "smart_trans"
        
        cfg = config or MODEL_CONFIG['smart_trans']
        self.d_model = cfg['d_model']
        self.cov_gamma = cfg.get('cov_gamma', 1.0)
        
        # 输入嵌入（可以增加特征交互层）
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(cfg['dropout'])
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=seq_len + 1, dropout=cfg['dropout'])
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=cfg['nhead'],
            dim_feedforward=cfg['dim_feedforward'],
            dropout=cfg['dropout'],
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg['num_layers'])
        
        # 分类头
        self.fc = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(self.d_model // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 输入嵌入
        x = self.input_embedding(x)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 取CLS token的输出
        cls_output = x[:, 0, :]
        
        # 分类
        logits = self.fc(cls_output)
        return logits
    
    def compute_sample_weights(self, correlations: torch.Tensor) -> torch.Tensor:
        """
        计算协方差加权的样本权重（工具方法，供外部调用或自定义训练循环使用）
        
        公式: w_t = 1 + γ × max(0, ρ_t)
        
        当 ρ_t > 0 时（个股与指数正相关），给予更高权重，
        因为此时OFI的价格预测信号更可靠。
        
        Args:
            correlations: 个股-指数相关系数 (batch,)
            
        Returns:
            weights: 样本权重 (batch,)
            
        Note:
            实际训练时，协方差加权在 Trainer.train_epoch() 中实现，
            以确保在 DataLoader shuffle 后索引正确对应。
        """
        weights = 1.0 + self.cov_gamma * torch.clamp(correlations, min=0)
        return weights


# ============================================================
# 机器学习基准模型
# ============================================================

if HAS_SKLEARN:
    class MLBaselines:
        """机器学习基准模型（非深度学习）"""
        
        def __init__(self):
            self.models = {}
            self.arima_threshold = None  # ARIMA分类阈值
        
        def train_logistic(self, X_train: np.ndarray, y_train: np.ndarray):
            """训练逻辑回归"""
            # 展平序列: (N, T, F) → (N, T*F)
            X_flat = X_train.reshape(X_train.shape[0], -1)
            
            model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                multi_class='multinomial',
                solver='lbfgs',
                n_jobs=-1
            )
            model.fit(X_flat, y_train)
            self.models['logistic'] = model
            return model
        
        def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray):
            """训练XGBoost"""
            if not HAS_XGB:
                print("  [WARN] XGBoost未安装，跳过")
                return None
                
            X_flat = X_train.reshape(X_train.shape[0], -1)
            
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective='multi:softprob',  # 使用softprob以支持predict_proba
                num_class=NUM_CLASSES,
                eval_metric='mlogloss',
                n_jobs=-1,
                verbosity=0  # 减少日志输出
            )
            model.fit(X_flat, y_train)
            self.models['xgboost'] = model
            return model
        
        def train_rf(self, X_train: np.ndarray, y_train: np.ndarray):
            """训练随机森林"""
            X_flat = X_train.reshape(X_train.shape[0], -1)
            
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                n_jobs=-1,
                random_state=42
            )
            model.fit(X_flat, y_train)
            self.models['rf'] = model
            return model
        
        def train_arima(self, X_train: np.ndarray, y_train: np.ndarray):
            """
            训练ARIMA基准模型
            
            ARIMA用于预测时间序列的下一步值，然后转换为三分类:
            - 使用序列最后几个时间步的趋势作为预测
            - 基于预测值的正负和幅度进行分类
            
            注意: 这是一个简化的ARIMA基准，主要用于对比
            """
            if not HAS_STATSMODELS:
                print("  [WARN] statsmodels未安装，使用简化版ARIMA")
            
            # 提取趋势特征用于预测（使用第一个特征，通常是收益率相关）
            # 计算每个样本的趋势（最后5个时间步的平均变化）
            trends = []
            for i in range(len(X_train)):
                seq = X_train[i, :, 0]  # 取第一个特征
                # 计算最后5个时间步的变化趋势
                last_changes = np.diff(seq[-6:])  # 最后5个变化
                trend = np.mean(last_changes)
                trends.append(trend)
            
            trends = np.array(trends)
            
            # 学习最优阈值将趋势转换为分类
            # 使用训练集的趋势分布来确定阈值
            sorted_trends = np.sort(trends)
            n = len(sorted_trends)
            
            # 使用分位数作为阈值
            lower_threshold = np.percentile(trends, 33)
            upper_threshold = np.percentile(trends, 67)
            
            # 存储阈值供预测使用
            self.arima_threshold = (lower_threshold, upper_threshold)
            self.models['arima'] = 'trend_based'
            
            print(f"    ARIMA阈值: 下={lower_threshold:.6f}, 上={upper_threshold:.6f}")
            
            return self.models['arima']
        
        def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
            """预测"""
            if model_name == 'arima':
                return self._predict_arima(X)
            
            X_flat = X.reshape(X.shape[0], -1)
            return self.models[model_name].predict(X_flat)
        
        def predict_proba(self, model_name: str, X: np.ndarray) -> np.ndarray:
            """预测概率"""
            if model_name == 'arima':
                return self._predict_arima_proba(X)
            
            X_flat = X.reshape(X.shape[0], -1)
            return self.models[model_name].predict_proba(X_flat)
        
        def _predict_arima(self, X: np.ndarray) -> np.ndarray:
            """ARIMA预测"""
            lower_th, upper_th = self.arima_threshold
            
            predictions = []
            for i in range(len(X)):
                seq = X[i, :, 0]
                last_changes = np.diff(seq[-6:])
                trend = np.mean(last_changes)
                
                # 基于趋势分类
                if trend < lower_th:
                    pred = 0  # 下跌
                elif trend > upper_th:
                    pred = 2  # 上涨
                else:
                    pred = 1  # 平稳
                
                predictions.append(pred)
            
            return np.array(predictions)
        
        def _predict_arima_proba(self, X: np.ndarray) -> np.ndarray:
            """
            ARIMA预测概率（基于趋势的软分类）
            
            使用softmax将趋势值转换为概率分布：
            - 计算趋势与各类别中心的距离
            - 通过softmax得到归一化概率
            """
            lower_th, upper_th = self.arima_threshold
            mid_th = (lower_th + upper_th) / 2  # 平稳区域中心
            
            # 类别中心（基于训练集阈值推断）
            down_center = lower_th - abs(lower_th)  # 下跌中心
            stable_center = mid_th  # 平稳中心
            up_center = upper_th + abs(upper_th)  # 上涨中心
            
            # 温度参数，控制概率分布的锐度
            temperature = max(abs(upper_th - lower_th), 1e-6) * 2
            
            probas = []
            for i in range(len(X)):
                seq = X[i, :, 0]
                last_changes = np.diff(seq[-6:])
                trend = np.mean(last_changes)
                
                # 计算到各类别中心的负距离（作为logits）
                logits = np.array([
                    -abs(trend - down_center),    # 下跌
                    -abs(trend - stable_center),  # 平稳
                    -abs(trend - up_center)       # 上涨
                ]) / temperature
                
                # Softmax转换为概率
                exp_logits = np.exp(logits - np.max(logits))  # 数值稳定
                proba = exp_logits / exp_logits.sum()
                
                probas.append(proba)
            
            return np.array(probas)
        
        def save_model(self, model_name: str, output_dir: Path):
            """保存ML模型"""
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if model_name == 'arima':
                # ARIMA只保存阈值参数
                params = {
                    'model_type': 'arima',
                    'threshold': self.arima_threshold
                }
                with open(output_dir / 'model.pkl', 'wb') as f:
                    pickle.dump(params, f)
            elif model_name in self.models:
                # 保存sklearn/xgboost模型
                params = {
                    'model_type': model_name,
                    'model': self.models[model_name]
                }
                with open(output_dir / 'model.pkl', 'wb') as f:
                    pickle.dump(params, f)
            
            print(f"    模型已保存: {output_dir / 'model.pkl'}")
        
        @classmethod
        def load_model(cls, model_path: Path):
            """加载ML模型"""
            with open(model_path, 'rb') as f:
                params = pickle.load(f)
            
            instance = cls()
            model_type = params['model_type']
            
            if model_type == 'arima':
                instance.arima_threshold = params['threshold']
                instance.models['arima'] = 'trend_based'
            else:
                instance.models[model_type] = params['model']
            
            return instance, model_type


# ============================================================
# 训练器
# ============================================================

class Trainer:
    """
    模型训练器
    
    功能:
    - 训练循环
    - 早停机制
    - 学习率调度
    - 协方差加权损失（用于Smart-Trans）
    - 模型保存/加载
    """
    
    def __init__(
        self,
        model: BaseModel,
        config: dict = None,
        use_cov_weight: bool = False,
        cov_gamma: float = 1.0
    ):
        self.model = model.to(DEVICE)
        self.config = config or TRAIN_CONFIG
        self.use_cov_weight = use_cov_weight
        self.cov_gamma = cov_gamma
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(reduction='none')  # 返回每个样本的损失
        
        # 优化器
        self.optimizer = Adam(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.config['lr_scheduler_patience'],
            factor=self.config['lr_scheduler_factor']
        )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # 早停
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        训练一个epoch
        
        注意：如果使用协方差加权，DataLoader中的每个batch应包含3个元素 (X, y, corr)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_data in train_loader:
            # 支持两种数据格式：(X, y) 或 (X, y, corr)
            if len(batch_data) == 3:
                X, y, batch_corr = batch_data
                X, y, batch_corr = X.to(DEVICE), y.to(DEVICE), batch_corr.to(DEVICE)
                has_corr = True
            else:
                X, y = batch_data
                X, y = X.to(DEVICE), y.to(DEVICE)
                has_corr = False
            
            self.optimizer.zero_grad()
            
            # 前向传播
            logits = self.model(X)
            
            # 计算损失
            losses = self.criterion(logits, y)  # (batch,)
            
            # 协方差加权（可选）
            # 相关系数现在与数据一起打乱，索引正确对应
            if self.use_cov_weight and has_corr:
                # 计算权重: w_t = 1 + γ × max(0, ρ_t)
                weights = 1.0 + self.cov_gamma * torch.clamp(batch_corr, min=0)
                loss = (losses * weights).mean()
            else:
                loss = losses.mean()
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item() * len(y)
            _, predicted = logits.max(1)
            correct += predicted.eq(y).sum().item()
            total += len(y)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """验证（不使用协方差加权，只评估标准损失）"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                # 支持两种数据格式：(X, y) 或 (X, y, corr)
                if len(batch_data) == 3:
                    X, y, _ = batch_data  # 验证时忽略相关系数
                else:
                    X, y = batch_data
                
                X, y = X.to(DEVICE), y.to(DEVICE)
                
                logits = self.model(X)
                loss = self.criterion(logits, y).mean()
                
                total_loss += loss.item() * len(y)
                _, predicted = logits.max(1)
                correct += predicted.eq(y).sum().item()
                total += len(y)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        epochs: int = None
    ) -> Dict:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器（如使用协方差加权，每batch应为(X,y,corr)）
            val_loader: 验证数据加载器
            epochs: 训练轮数
            
        Returns:
            训练历史
        """
        epochs = epochs or self.config['max_epochs']
        
        print(f"\n  开始训练 {self.model.model_name}...")
        print(f"  设备: {DEVICE}")
        print(f"  参数量: {self.model.get_num_params():,}")
        if self.use_cov_weight:
            print(f"  协方差加权: γ={self.cov_gamma}")
        
        for epoch in range(1, epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 打印进度
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
            
            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['early_stopping_patience']:
                    print(f"  早停于 epoch {epoch}")
                    break
        
        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self.history


# ============================================================
# 评估器
# ============================================================

class Evaluator:
    """
    模型评估器
    
    指标:
    - Accuracy
    - Macro F1
    - Weighted F1
    - AUC (one-vs-rest)
    - 混淆矩阵
    """
    
    def __init__(self, model: BaseModel = None):
        self.model = model
    
    def predict(self, model: BaseModel, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取预测结果
        
        Returns:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率
        """
        model.eval()
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch_data in data_loader:
                # 支持两种数据格式：(X, y) 或 (X, y, corr)
                if len(batch_data) == 3:
                    X, y, _ = batch_data
                else:
                    X, y = batch_data
                
                X = X.to(DEVICE)
                
                logits = model(X)
                probs = F.softmax(logits, dim=1)
                _, preds = logits.max(1)
                
                all_labels.append(y.numpy())
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        y_prob = np.concatenate(all_probs)
        
        return y_true, y_pred, y_prob
    
    def compute_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: np.ndarray
    ) -> Dict:
        """
        计算评估指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率
            
        Returns:
            指标字典
        """
        metrics = {}
        
        if HAS_SKLEARN:
            # Accuracy
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            # F1 Scores
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
            metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
            
            # AUC (one-vs-rest)
            try:
                y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
                metrics['auc_macro'] = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
                metrics['auc_weighted'] = roc_auc_score(y_true_bin, y_prob, average='weighted', multi_class='ovr')
            except Exception as e:
                metrics['auc_macro'] = np.nan
                metrics['auc_weighted'] = np.nan
            
            # 混淆矩阵
            metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
            
            # 分类报告
            metrics['classification_report'] = classification_report(
                y_true, y_pred, 
                target_names=['下跌', '平稳', '上涨'],
                output_dict=True
            )
        else:
            # 简化版指标（不依赖sklearn）
            metrics['accuracy'] = (y_true == y_pred).mean()
            metrics['f1_macro'] = np.nan
            metrics['f1_weighted'] = np.nan
            metrics['auc_macro'] = np.nan
            metrics['auc_weighted'] = np.nan
            
            # 简化版混淆矩阵
            cm = np.zeros((3, 3), dtype=int)
            for t, p in zip(y_true, y_pred):
                cm[int(t), int(p)] += 1
            metrics['confusion_matrix'] = cm.tolist()
            metrics['classification_report'] = {}
        
        return metrics
    
    def evaluate(self, model: BaseModel, data_loader: DataLoader) -> Dict:
        """完整评估流程"""
        y_true, y_pred, y_prob = self.predict(model, data_loader)
        metrics = self.compute_metrics(y_true, y_pred, y_prob)
        return metrics
    
    def print_metrics(self, metrics: Dict, model_name: str = "Model"):
        """打印评估结果"""
        print(f"\n  {model_name} 评估结果:")
        print(f"  {'='*40}")
        print(f"  Accuracy:     {metrics['accuracy']:.4f}")
        print(f"  F1 (macro):   {metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted):{metrics['f1_weighted']:.4f}")
        print(f"  AUC (macro):  {metrics['auc_macro']:.4f}")
        print(f"  {'='*40}")
        print(f"  混淆矩阵:")
        cm = np.array(metrics['confusion_matrix'])
        print(f"           预测")
        print(f"         下跌  平稳  上涨")
        print(f"  真实 下跌 {cm[0,0]:4d} {cm[0,1]:4d} {cm[0,2]:4d}")
        print(f"       平稳 {cm[1,0]:4d} {cm[1,1]:4d} {cm[1,2]:4d}")
        print(f"       上涨 {cm[2,0]:4d} {cm[2,1]:4d} {cm[2,2]:4d}")


# ============================================================
# 模型工厂
# ============================================================

def create_model(model_name: str, input_dim: int, seq_len: int) -> BaseModel:
    """
    创建模型
    
    Args:
        model_name: 模型名称 ('lstm', 'gru', 'deeplob', 'transformer', 'smart_trans')
        input_dim: 输入特征维度
        seq_len: 序列长度
        
    Returns:
        模型实例
    """
    model_map = {
        'lstm': LSTMModel,
        'gru': GRUModel,
        'deeplob': DeepLOBModel,
        'transformer': TransformerModel,
        'smart_trans': SmartTransformer
    }
    
    if model_name not in model_map:
        raise ValueError(f"未知模型: {model_name}. 可选: {list(model_map.keys())}")
    
    return model_map[model_name](input_dim, seq_len)


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='模型训练模块')
    parser.add_argument('--data', type=str, default='data/processed/combined/dataset_T100_k20',
                        help='数据集目录')
    parser.add_argument('--model', type=str, nargs='+', default=['lstm'],
                        help='要训练的模型 (lstm, gru, deeplob, transformer, smart_trans, all, ml)')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=64, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--output', type=str, default='models', help='模型保存目录')
    parser.add_argument('--compare', action='store_true', help='对比所有模型')
    
    args = parser.parse_args()
    
    print("="*60)
    print("  OFI论文 - 模型训练模块")
    print("="*60)
    print(f"  设备: {DEVICE}")
    print(f"  数据目录: {args.data}")
    print(f"  训练模型: {args.model}")
    print(f"  训练轮数: {args.epochs}")
    
    # 加载数据
    print("\n[1] 加载数据...")
    splits = load_dataset(Path(args.data))
    
    X_train, y_train = splits['train']
    input_dim = X_train.shape[2]
    seq_len = X_train.shape[1]
    print(f"  输入维度: {input_dim}, 序列长度: {seq_len}")
    
    # 尝试加载相关系数（用于Smart-Transformer的协方差加权）
    correlations = None
    data_dir = Path(args.data)
    corr_file = data_dir / 'correlations.npy'
    if corr_file.exists():
        print(f"  发现相关系数文件: {corr_file}")
        corr_data = np.load(corr_file, allow_pickle=True).item()
        if isinstance(corr_data, dict):
            correlations = corr_data
            print(f"  已加载相关系数 (train: {len(correlations.get('train', []))} 样本)")
        else:
            print(f"  [WARN] 相关系数格式不匹配，跳过")
    
    # 确定要训练的模型
    if 'all' in args.model:
        model_names = ['lstm', 'gru', 'deeplob', 'transformer', 'smart_trans']
    else:
        model_names = args.model
    
    # 训练结果存储
    all_results = {}
    evaluator = Evaluator()
    
    # 训练深度学习模型
    for model_name in model_names:
        if model_name == 'ml':
            continue
            
        print(f"\n{'='*60}")
        print(f"  训练模型: {model_name.upper()}")
        print("="*60)
        
        # 创建模型
        model = create_model(model_name, input_dim, seq_len)
        print(f"  参数量: {model.get_num_params():,}")
        
        # 决定是否使用协方差加权
        use_cov_weight = (model_name == 'smart_trans') and (correlations is not None)
        
        # 为Smart-Trans创建带相关系数的DataLoader，其他模型使用普通DataLoader
        if use_cov_weight:
            print("  启用协方差加权训练")
            loaders = create_dataloaders(splits, batch_size=args.batch_size, correlations=correlations)
        else:
            loaders = create_dataloaders(splits, batch_size=args.batch_size)
        
        # 训练
        trainer = Trainer(model, use_cov_weight=use_cov_weight)
        
        history = trainer.train(
            loaders['train'],
            loaders['val'],
            epochs=args.epochs
        )
        
        # 评估
        print("\n[测试集评估]")
        metrics = evaluator.evaluate(model, loaders['test'])
        evaluator.print_metrics(metrics, model_name.upper())
        
        # 保存模型
        output_dir = Path(args.output) / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model.save(output_dir / 'model.pt')
        
        with open(output_dir / 'metrics.json', 'w') as f:
            # 移除不可序列化的项
            save_metrics = {k: v for k, v in metrics.items() if k != 'classification_report'}
            save_metrics['history'] = history
            json.dump(save_metrics, f, indent=2)
        
        all_results[model_name] = metrics
    
    # 训练机器学习模型
    if 'ml' in args.model and HAS_ML:
        print(f"\n{'='*60}")
        print("  训练机器学习基准模型")
        print("="*60)
        
        ml_baselines = MLBaselines()
        X_train, y_train = splits['train']
        X_test, y_test = splits['test']
        
        for ml_name, train_func in [
            ('arima', ml_baselines.train_arima),
            ('logistic', ml_baselines.train_logistic),
            ('xgboost', ml_baselines.train_xgboost),
            ('rf', ml_baselines.train_rf)
        ]:
            print(f"\n  训练 {ml_name}...")
            result = train_func(X_train, y_train)
            
            if result is None:
                continue
            
            y_pred = ml_baselines.predict(ml_name, X_test)
            y_prob = ml_baselines.predict_proba(ml_name, X_test)
            
            metrics = evaluator.compute_metrics(y_test, y_pred, y_prob)
            evaluator.print_metrics(metrics, ml_name.upper())
            all_results[ml_name] = metrics
            
            # 保存ML模型
            ml_output_dir = Path(args.output) / ml_name
            ml_baselines.save_model(ml_name, ml_output_dir)
            
            # 保存指标
            with open(ml_output_dir / 'metrics.json', 'w') as f:
                save_metrics = {k: v for k, v in metrics.items() if k != 'classification_report'}
                json.dump(save_metrics, f, indent=2)
    
    # 对比结果
    if args.compare or len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  模型对比")
        print("="*60)
        
        comparison = []
        for name, metrics in all_results.items():
            comparison.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'F1_macro': metrics['f1_macro'],
                'F1_weighted': metrics['f1_weighted'],
                'AUC_macro': metrics.get('auc_macro', np.nan)
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('Accuracy', ascending=False)
        print(df.to_string(index=False))
        
        # 保存对比结果
        df.to_csv(Path(args.output) / 'comparison.csv', index=False)
    
    print("\n[DONE] 训练完成！")


if __name__ == "__main__":
    main()
