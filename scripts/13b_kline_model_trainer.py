"""
K线模型训练模块
包含论文第三章定义的创新模型架构

模型架构（与论文第三章第四节对齐）:
1. PV-CrossAttention: 量价交叉注意力，显式建模价格与成交量关系
2. LSF (Learnable Scale Fusion): 可学习多尺度融合，自适应加权不同时间粒度
3. Learnable Positional Encoding: 可学习位置编码

使用方法:
    python 13b_kline_model_trainer.py --code HK.00700 --model pv_transformer
    python 13b_kline_model_trainer.py --code HK.00700 --model multi_scale --epochs 100
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

# 解决Windows编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

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

# sklearn
try:
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ============================================================
# 配置
# ============================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# K线特征配置（与论文表3.3-1对齐，共22维）
# K线形态特征（2维）
KLINE_FEATURES = ['kline_position', 'range_pct']
# 价格相关特征（5维）
PRICE_FEATURES = ['return_1', 'return_5', 'return_20', 'return_60', 'return_zscore']
# 波动率特征（2维）
VOLATILITY_FEATURES = ['atr_pct', 'volatility_20']
# 成交不平衡特征（4维）
TI_FEATURES = ['ti', 'ti_5', 'ti_60', 'ti_zscore']
# 成交量特征（3维）
VOLUME_FEATURES = ['relative_volume', 'volume_change', 'pv_corr']
# 技术指标（5维）
TECH_FEATURES = ['rsi', 'bb_position', 'macd_dif', 'macd_dea', 'macd']
# 市场状态（1维）
REGIME_FEATURES = ['market_regime']

# 全部特征（22维，与论文表3.3-1对齐）
ALL_FEATURES = (KLINE_FEATURES + PRICE_FEATURES + VOLATILITY_FEATURES + 
                TI_FEATURES + VOLUME_FEATURES + TECH_FEATURES + REGIME_FEATURES)

# 价格相关特征（用于PV-CrossAttention的Query）
PRICE_RELATED = KLINE_FEATURES + PRICE_FEATURES + VOLATILITY_FEATURES + TECH_FEATURES
# 成交量相关特征（用于PV-CrossAttention的Key/Value）
VOLUME_RELATED = TI_FEATURES + VOLUME_FEATURES + REGIME_FEATURES

# 模型超参数（与论文表3.4-3对齐）
MODEL_CONFIG = {
    'pv_transformer': {
        'd_model': 256,      # 论文表3.4-3: 输出维度256
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 512,
        'dropout': 0.1,
    },
    'multi_scale': {
        'd_model': 64,       # 缩小模型以适应多尺度数据量（~5000样本）
        'nhead': 4,
        'num_layers': 1,
        'dropout': 0.2,      # 加大dropout防过拟合
    },
    'transformer': {         # 原生Transformer基线（论文表3.4-2a）
        'd_model': 256,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 512,
        'dropout': 0.1,
    },
    'lstm': {                # LSTM基线（论文表3.4-2a）
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.2,
    },
    'gru': {                 # GRU基线（论文表3.4-2a）
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.2,
    },
}

# 训练超参数（与论文第三章第四节对齐）
TRAIN_CONFIG = {
    'batch_size': 64,
    'learning_rate': 1e-4,
    'max_epochs': 100,
    'early_stopping_patience': 10,  # 论文：验证集损失连续10个epoch不下降时终止
    'weight_decay': 1e-5,
    'warmup_epochs': 5,
}

NUM_CLASSES = 3  # 下跌(-1→0), 平稳(0→1), 上涨(+1→2)

# 路径
DATA_DIR = Path("data/datasets")
MODEL_DIR = Path("models")


# ============================================================
# 可学习位置编码 (Learnable Positional Encoding)
# ============================================================

class LearnablePositionalEncoding(nn.Module):
    """
    可学习位置编码（论文第三章第四节）
    
    相比固定的Sinusoidal编码，可学习编码能够自适应金融时序数据的时间依赖模式
    """
    
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 可学习的位置嵌入
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            带位置编码的张量
        """
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return self.dropout(x)


# ============================================================
# PV-CrossAttention 模块（论文创新点1）
# ============================================================

class PVCrossAttention(nn.Module):
    """
    量价交叉注意力模块 (Price-Volume Cross-Attention)
    
    论文公式3.4-1:
    CrossAttn(Q_price, K_volume, V_volume) = softmax(Q_price × K_volume^T / √d) × V_volume
    
    核心思想：
    - 价格序列作为Query，询问"哪些成交量变化与当前价格走势相关"
    - 成交量序列作为Key/Value，提供供需失衡信息
    - 输出：价格变化的成交量驱动因子
    """
    
    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        assert d_model % nhead == 0, "d_model必须能被nhead整除"
        
        # 投影层
        self.W_q = nn.Linear(d_model, d_model)  # Price → Query
        self.W_k = nn.Linear(d_model, d_model)  # Volume → Key
        self.W_v = nn.Linear(d_model, d_model)  # Volume → Value
        self.W_o = nn.Linear(d_model, d_model)  # 输出投影
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self, 
        price_features: torch.Tensor, 
        volume_features: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            price_features: (batch, seq_len, d_model) 价格特征
            volume_features: (batch, seq_len, d_model) 成交量特征
            return_attention: 是否返回注意力权重（用于可视化）
            
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, nhead, seq_len, seq_len) if return_attention
        """
        batch_size, seq_len, _ = price_features.shape
        
        # 多头投影
        Q = self.W_q(price_features).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        K = self.W_k(volume_features).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = self.W_v(volume_features).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, V)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(attn_output)
        
        if return_attention:
            return output, attn_weights
        return output


# ============================================================
# LSF 模块（论文创新点2）
# ============================================================

class LearnableScaleFusion(nn.Module):
    """
    可学习尺度融合模块 (Learnable Scale Fusion, LSF)
    
    论文公式3.4-2:
    h_fused = Σ_s (w_s × h_s), 其中 w = softmax(MLP(concat([h_1, ..., h_S])))
    
    特点：
    - 权重w之和为1（softmax保证）
    - 权重可解释：反映各时间尺度的贡献度
    - 动态权重：根据输入自适应调整
    """
    
    def __init__(self, d_model: int, num_scales: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_scales = num_scales
        self.d_model = d_model
        
        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(d_model * num_scales, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_scales),
        )
    
    def forward(
        self, 
        scale_features: List[torch.Tensor],
        return_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            scale_features: [(batch, d_model), ...] 各尺度特征列表
            return_weights: 是否返回融合权重（用于可解释性）
            
        Returns:
            fused: (batch, d_model) 融合后的特征
            weights: (batch, num_scales) if return_weights
        """
        # 拼接所有尺度特征
        concat = torch.cat(scale_features, dim=-1)  # (batch, d_model * num_scales)
        
        # 计算门控权重
        gate_logits = self.gate(concat)  # (batch, num_scales)
        weights = F.softmax(gate_logits, dim=-1)  # (batch, num_scales)
        
        # 加权融合
        stacked = torch.stack(scale_features, dim=-1)  # (batch, d_model, num_scales)
        weights_expanded = weights.unsqueeze(1)  # (batch, 1, num_scales)
        fused = (stacked * weights_expanded).sum(dim=-1)  # (batch, d_model)
        
        if return_weights:
            return fused, weights
        return fused


# ============================================================
# PV-Transformer 单尺度模型
# ============================================================

class PVTransformer(nn.Module):
    """
    量价交叉注意力Transformer (论文第三章第四节)
    
    架构:
    1. 特征分离: 价格特征 / 成交量特征
    2. 独立嵌入: 各自投影到d_model维度
    3. PV-CrossAttention: 交叉注意力建模量价关系
    4. Transformer Encoder: 时序依赖建模
    5. 分类头: 三分类预测
    """
    
    def __init__(
        self,
        price_dim: int,
        volume_dim: int,
        seq_len: int,
        num_classes: int = NUM_CLASSES,
        config: dict = None
    ):
        super().__init__()
        cfg = config or MODEL_CONFIG['pv_transformer']
        self.d_model = cfg['d_model']
        self.model_name = "pv_transformer"
        
        # 特征嵌入层
        self.price_embedding = nn.Linear(price_dim, self.d_model)
        self.volume_embedding = nn.Linear(volume_dim, self.d_model)
        
        # 可学习位置编码
        self.pos_encoder = LearnablePositionalEncoding(self.d_model, max_len=seq_len + 1)
        
        # PV-CrossAttention
        self.pv_cross_attn = PVCrossAttention(self.d_model, nhead=cfg['nhead'], dropout=cfg['dropout'])
        
        # 融合层（价格 + 交叉注意力输出）
        self.fusion = nn.Linear(self.d_model * 2, self.d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=cfg['nhead'],
            dim_feedforward=cfg['dim_feedforward'],
            dropout=cfg['dropout'],
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg['num_layers'])
        
        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(self.d_model // 2, num_classes)
        )
    
    def forward(
        self, 
        price_features: torch.Tensor, 
        volume_features: torch.Tensor,
        return_attention: bool = False
    ):
        """
        Args:
            price_features: (batch, seq_len, price_dim)
            volume_features: (batch, seq_len, volume_dim)
        """
        batch_size = price_features.size(0)
        
        # 嵌入
        price_emb = self.price_embedding(price_features)
        volume_emb = self.volume_embedding(volume_features)
        
        # PV-CrossAttention
        if return_attention:
            cross_attn_out, attn_weights = self.pv_cross_attn(price_emb, volume_emb, return_attention=True)
        else:
            cross_attn_out = self.pv_cross_attn(price_emb, volume_emb)
        
        # 融合价格特征和交叉注意力输出
        fused = self.fusion(torch.cat([price_emb, cross_attn_out], dim=-1))
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, fused], dim=1)
        
        # 位置编码 + Transformer
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # 取CLS token输出进行分类
        cls_output = x[:, 0, :]
        logits = self.classifier(cls_output)
        
        if return_attention:
            return logits, attn_weights
        return logits


# ============================================================
# 多尺度PV-Transformer（含LSF）
# ============================================================

class MultiScalePVTransformer(nn.Module):
    """
    多尺度量价交叉注意力Transformer
    
    架构:
    1. 各尺度独立的PV-Transformer编码器
    2. LSF模块融合多尺度特征
    3. 分类头
    """
    
    def __init__(
        self,
        price_dim: int,
        volume_dim: int,
        scale_seq_lens: Dict[str, int],  # {'1M': 60, '5M': 24, '60M': 12}
        num_classes: int = NUM_CLASSES,
        config: dict = None
    ):
        super().__init__()
        cfg = config or MODEL_CONFIG['multi_scale']
        self.d_model = cfg['d_model']
        self.scale_names = list(scale_seq_lens.keys())
        self.model_name = "multi_scale_pv_transformer"
        
        # 各尺度编码器
        self.scale_encoders = nn.ModuleDict()
        for scale, seq_len in scale_seq_lens.items():
            self.scale_encoders[scale] = self._build_scale_encoder(
                price_dim, volume_dim, seq_len, cfg
            )
        
        # LSF融合模块
        self.lsf = LearnableScaleFusion(self.d_model, num_scales=len(scale_seq_lens))
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(self.d_model, num_classes)
        )
    
    def _build_scale_encoder(self, price_dim, volume_dim, seq_len, cfg):
        """构建单尺度编码器"""
        return nn.ModuleDict({
            'price_emb': nn.Linear(price_dim, self.d_model),
            'volume_emb': nn.Linear(volume_dim, self.d_model),
            'cross_attn': PVCrossAttention(self.d_model, nhead=cfg['nhead'], dropout=cfg['dropout']),
            'fusion': nn.Linear(self.d_model * 2, self.d_model),
            'pos_enc': LearnablePositionalEncoding(self.d_model, max_len=seq_len),
            'transformer': nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=cfg['nhead'],
                    dim_feedforward=self.d_model * 2,
                    dropout=cfg['dropout'],
                    batch_first=True
                ),
                num_layers=cfg['num_layers']
            ),
            'pool': nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        })
    
    def encode_scale(self, encoder, price_feat, volume_feat):
        """编码单个尺度"""
        price_emb = encoder['price_emb'](price_feat)
        volume_emb = encoder['volume_emb'](volume_feat)
        cross_out = encoder['cross_attn'](price_emb, volume_emb)
        fused = encoder['fusion'](torch.cat([price_emb, cross_out], dim=-1))
        fused = encoder['pos_enc'](fused)
        encoded = encoder['transformer'](fused)
        # 全局池化 -> (batch, d_model)
        pooled = encoder['pool'](encoded.transpose(1, 2)).squeeze(-1)
        return pooled
    
    def forward(
        self, 
        scale_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        return_weights: bool = False
    ):
        """
        Args:
            scale_data: {
                '1M': (price_features, volume_features),
                '5M': ...,
                ...
            }
        """
        scale_features = []
        for scale in self.scale_names:
            if scale in scale_data:
                price_feat, volume_feat = scale_data[scale]
                encoded = self.encode_scale(self.scale_encoders[scale], price_feat, volume_feat)
                scale_features.append(encoded)
        
        # LSF融合
        if return_weights:
            fused, weights = self.lsf(scale_features, return_weights=True)
        else:
            fused = self.lsf(scale_features)
        
        # 分类
        logits = self.classifier(fused)
        
        if return_weights:
            return logits, weights
        return logits


# ============================================================
# 基准模型：LSTM
# ============================================================

class LSTMBaseline(nn.Module):
    """LSTM基准模型"""
    
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int = 128, 
                 num_layers: int = 2, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.model_name = "lstm"
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.classifier(last_output)


# ============================================================
# 基准模型：GRU（论文表3.4-2a）
# ============================================================

class GRUBaseline(nn.Module):
    """GRU基准模型（论文表3.4-2a：2层堆叠，hidden=128，dropout=0.2）"""
    
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int = 128, 
                 num_layers: int = 2, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.model_name = "gru"
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        return self.classifier(last_output)


# ============================================================
# 基准模型：原生Transformer（论文表3.4-2a）
# ============================================================

class TransformerBaseline(nn.Module):
    """原生Transformer基准模型（论文表3.4-2a：4层Encoder，d_model=256，nhead=8）"""
    
    def __init__(self, input_dim: int, seq_len: int, num_classes: int = NUM_CLASSES,
                 config: dict = None):
        super().__init__()
        cfg = config or MODEL_CONFIG['transformer']
        self.model_name = "transformer"
        self.d_model = cfg['d_model']
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, self.d_model)
        
        # 可学习位置编码
        self.pos_encoder = LearnablePositionalEncoding(self.d_model, max_len=seq_len + 1)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=cfg['nhead'],
            dim_feedforward=cfg['dim_feedforward'],
            dropout=cfg['dropout'],
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg['num_layers'])
        
        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(self.d_model // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        
        # 投影到d_model维度
        x = self.input_projection(x)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 位置编码 + Transformer
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # 取CLS token输出进行分类
        cls_output = x[:, 0, :]
        return self.classifier(cls_output)


# ============================================================
# 基准模型：CNN-LSTM（论文表3.4-2a）
# ============================================================

class CNNLSTMBaseline(nn.Module):
    """
    CNN-LSTM混合模型（论文表3.4-2a）
    
    架构：CNN(3层) + LSTM(1层)
    - CNN层提取局部特征模式
    - LSTM层捕捉时序依赖
    """
    
    def __init__(self, input_dim: int, seq_len: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.model_name = "cnn_lstm"
        
        # CNN层（3层卷积）
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # LSTM层（1层）
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, input_dim)
        # CNN需要 (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # CNN特征提取
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # 转回 (batch, seq_len, channels) 给LSTM
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        return self.classifier(last_output)


# ============================================================
# sklearn基准模型包装器（论文表3.4-1）
# ============================================================

class SklearnModelWrapper:
    """
    sklearn模型包装器，提供统一接口
    
    支持模型（论文表3.4-1）：
    - LogisticRegression: 线性模型基准
    - RandomForest: 非线性机器学习基准
    - XGBoost: 梯度提升基准
    """
    
    def __init__(self, model_type: str = 'xgboost', num_classes: int = NUM_CLASSES, **kwargs):
        self.model_type = model_type
        self.model_name = model_type
        self.num_classes = num_classes
        self.model = None
        self.kwargs = kwargs
        self._init_model()
    
    def _init_model(self):
        if self.model_type == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(
                C=self.kwargs.get('C', 1.0),
                max_iter=1000,
                solver='lbfgs',
                random_state=42
            )
        elif self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=self.kwargs.get('n_estimators', 300),
                max_depth=self.kwargs.get('max_depth', 10),
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                # 使用hist方法（CPU优化，比默认快很多）
                use_gpu = torch.cuda.is_available()
                self.model = xgb.XGBClassifier(
                    n_estimators=self.kwargs.get('n_estimators', 300),
                    max_depth=self.kwargs.get('max_depth', 6),
                    learning_rate=self.kwargs.get('learning_rate', 0.1),
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='multi:softprob',
                    num_class=self.num_classes,
                    random_state=42,
                    tree_method='hist',
                    device='cuda' if use_gpu else 'cpu',
                    n_jobs=-1,
                    eval_metric='mlogloss'
                )
                print(f"  [INFO] XGBoost using hist method with {'GPU (cuda)' if use_gpu else 'CPU multi-core'}")
            except ImportError:
                print("[WARN] XGBoost not installed, falling back to RandomForest")
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(n_estimators=300, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练模型（输入需要展平为2D）"""
        # 将3D (batch, seq, features) 展平为 2D (batch, seq*features)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self.model.score(X, y)


# ============================================================
# 模型工厂函数
# ============================================================

def create_model(model_name: str, input_dim: int, seq_len: int, 
                 num_classes: int = NUM_CLASSES, **kwargs):
    """
    创建模型的工厂函数
    
    Args:
        model_name: 模型名称 ['pv_transformer', 'multi_scale', 'lstm', 'gru', 'cnn_lstm',
                             'transformer', 'logistic_regression', 'random_forest', 'xgboost']
        input_dim: 输入特征维度
        seq_len: 序列长度
        num_classes: 分类数量
        
    Returns:
        model: 模型实例
    """
    model_name = model_name.lower()
    
    if model_name == 'lstm':
        return LSTMBaseline(input_dim, seq_len, num_classes=num_classes)
    
    elif model_name == 'gru':
        return GRUBaseline(input_dim, seq_len, num_classes=num_classes)
    
    elif model_name == 'cnn_lstm':
        return CNNLSTMBaseline(input_dim, seq_len, num_classes=num_classes)
    
    elif model_name == 'transformer':
        return TransformerBaseline(input_dim, seq_len, num_classes=num_classes)
    
    elif model_name == 'pv_transformer':
        # 需要分离价格和成交量特征
        price_dim = len(PRICE_RELATED)
        volume_dim = len(VOLUME_RELATED)
        return PVTransformer(price_dim, volume_dim, seq_len, num_classes=num_classes)
    
    elif model_name == 'multi_scale':
        # 多尺度模型需要不同的输入格式
        price_dim = len(PRICE_RELATED)
        volume_dim = len(VOLUME_RELATED)
        scale_seq_lens = kwargs.get('scale_seq_lens', {'1M': 60, '5M': 24, '60M': 12})
        return MultiScalePVTransformer(price_dim, volume_dim, scale_seq_lens, num_classes=num_classes)
    
    elif model_name in ['logistic_regression', 'random_forest', 'xgboost']:
        return SklearnModelWrapper(model_name, num_classes=num_classes, **kwargs)
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: "
                        f"lstm, gru, cnn_lstm, transformer, pv_transformer, multi_scale, "
                        f"logistic_regression, random_forest, xgboost")


# ============================================================
# 训练器
# ============================================================

class KlineModelTrainer:
    """K线模型训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = DEVICE,
        config: dict = None,
        class_weights: torch.Tensor = None
    ):
        self.model = model.to(device)
        self.device = device
        self.cfg = config or TRAIN_CONFIG
        
        # 论文：Adam优化器 (β1=0.9, β2=0.999)
        self.optimizer = Adam(
            model.parameters(),
            lr=self.cfg['learning_rate'],
            betas=(0.9, 0.999),
            weight_decay=self.cfg['weight_decay']
        )
        
        # 论文：ReduceLROnPlateau调度策略 (patience=5, factor=0.5)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # 加权交叉熵损失（缓解类别不平衡）
        # 参考: DeepLOB (Zhang et al., 2019) 使用 weighted categorical cross-entropy
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            print(f"  [Loss] Weighted CrossEntropyLoss, weights={class_weights.tolist()}")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print(f"  [Loss] CrossEntropyLoss (unweighted)")
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
        self.best_val_f1 = 0
        self.patience_counter = 0
    
    def _prepare_input(self, batch):
        """准备模型输入，支持单尺度和多尺度"""
        if isinstance(batch[0], dict):
            # 多尺度数据：batch[0] = {'1M': tensor, '5M': tensor, ...}
            scale_data = {}
            for scale, X in batch[0].items():
                X = X.to(self.device)
                # 分离价格和成交量特征
                price_dim = len(PRICE_RELATED)
                price_x = X[:, :, :price_dim]
                volume_x = X[:, :, price_dim:]
                scale_data[scale] = (price_x, volume_x)
            y = batch[1].to(self.device)
            return scale_data, y, True  # is_multi_scale=True
        else:
            # 单尺度数据
            X = batch[0].to(self.device)
            y = batch[1].to(self.device)
            return X, y, False
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            data, y, is_multi_scale = self._prepare_input(batch)
            
            self.optimizer.zero_grad()
            
            if is_multi_scale:
                # 多尺度模型
                logits = self.model(data)
            elif hasattr(self.model, 'model_name') and 'pv' in self.model.model_name:
                # 单尺度PV-Transformer需要分离价格和成交量特征
                price_dim = len(PRICE_RELATED)
                price_x = data[:, :, :price_dim]
                volume_x = data[:, :, price_dim:]
                logits = self.model(price_x, volume_x)
            else:
                logits = self.model(data)
            
            loss = self.criterion(logits, y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                data, y, is_multi_scale = self._prepare_input(batch)
                
                if is_multi_scale:
                    logits = self.model(data)
                elif hasattr(self.model, 'model_name') and 'pv' in self.model.model_name:
                    price_dim = len(PRICE_RELATED)
                    price_x = data[:, :, :price_dim]
                    volume_x = data[:, :, price_dim:]
                    logits = self.model(price_x, volume_x)
                else:
                    logits = self.model(data)
                
                loss = self.criterion(logits, y)
                total_loss += loss.item()
                
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = None,
        save_path: Path = None
    ):
        """完整训练流程"""
        max_epochs = max_epochs or self.cfg['max_epochs']
        patience = self.cfg['early_stopping_patience']
        
        print(f"\n开始训练 (max_epochs={max_epochs}, patience={patience})")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, max_epochs + 1):
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = self.evaluate(val_loader)
            
            # 学习率调度（基于验证损失）
            self.scheduler.step(val_metrics['loss'])
            
            # 记录
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1_macro'])
            
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val F1: {val_metrics['f1_macro']:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping
            if val_metrics['f1_macro'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1_macro']
                self.patience_counter = 0
                if save_path:
                    self.save_model(save_path)
                    print(f"  [BEST] 模型已保存")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\n[STOP] Early stopping at epoch {epoch}")
                    break
        
        return self.history
    
    def save_model(self, path: Path):
        """保存模型"""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': getattr(self.model, 'model_name', 'unknown'),
            'best_val_f1': self.best_val_f1,
            'history': self.history
        }, path)
    
    def load_model(self, path: Path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)


# ============================================================
# 类别权重计算
# ============================================================

def compute_class_weights(labels: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    """
    根据训练集标签分布计算类别权重（逆频率法）
    
    权重公式: w_c = N / (K * n_c)
    其中 N=总样本数, K=类别数, n_c=第c类样本数
    
    参考: DeepLOB (Zhang et al., 2019) 使用加权交叉熵缓解类别不平衡
    
    Args:
        labels: 标签数组（PyTorch格式: 0/1/2）
        num_classes: 类别数
    
    Returns:
        weights: 类别权重张量
    """
    counts = np.bincount(labels.astype(int), minlength=num_classes)
    total = len(labels)
    
    # 逆频率权重
    weights = total / (num_classes * counts.astype(float))
    # 防止除零
    weights = np.where(counts == 0, 1.0, weights)
    
    print(f"  [ClassWeights] counts={counts.tolist()}, weights={[f'{w:.3f}' for w in weights]}")
    return torch.FloatTensor(weights)


# ============================================================
# 数据加载
# ============================================================

def load_kline_dataset(dataset_path: Path) -> Dict:
    """加载K线数据集"""
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    print(f"数据集加载完成: {dataset_path}")
    return data


def is_multi_scale_dataset(dataset: Dict) -> bool:
    """判断是否为多尺度数据集"""
    if 'train' in dataset:
        train_data = dataset['train']
        # 多尺度数据集的train是dict，包含'1M','5M'等key
        if isinstance(train_data, dict) and '1M' in train_data:
            return True
    return False


def create_dataloaders(
    dataset: Dict,
    batch_size: int = TRAIN_CONFIG['batch_size'],
    feature_order: List[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建DataLoader（单尺度）"""
    
    loaders = {}
    for split in ['train', 'val', 'test']:
        if split in dataset:
            ds = dataset[split]
            if hasattr(ds, 'X'):
                X, y = ds.X, ds.y
            else:
                X, y = ds
            
            # 确保标签从0开始
            if isinstance(y, torch.Tensor):
                y_np = y.numpy()
            else:
                y_np = y
            
            if y_np.min() == -1:
                y_np = y_np + 1
            
            tensor_ds = TensorDataset(
                torch.FloatTensor(X) if not isinstance(X, torch.Tensor) else X,
                torch.LongTensor(y_np)
            )
            
            loaders[split] = DataLoader(
                tensor_ds,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=0,
                pin_memory=False  # 禁用以避免内存问题
            )
            print(f"  {split}: {len(tensor_ds)} 样本")
    
    return loaders.get('train'), loaders.get('val'), loaders.get('test')


class MultiScaleDataset(Dataset):
    """多尺度数据集（用于MultiScalePVTransformer）"""
    
    def __init__(self, scale_data: Dict[str, np.ndarray], labels: np.ndarray):
        """
        Args:
            scale_data: {'1M': X_1m, '5M': X_5m, ...}
            labels: 标签数组
        """
        self.scale_data = {k: torch.FloatTensor(v) for k, v in scale_data.items() if k != 'labels'}
        self.labels = torch.LongTensor(labels)
        self.scales = [k for k in scale_data.keys() if k != 'labels']
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # 返回 {scale: X[idx]} 和 label
        scale_batch = {scale: self.scale_data[scale][idx] for scale in self.scales}
        return scale_batch, self.labels[idx]


def multi_scale_collate_fn(batch):
    """多尺度数据集的collate函数"""
    scale_data = {}
    labels = []
    
    scales = batch[0][0].keys()
    for scale in scales:
        scale_data[scale] = torch.stack([item[0][scale] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    
    return scale_data, labels


def create_multi_scale_dataloaders(
    dataset: Dict,
    batch_size: int = TRAIN_CONFIG['batch_size']
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建多尺度DataLoader"""
    
    loaders = {}
    for split in ['train', 'val', 'test']:
        if split in dataset:
            split_data = dataset[split]
            
            # 获取标签
            labels = split_data.get('labels')
            if labels is None:
                print(f"  [WARN] {split} 没有labels，跳过")
                continue
            
            # 确保标签从0开始
            if hasattr(labels, 'min') and labels.min() == -1:
                labels = labels + 1
            
            # 提取各尺度特征（排除labels）
            scale_features = {k: v for k, v in split_data.items() if k in ['1M', '5M', '60M']}
            
            if not scale_features:
                print(f"  [WARN] {split} 没有尺度特征，跳过")
                continue
            
            ms_dataset = MultiScaleDataset(scale_features, labels)
            
            loaders[split] = DataLoader(
                ms_dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=0,
                pin_memory=False,
                collate_fn=multi_scale_collate_fn
            )
            print(f"  {split}: {len(ms_dataset)} 样本 (多尺度)")
    
    return loaders.get('train'), loaders.get('val'), loaders.get('test')


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='K线模型训练')
    parser.add_argument('--dataset', type=str, help='数据集路径')
    parser.add_argument('--code', type=str, default='HK.00700', help='股票代码')
    parser.add_argument('--ktype', type=str, default='1M', help='K线类型')
    parser.add_argument('--model', type=str, default='pv_transformer',
                        choices=['pv_transformer', 'multi_scale', 'lstm', 'gru', 'cnn_lstm', 
                                 'transformer', 'logistic_regression', 'random_forest', 'xgboost'])
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['max_epochs'])
    parser.add_argument('--batch-size', type=int, default=TRAIN_CONFIG['batch_size'])
    parser.add_argument('--lr', type=float, default=TRAIN_CONFIG['learning_rate'])
    
    args = parser.parse_args()
    
    print("="*60)
    print("  K线模型训练")
    print("="*60)
    print(f"  设备: {DEVICE}")
    print(f"  模型: {args.model}")
    
    # 加载数据集
    if args.dataset:
        dataset_path = Path(args.dataset)
    else:
        code_str = args.code.replace('.', '_')
        # multi_scale模型使用multi_scale数据集
        if args.model == 'multi_scale':
            dataset_path = DATA_DIR / f"dataset_{code_str}_multi_scale.pkl"
        else:
            dataset_path = DATA_DIR / f"dataset_{code_str}_{args.ktype}.pkl"
    
    if not dataset_path.exists():
        print(f"[ERROR] 数据集不存在: {dataset_path}")
        if args.model == 'multi_scale':
            print("请先运行: python scripts/12b_kline_dataset_builder.py --code HK.00700 --multi-scale")
        else:
            print("请先运行: python 12b_kline_dataset_builder.py")
        return
    
    dataset = load_kline_dataset(dataset_path)
    
    # 检测是否为多尺度数据集
    is_multi_scale = is_multi_scale_dataset(dataset)
    
    if is_multi_scale:
        print("  检测到多尺度数据集")
        train_loader, val_loader, test_loader = create_multi_scale_dataloaders(dataset, args.batch_size)
        # 多尺度模型 - 不需要input_dim和seq_len，内部自动配置
        input_dim = len(ALL_FEATURES)
        seq_len = 60  # dummy，multi_scale模型内部会根据scale配置
        model = create_model('multi_scale', input_dim=input_dim, seq_len=seq_len)
    else:
        train_loader, val_loader, test_loader = create_dataloaders(dataset, args.batch_size)
        # 获取输入维度
        sample_X, _ = next(iter(train_loader))
        seq_len, input_dim = sample_X.shape[1], sample_X.shape[2]
        print(f"  输入维度: seq_len={seq_len}, features={input_dim}")
        # 创建模型（使用工厂函数）
        model = create_model(args.model, input_dim=input_dim, seq_len=seq_len)
    
    # 计算类别权重（从训练集标签）
    print("\n计算类别权重...")
    if is_multi_scale:
        train_labels = dataset['train'].get('labels', np.array([]))
    else:
        train_labels = dataset['train'][1] if isinstance(dataset['train'], tuple) else np.array([])
    
    if len(train_labels) > 0:
        # 确保标签是 PyTorch 格式 {0,1,2}
        if hasattr(train_labels, 'min') and train_labels.min() < 0:
            train_labels_for_weights = (train_labels + 1).astype(int)
        else:
            train_labels_for_weights = train_labels.astype(int)
        class_weights = compute_class_weights(train_labels_for_weights, num_classes=3)
    else:
        class_weights = None
        print("  [WARN] 无法获取训练标签，使用等权损失函数")
    
    # sklearn模型使用不同的训练流程
    sklearn_models = ['logistic_regression', 'random_forest', 'xgboost']
    trainer = None
    if args.model in sklearn_models:
        print(f"\n训练sklearn模型: {args.model}")
        # 获取numpy数据
        X_train = np.vstack([batch[0].numpy() for batch in train_loader])
        y_train = np.hstack([batch[1].numpy() for batch in train_loader])
        X_val = np.vstack([batch[0].numpy() for batch in val_loader])
        y_val = np.hstack([batch[1].numpy() for batch in val_loader])
        
        # sklearn模型的类别平衡通过 class_weight='balanced' 参数实现
        if hasattr(model, 'set_params'):
            try:
                model.set_params(class_weight='balanced')
                print("  [sklearn] class_weight='balanced' applied")
            except (TypeError, ValueError):
                pass
        
        # 训练
        model.fit(X_train, y_train)
        
        # 评估
        train_acc = model.score(X_train, y_train)
        val_acc = model.score(X_val, y_val)
        print(f"  Train Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f}")
        
        # 保存
        save_path = MODEL_DIR / args.model / f"model_{args.code.replace('.', '_')}_{args.ktype}.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  模型已保存: {save_path}")
        
        # sklearn测试集评估
        if test_loader:
            X_test = np.vstack([batch[0].numpy() for batch in test_loader])
            y_test = np.hstack([batch[1].numpy() for batch in test_loader])
            
            from sklearn.metrics import accuracy_score, f1_score
            y_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            test_f1_macro = f1_score(y_test, y_pred, average='macro')
            test_f1_weighted = f1_score(y_test, y_pred, average='weighted')
            
            print(f"\n{'='*60}")
            print(f"  测试集评估")
            print(f"{'='*60}")
            print(f"  accuracy:    {test_acc:.4f}")
            print(f"  f1_macro:    {test_f1_macro:.4f}")
            print(f"  f1_weighted: {test_f1_weighted:.4f}")
    else:
        # 深度学习模型（传入class_weights）
        trainer = KlineModelTrainer(model, config={
            **TRAIN_CONFIG,
            'learning_rate': args.lr,
            'max_epochs': args.epochs,
            'batch_size': args.batch_size
        }, class_weights=class_weights)
        
        save_path = MODEL_DIR / args.model / f"model_{args.code.replace('.', '_')}_{args.ktype}.pt"
        trainer.train(train_loader, val_loader, save_path=save_path)
        
        # 深度学习测试集评估
        if test_loader:
            print(f"\n{'='*60}")
            print(f"  测试集评估")
            print(f"{'='*60}")
            test_metrics = trainer.evaluate(test_loader)
            for k, v in test_metrics.items():
                print(f"  {k}: {v:.4f}")
    
    print("\n[DONE] 训练完成！")


if __name__ == "__main__":
    main()
