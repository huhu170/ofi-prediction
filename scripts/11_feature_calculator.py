"""
特征计算模块
基于清洗后的订单簿数据计算OFI系列特征和预测标签

特征体系（与论文对齐）:
1. 基础OFI (OFI-L1): 单档订单流不平衡
2. 多档OFI (OFI-L5, OFI-L10): 深度衰减加权
3. Smart-OFI: 基于订单数变化的"承诺强度"修正
4. 滚动统计特征: MA, STD, Z-score
5. 预测标签: 波动率自适应阈值

使用方法:
    python 11_feature_calculator.py --code HK.00700 --days 20
    python 11_feature_calculator.py --input data/processed/HK_00700/cleaned_20260114.parquet
"""

import os
import sys
import io
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

# 解决Windows编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 加载环境变量
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".apikey.env"
load_dotenv(env_path, override=True)

import numpy as np
import pandas as pd
import psycopg2

# ============================================================
# 配置
# ============================================================

# 数据库配置
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": int(os.getenv("DB_PORT", "5433")),
    "database": os.getenv("DB_NAME", "futu_ofi"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "ofi123456")
}

# 特征计算参数
DECAY_ALPHA = 0.5           # 深度衰减系数（指数衰减）
ROLLING_WINDOW = 10         # 滚动统计窗口（10个时间步 = 100秒）
LABEL_ALPHA = 0.3           # 标签阈值系数（0.3倍标准差）
LABEL_WINDOW = 100          # 标签阈值计算窗口

# 预测步长
PREDICTION_HORIZONS = [20, 50, 100]  # 对应约3.3分钟, 8.3分钟, 16.7分钟

# 数据路径
DATA_PROCESSED = Path(os.getenv("DATA_PROCESSED", "data/processed"))


# ============================================================
# OFI计算器
# ============================================================

class OFICalculator:
    """
    订单流不平衡(OFI)特征计算器
    
    特征层级:
    1. OFI-L1: 单档OFI（第1档）
    2. OFI-L5: 5档加权OFI
    3. OFI-L10: 10档加权OFI
    4. Smart-OFI: 承诺强度修正的OFI
    
    支持三种深度加权方案:
    - exponential: 指数衰减 w_k = exp(-α×k)，远档快速衰减（默认）
    - linear: 线性衰减 w_k = K-k+1，平缓下降
    - equal: 等权重 w_k = 1/K，所有档位同等重要
    """
    
    # 支持的加权方案
    WEIGHT_METHODS = ['exponential', 'linear', 'equal']
    
    def __init__(self, decay_alpha: float = DECAY_ALPHA, weight_method: str = 'exponential'):
        """
        初始化OFI计算器
        
        Args:
            decay_alpha: 深度衰减系数（仅用于指数衰减），越大衰减越快
            weight_method: 加权方案 'exponential' | 'linear' | 'equal'
        """
        if weight_method not in self.WEIGHT_METHODS:
            raise ValueError(f"weight_method must be one of {self.WEIGHT_METHODS}, got '{weight_method}'")
        
        self.decay_alpha = decay_alpha
        self.weight_method = weight_method
        self.weights_5 = self._compute_weights(5)
        self.weights_10 = self._compute_weights(10)
        
    def _compute_weights(self, n_levels: int) -> np.ndarray:
        """
        计算深度加权权重
        
        三种方案:
        - exponential: w_k = exp(-α × k), 归一化使得 Σw_k = 1
        - linear: w_k = (K - k + 1) / Σi, 线性递减
        - equal: w_k = 1/K, 等权重
        
        Args:
            n_levels: 档位数量
            
        Returns:
            归一化的权重数组
        """
        if self.weight_method == 'exponential':
            # 指数衰减: 第1档权重最高，快速衰减
            weights = np.array([np.exp(-self.decay_alpha * k) for k in range(1, n_levels + 1)])
        elif self.weight_method == 'linear':
            # 线性衰减: 第1档权重最高，平缓下降
            weights = np.array([n_levels - k + 1 for k in range(1, n_levels + 1)], dtype=float)
        elif self.weight_method == 'equal':
            # 等权重: 所有档位同等重要
            weights = np.ones(n_levels)
        else:
            raise ValueError(f"Unknown weight_method: {self.weight_method}")
        
        return weights / weights.sum()
    
    def get_weights_info(self) -> str:
        """
        获取当前权重配置信息（用于打印和日志）
        
        Returns:
            格式化的权重信息字符串
        """
        info_lines = [
            f"深度加权方案: {self.weight_method}",
            f"  5档权重: [{', '.join(f'{w:.3f}' for w in self.weights_5)}]",
            f"  10档权重: [{', '.join(f'{w:.3f}' for w in self.weights_10)}]",
        ]
        if self.weight_method == 'exponential':
            info_lines.insert(1, f"  衰减系数α: {self.decay_alpha}")
        return '\n'.join(info_lines)
    
    def compute_delta_volumes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算各档位的挂单量变化
        
        Args:
            df: 清洗后的订单簿数据
            
        Returns:
            添加了Δvol列的DataFrame
        """
        df = df.copy()
        
        # 计算买盘各档变化量
        for i in range(1, 11):
            col = f'bid{i}_vol'
            if col in df.columns:
                # 将None转为NaN，然后填充为0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                df[f'delta_bid{i}_vol'] = df[col].diff()
        
        # 计算卖盘各档变化量
        for i in range(1, 11):
            col = f'ask{i}_vol'
            if col in df.columns:
                # 将None转为NaN，然后填充为0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                df[f'delta_ask{i}_vol'] = df[col].diff()
        
        return df
    
    def compute_ofi_l1(self, df: pd.DataFrame) -> pd.Series:
        """
        计算单档OFI（Level 1）
        
        OFI_L1 = Δbid1_vol - Δask1_vol
        
        正值表示买方压力增强，负值表示卖方压力增强
        
        Args:
            df: 包含delta_vol列的DataFrame
            
        Returns:
            OFI-L1序列
        """
        ofi_l1 = df['delta_bid1_vol'] - df['delta_ask1_vol']
        return ofi_l1
    
    def compute_ofi_multi(self, df: pd.DataFrame, n_levels: int = 5) -> pd.Series:
        """
        计算多档加权OFI
        
        OFI_multi = Σ(w_k × (Δbid_k - Δask_k)), k=1..K
        
        Args:
            df: 包含delta_vol列的DataFrame
            n_levels: 档位数量（5或10）
            
        Returns:
            多档加权OFI序列
        """
        weights = self.weights_5 if n_levels == 5 else self.weights_10
        
        ofi_multi = pd.Series(0.0, index=df.index)
        
        for i, w in enumerate(weights, start=1):
            delta_bid_col = f'delta_bid{i}_vol'
            delta_ask_col = f'delta_ask{i}_vol'
            
            if delta_bid_col in df.columns and delta_ask_col in df.columns:
                ofi_k = df[delta_bid_col].fillna(0) - df[delta_ask_col].fillna(0)
                ofi_multi += w * ofi_k
        
        return ofi_multi
    
    def compute_cancellation_rate(self, df: pd.DataFrame, level: int = 1) -> pd.Series:
        """
        计算第k档的撤单率（Cancellation Rate）- 混合估计方法
        
        基于论文公式(2.2-3)的定义：
        CR_k = 撤单量 / 挂单量
        
        混合估计策略（解决"挂单量减少≠撤单"的问题）：
        
        1. 如果有订单数数据（bid*_orders）：
           - 量减+笔数减：正常撤单或成交消耗，CR = |Δvol| / prev_vol × 0.5
           - 量减+笔数增/不变：主要是成交消耗（大单被拆分），CR = |Δvol| / prev_vol × 0.2
           - 量增+笔数减：大单进入，CR = 0（高质量，负撤单率效果）
           - 量增+笔数增：正常挂单，CR = 0
        
        2. 如果没有订单数数据：
           - 使用挂单量变化波动率作为代理
           - 高波动率 → 频繁撤单 → 高CR
        
        Args:
            df: 包含订单簿数据的DataFrame
            level: 档位（1-10）
            
        Returns:
            撤单率序列（0-1之间）
        """
        bid_vol_col = f'bid{level}_vol'
        ask_vol_col = f'ask{level}_vol'
        bid_orders_col = f'bid{level}_orders'
        ask_orders_col = f'ask{level}_orders'
        
        has_orders = bid_orders_col in df.columns
        
        cr_bid = pd.Series(0.0, index=df.index)
        cr_ask = pd.Series(0.0, index=df.index)
        
        # ========== 买盘撤单率 ==========
        if bid_vol_col in df.columns:
            prev_bid_vol = df[bid_vol_col].shift(1).fillna(0)
            delta_bid_vol = df[bid_vol_col] - prev_bid_vol
            
            if has_orders:
                # 方法1：结合订单数信息
                prev_bid_orders = df[bid_orders_col].shift(1).fillna(0)
                delta_bid_orders = df[bid_orders_col] - prev_bid_orders
                
                # 量减+笔数减：可能是撤单或成交
                mask_both_decrease = (delta_bid_vol < 0) & (delta_bid_orders < 0)
                cr_bid[mask_both_decrease] = np.abs(delta_bid_vol[mask_both_decrease]) / (prev_bid_vol[mask_both_decrease] + 1e-8) * 0.5
                
                # 量减+笔数不变或增加：主要是成交消耗
                mask_vol_down_orders_up = (delta_bid_vol < 0) & (delta_bid_orders >= 0)
                cr_bid[mask_vol_down_orders_up] = np.abs(delta_bid_vol[mask_vol_down_orders_up]) / (prev_bid_vol[mask_vol_down_orders_up] + 1e-8) * 0.2
                
                # 量增+笔数减：大单进入（高质量信号，负CR效果）
                # 这里设为负值，使得 (1-CR) > 1，放大高质量信号
                mask_big_order = (delta_bid_vol > 0) & (delta_bid_orders < 0)
                cr_bid[mask_big_order] = -0.3  # 承诺权重 = 1.3
            else:
                # 方法2：纯粹基于挂单量减少（fallback）
                mask_decrease = delta_bid_vol < 0
                cr_bid[mask_decrease] = np.abs(delta_bid_vol[mask_decrease]) / (prev_bid_vol[mask_decrease] + 1e-8) * 0.5
            
            cr_bid = cr_bid.clip(-0.5, 1)  # 允许负值（放大高质量信号）
        
        # ========== 卖盘撤单率 ==========
        if ask_vol_col in df.columns:
            prev_ask_vol = df[ask_vol_col].shift(1).fillna(0)
            delta_ask_vol = df[ask_vol_col] - prev_ask_vol
            
            if has_orders and ask_orders_col in df.columns:
                prev_ask_orders = df[ask_orders_col].shift(1).fillna(0)
                delta_ask_orders = df[ask_orders_col] - prev_ask_orders
                
                mask_both_decrease = (delta_ask_vol < 0) & (delta_ask_orders < 0)
                cr_ask[mask_both_decrease] = np.abs(delta_ask_vol[mask_both_decrease]) / (prev_ask_vol[mask_both_decrease] + 1e-8) * 0.5
                
                mask_vol_down_orders_up = (delta_ask_vol < 0) & (delta_ask_orders >= 0)
                cr_ask[mask_vol_down_orders_up] = np.abs(delta_ask_vol[mask_vol_down_orders_up]) / (prev_ask_vol[mask_vol_down_orders_up] + 1e-8) * 0.2
                
                mask_big_order = (delta_ask_vol > 0) & (delta_ask_orders < 0)
                cr_ask[mask_big_order] = -0.3
            else:
                mask_decrease = delta_ask_vol < 0
                cr_ask[mask_decrease] = np.abs(delta_ask_vol[mask_decrease]) / (prev_ask_vol[mask_decrease] + 1e-8) * 0.5
            
            cr_ask = cr_ask.clip(-0.5, 1)
        
        # 取买卖盘撤单率的平均
        cr = (cr_bid + cr_ask) / 2
        
        return cr
    
    def compute_cancellation_weights(self, df: pd.DataFrame, n_levels: int = 5) -> Dict[int, pd.Series]:
        """
        计算各档的承诺权重（1 - CR_k）
        
        基于论文公式(2.2-3)：
        承诺权重 = (1 - CR_k)
        撤单率越高，权重越低
        
        Args:
            df: 订单簿数据
            n_levels: 计算档位数（默认5档）
            
        Returns:
            {档位: 承诺权重序列} 的字典
        """
        commitment_weights = {}
        
        for k in range(1, n_levels + 1):
            cr_k = self.compute_cancellation_rate(df, level=k)
            commitment_weights[k] = 1.0 - cr_k
        
        return commitment_weights
    
    def compute_smart_ofi(self, df: pd.DataFrame, n_levels: int = 5) -> pd.Series:
        """
        计算Smart-OFI（撤单率修正的有效OFI）
        
        基于论文公式(2.2-3)：
        OFI_t^{effective} = Σ(k=1 to K) (1 - CR_k) × OFI_t^(k)
        
        其中：
        - CR_k = 第k档的撤单率
        - (1 - CR_k) = 承诺权重，撤单率越高权重越低
        - OFI_t^(k) = 第k档的订单流失衡
        
        Args:
            df: 订单簿数据
            n_levels: 计算档位数（默认5档）
            
        Returns:
            Smart-OFI序列
        """
        # 计算各档的承诺权重
        commitment_weights = self.compute_cancellation_weights(df, n_levels)
        
        # 计算撤单率修正的OFI
        smart_ofi = pd.Series(0.0, index=df.index)
        
        for k in range(1, n_levels + 1):
            delta_bid_col = f'delta_bid{k}_vol'
            delta_ask_col = f'delta_ask{k}_vol'
            
            if delta_bid_col in df.columns and delta_ask_col in df.columns:
                # 第k档的OFI
                ofi_k = df[delta_bid_col].fillna(0) - df[delta_ask_col].fillna(0)
                # 乘以承诺权重（1 - CR_k）
                smart_ofi += commitment_weights[k] * ofi_k
        
        return smart_ofi
    
    def compute_level_ofi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算各档独立的OFI特征（ofi_level_1 到 ofi_level_10）
        
        基于论文公式(2.2-1)中的单档OFI定义：
        OFI_t^(k) = Δbid_k_vol(t) - Δask_k_vol(t)
        
        Args:
            df: 包含delta_vol列的DataFrame
            
        Returns:
            添加了ofi_level_1..10列的DataFrame
        """
        df = df.copy()
        
        for k in range(1, 11):
            delta_bid_col = f'delta_bid{k}_vol'
            delta_ask_col = f'delta_ask{k}_vol'
            
            if delta_bid_col in df.columns and delta_ask_col in df.columns:
                df[f'ofi_level_{k}'] = df[delta_bid_col].fillna(0) - df[delta_ask_col].fillna(0)
            else:
                df[f'ofi_level_{k}'] = 0.0
        
        return df
    
    def compute_all_ofi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有OFI特征
        
        特征体系（与论文对齐）：
        1. 基础OFI (OFI-L1): 单档订单流不平衡
        2. 多档加权OFI (OFI-L5, OFI-L10): 深度衰减加权
        3. 分档OFI (ofi_level_1..10): 各档独立OFI
        4. Smart-OFI: 基于撤单率(1-CR_k)修正的有效OFI
        
        Args:
            df: 清洗后的订单簿数据
            
        Returns:
            添加了OFI特征列的DataFrame
        """
        # 先计算各档变化量
        df = self.compute_delta_volumes(df)
        
        # 1. 基础OFI（第1档）
        df['ofi_l1'] = self.compute_ofi_l1(df)
        
        # 2. 多档加权OFI（指数衰减）
        df['ofi_l5'] = self.compute_ofi_multi(df, n_levels=5)
        df['ofi_l10'] = self.compute_ofi_multi(df, n_levels=10)
        
        # 3. 分档独立OFI（ofi_level_1 到 ofi_level_10）
        df = self.compute_level_ofi(df)
        
        # 4. Smart-OFI（撤单率修正）
        # 按照论文公式(2.2-3): OFI^{effective} = Σ(1-CR_k) × OFI^(k)
        df['smart_ofi'] = self.compute_smart_ofi(df, n_levels=5)
        
        return df


# ============================================================
# 滚动统计特征计算
# ============================================================

class RollingFeatureCalculator:
    """
    滚动统计特征计算器
    
    计算OFI和收益率的滚动统计量:
    - 移动平均 (MA)
    - 移动标准差 (STD)
    - Z-score标准化
    """
    
    def __init__(self, window: int = ROLLING_WINDOW):
        self.window = window
    
    def compute_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有滚动统计特征
        
        Args:
            df: 包含OFI特征的DataFrame
            
        Returns:
            添加了滚动特征的DataFrame
        """
        df = df.copy()
        
        # OFI-L1的滚动统计
        if 'ofi_l1' in df.columns:
            df['ofi_ma_10'] = df['ofi_l1'].rolling(self.window).mean()
            df['ofi_std_10'] = df['ofi_l1'].rolling(self.window).std()
            df['ofi_zscore'] = (df['ofi_l1'] - df['ofi_ma_10']) / (df['ofi_std_10'] + 1e-8)
        
        # Smart-OFI的滚动统计
        if 'smart_ofi' in df.columns:
            df['smart_ofi_ma_10'] = df['smart_ofi'].rolling(self.window).mean()
            df['smart_ofi_std_10'] = df['smart_ofi'].rolling(self.window).std()
            df['smart_ofi_zscore'] = (df['smart_ofi'] - df['smart_ofi_ma_10']) / (df['smart_ofi_std_10'] + 1e-8)
        
        # 收益率的滚动统计
        if 'return_pct' in df.columns:
            df['return_ma_10'] = df['return_pct'].rolling(self.window).mean()
            df['return_std_10'] = df['return_pct'].rolling(self.window).std()
        
        # 深度不平衡的滚动统计（MA、STD、Z-score）
        if 'depth_imbalance_5' in df.columns:
            df['depth_imb_ma_10'] = df['depth_imbalance_5'].rolling(self.window).mean()
            df['depth_imb_std_10'] = df['depth_imbalance_5'].rolling(self.window).std()
            df['depth_imb_zscore'] = (df['depth_imbalance_5'] - df['depth_imb_ma_10']) / (df['depth_imb_std_10'] + 1e-8)

        # 成交不平衡的滚动统计（MA、STD、Z-score）
        if 'trade_imbalance' in df.columns:
            df['trade_imb_ma_10'] = df['trade_imbalance'].rolling(self.window).mean()
            df['trade_imb_std_10'] = df['trade_imbalance'].rolling(self.window).std()
            df['trade_imb_zscore'] = (df['trade_imbalance'] - df['trade_imb_ma_10']) / (df['trade_imb_std_10'] + 1e-8)

        return df


# ============================================================
# 市场状态检测器
# ============================================================

class MarketRegimeDetector:
    """
    市场状态检测器
    
    检测市场当前处于什么状态，不同状态下OFI信号含义可能不同：
    - regime=0: 平稳期（低波动、正常流动性）
    - regime=1: 波动期（高波动、流动性变化）
    - regime=2: 极端期（极端波动、流动性枯竭）
    
    支持两种检测方法：
    1. 基于规则的阈值方法（默认，无需额外依赖）
    2. HMM隐马尔可夫模型（需要hmmlearn库）
    """
    
    def __init__(
        self,
        method: str = 'threshold',  # 'threshold' 或 'hmm'
        volatility_window: int = 50,
        n_regimes: int = 3
    ):
        """
        Args:
            method: 检测方法 ('threshold'=基于阈值, 'hmm'=隐马尔可夫)
            volatility_window: 波动率计算窗口
            n_regimes: 状态数量（默认3：平稳/波动/极端）
        """
        self.method = method
        self.volatility_window = volatility_window
        self.n_regimes = n_regimes
        self.hmm_model = None
    
    def detect_regime_threshold(self, df: pd.DataFrame) -> pd.Series:
        """
        基于阈值的市场状态检测
        
        使用波动率和价差的分位数来划分状态：
        - 波动率 < 50%分位 且 价差 < 50%分位 → 平稳期(0)
        - 波动率 > 75%分位 或 价差 > 75%分位 → 波动期(1)
        - 波动率 > 95%分位 或 价差 > 95%分位 → 极端期(2)
        
        Args:
            df: 包含return_pct和spread_bps的DataFrame
            
        Returns:
            市场状态序列 (0/1/2)
        """
        regime = pd.Series(0, index=df.index)  # 默认平稳期
        
        # 计算滚动波动率
        if 'return_pct' in df.columns:
            volatility = df['return_pct'].rolling(self.volatility_window).std().fillna(0)
            vol_median = volatility.median()
            vol_75 = volatility.quantile(0.75)
            vol_95 = volatility.quantile(0.95)
        else:
            volatility = pd.Series(0, index=df.index)
            vol_median = vol_75 = vol_95 = 0
        
        # 价差作为流动性指标
        if 'spread_bps' in df.columns:
            spread = df['spread_bps']
            spread_median = spread.median()
            spread_75 = spread.quantile(0.75)
            spread_95 = spread.quantile(0.95)
        else:
            spread = pd.Series(0, index=df.index)
            spread_median = spread_75 = spread_95 = 0
        
        # 状态判断
        # 波动期：波动率或价差超过75%分位
        mask_volatile = (volatility > vol_75) | (spread > spread_75)
        regime[mask_volatile] = 1
        
        # 极端期：波动率或价差超过95%分位
        mask_extreme = (volatility > vol_95) | (spread > spread_95)
        regime[mask_extreme] = 2
        
        return regime
    
    def detect_regime_hmm(self, df: pd.DataFrame) -> pd.Series:
        """
        基于HMM的市场状态检测
        
        使用隐马尔可夫模型自动发现市场状态
        
        Args:
            df: 包含特征的DataFrame
            
        Returns:
            市场状态序列
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            print("  警告：未安装hmmlearn库，回退到阈值方法")
            print("  可通过 pip install hmmlearn 安装")
            return self.detect_regime_threshold(df)
        
        # 准备特征
        features = []
        if 'return_std_10' in df.columns:
            features.append('return_std_10')
        elif 'return_pct' in df.columns:
            df['_vol_temp'] = df['return_pct'].rolling(10).std()
            features.append('_vol_temp')
        
        if 'spread_bps' in df.columns:
            features.append('spread_bps')
        
        if 'depth_imbalance_5' in df.columns:
            features.append('depth_imbalance_5')
        
        if not features:
            print("  警告：缺少必要特征，回退到阈值方法")
            return self.detect_regime_threshold(df)
        
        # 准备数据
        X = df[features].dropna().values
        if len(X) < 100:
            print("  警告：数据量不足，回退到阈值方法")
            return self.detect_regime_threshold(df)
        
        # 训练HMM
        try:
            self.hmm_model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type='diag',
                n_iter=100,
                random_state=42
            )
            self.hmm_model.fit(X)
            
            # 预测状态
            states = self.hmm_model.predict(X)
            
            # 映射回原索引
            regime = pd.Series(0, index=df.index)
            valid_idx = df[features].dropna().index
            regime.loc[valid_idx] = states
            
            # 按波动率均值重新排序状态（确保0=平稳, 2=极端）
            state_vols = {}
            for s in range(self.n_regimes):
                mask = regime == s
                if mask.sum() > 0 and 'return_std_10' in df.columns:
                    state_vols[s] = df.loc[mask, 'return_std_10'].mean()
                else:
                    state_vols[s] = s
            
            sorted_states = sorted(state_vols.keys(), key=lambda x: state_vols[x])
            state_mapping = {old: new for new, old in enumerate(sorted_states)}
            regime = regime.map(state_mapping)
            
            return regime
            
        except Exception as e:
            print(f"  警告：HMM训练失败 ({e})，回退到阈值方法")
            return self.detect_regime_threshold(df)
    
    def detect(self, df: pd.DataFrame) -> pd.Series:
        """
        检测市场状态
        
        Args:
            df: 特征DataFrame
            
        Returns:
            市场状态序列 (0=平稳, 1=波动, 2=极端)
        """
        if self.method == 'hmm':
            return self.detect_regime_hmm(df)
        else:
            return self.detect_regime_threshold(df)


# ============================================================
# 标签生成器
# ============================================================

class LabelGenerator:
    """
    预测标签生成器
    
    使用波动率自适应阈值生成三分类标签:
    - 1: 上涨 (future_return > threshold)
    - 0: 平稳 (-threshold <= future_return <= threshold)
    - -1: 下跌 (future_return < -threshold)
    
    阈值 = alpha × rolling_std(return)
    """
    
    def __init__(
        self, 
        horizons: List[int] = PREDICTION_HORIZONS,
        alpha: float = LABEL_ALPHA,
        window: int = LABEL_WINDOW
    ):
        """
        初始化标签生成器
        
        Args:
            horizons: 预测步长列表 [20, 50, 100]
            alpha: 阈值系数（标准差的倍数）
            window: 计算滚动标准差的窗口大小
        """
        self.horizons = horizons
        self.alpha = alpha
        self.window = window
    
    def compute_future_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算未来收益率
        
        future_return_k = (mid_price[t+k] - mid_price[t]) / mid_price[t]
        
        Args:
            df: 包含mid_price的DataFrame
            
        Returns:
            添加了未来收益率列的DataFrame
        """
        df = df.copy()
        
        for k in self.horizons:
            # 未来第k步的中间价
            future_mid = df['mid_price'].shift(-k)
            
            # 未来收益率
            df[f'future_return_{k}'] = (future_mid - df['mid_price']) / df['mid_price']
        
        return df
    
    def compute_adaptive_threshold(self, df: pd.DataFrame) -> pd.Series:
        """
        计算波动率自适应阈值
        
        threshold = alpha × rolling_std(return)
        
        注意：threshold与future_return使用相同单位（小数形式）
        
        Args:
            df: 包含return_pct的DataFrame
            
        Returns:
            阈值序列（小数形式）
        """
        if 'return_pct' in df.columns:
            # return_pct是百分比形式，需要转为小数
            rolling_std = df['return_pct'].rolling(self.window).std() / 100
        elif 'return_std_10' in df.columns:
            rolling_std = df['return_std_10'] / 100
        else:
            # 使用固定阈值作为后备
            rolling_std = pd.Series(0.0001, index=df.index)
        
        threshold = self.alpha * rolling_std
        
        # 设置最小阈值，防止阈值过小（0.001% = 0.00001）
        min_threshold = 0.00001
        threshold = threshold.clip(lower=min_threshold)
        
        return threshold
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成预测标签
        
        Args:
            df: 包含mid_price和return_pct的DataFrame
            
        Returns:
            添加了标签列的DataFrame
        """
        # 计算未来收益率
        df = self.compute_future_returns(df)
        
        # 计算自适应阈值
        df['threshold'] = self.compute_adaptive_threshold(df)
        
        # 生成各步长的标签
        for k in self.horizons:
            future_return_col = f'future_return_{k}'
            label_col = f'label_{k}'
            
            df[label_col] = 0  # 默认为平稳
            
            # 上涨: future_return > threshold
            df.loc[df[future_return_col] > df['threshold'], label_col] = 1
            
            # 下跌: future_return < -threshold
            df.loc[df[future_return_col] < -df['threshold'], label_col] = -1
        
        return df
    
    def get_label_distribution(self, df: pd.DataFrame) -> Dict[str, Dict[int, float]]:
        """
        统计标签分布
        
        Args:
            df: 包含标签列的DataFrame
            
        Returns:
            各步长的标签分布比例
        """
        distribution = {}
        
        for k in self.horizons:
            label_col = f'label_{k}'
            if label_col in df.columns:
                counts = df[label_col].value_counts(normalize=True)
                distribution[label_col] = {
                    '上涨(1)': counts.get(1, 0),
                    '平稳(0)': counts.get(0, 0),
                    '下跌(-1)': counts.get(-1, 0)
                }
        
        return distribution


# ============================================================
# 动态协方差计算器
# ============================================================

class CovarianceCalculator:
    """
    动态协方差计算器
    
    计算个股与指数收益率的滚动协方差/相关系数
    用于后续模型训练的样本加权
    """
    
    def __init__(self, window: int = 100):
        """
        Args:
            window: 滚动窗口大小（时间步数）
        """
        self.window = window
        self.conn = None
    
    def load_index_kline(self, index_code: str = 'HK.800000') -> Optional[pd.DataFrame]:
        """
        从数据库加载指数K线数据
        
        Args:
            index_code: 指数代码（恒生指数 HK.800000）
            
        Returns:
            指数K线数据，包含收益率
        """
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            
            sql = """
                SELECT ts, code, close_price 
                FROM kline 
                WHERE code = %s AND ktype = '1M'
                ORDER BY ts
            """
            
            df = pd.read_sql(sql, self.conn, params=[index_code])
            
            if df.empty:
                print(f"  [WARN] 未找到指数 {index_code} 的K线数据")
                return None
            
            df['ts'] = pd.to_datetime(df['ts'])
            df['index_return'] = df['close_price'].pct_change()
            
            print(f"  加载指数K线: {len(df)} 条 ({index_code})")
            return df
            
        except Exception as e:
            print(f"  [ERROR] 加载指数数据失败: {e}")
            return None
        finally:
            if self.conn:
                self.conn.close()
    
    def compute_rolling_covariance(
        self, 
        df: pd.DataFrame, 
        index_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        计算个股与指数的滚动协方差
        
        Args:
            df: 个股特征数据（包含return_pct）
            index_df: 指数K线数据（包含index_return）
            
        Returns:
            添加了协方差列的DataFrame
        """
        df = df.copy()
        
        if index_df is None or index_df.empty:
            # 如果没有指数数据，跳过协方差计算
            df['cov_stock_index'] = np.nan
            df['corr_stock_index'] = np.nan
            return df
        
        # 将10秒数据对齐到1分钟
        df['ts_1min'] = df['ts'].dt.floor('1min')
        
        # 合并指数数据
        index_df = index_df.rename(columns={'ts': 'ts_1min'})
        df = df.merge(
            index_df[['ts_1min', 'index_return']], 
            on='ts_1min', 
            how='left'
        )
        
        # 前向填充缺失的指数收益率
        df['index_return'] = df['index_return'].ffill()
        
        # 计算滚动协方差（需要足够的数据点）
        if 'return_pct' in df.columns and 'index_return' in df.columns:
            # 使用rolling().cov()计算协方差
            stock_return = df['return_pct'] / 100  # 转为小数
            index_return = df['index_return']
            
            # 滚动协方差
            df['cov_stock_index'] = stock_return.rolling(self.window).cov(index_return)
            
            # 滚动相关系数
            df['corr_stock_index'] = stock_return.rolling(self.window).corr(index_return)
        
        # 清理临时列
        df = df.drop(columns=['ts_1min'], errors='ignore')
        
        return df


# ============================================================
# 主流程
# ============================================================

class FeatureCalculator:
    """
    特征计算主流程
    
    支持参数:
    - weight_method: 深度加权方案 ('exponential'|'linear'|'equal')
    - regime_method: 市场状态检测方法 ('threshold'|'hmm')
    """

    def __init__(self, weight_method: str = 'exponential', regime_method: str = 'threshold'):
        """
        Args:
            weight_method: 深度加权方案 'exponential' | 'linear' | 'equal'
            regime_method: 市场状态检测方法 ('threshold' 或 'hmm')
        """
        self.ofi_calc = OFICalculator(weight_method=weight_method)
        self.rolling_calc = RollingFeatureCalculator()
        self.label_gen = LabelGenerator()
        self.cov_calc = CovarianceCalculator()
        self.regime_detector = MarketRegimeDetector(method=regime_method)
        self.weight_method = weight_method

        self.stats = {
            'input_rows': 0,
            'output_rows': 0,
            'ofi_features': 4,
            'rolling_features': 0,
            'label_distribution': {},
            'regime_distribution': {}
        }
    
    def process(
        self, 
        df: pd.DataFrame,
        compute_covariance: bool = True,
        compute_regime: bool = True,
        index_code: str = 'HK.800000'
    ) -> pd.DataFrame:
        """
        执行完整的特征计算流程
        
        Args:
            df: 清洗后的订单簿数据
            compute_covariance: 是否计算动态协方差
            compute_regime: 是否计算市场状态
            index_code: 指数代码
            
        Returns:
            计算完特征的DataFrame
        """
        self.stats['input_rows'] = len(df)
        
        print("\n[1/5] 计算OFI特征...")
        df = self.ofi_calc.compute_all_ofi_features(df)
        
        print("[2/5] 计算滚动统计特征...")
        df = self.rolling_calc.compute_rolling_features(df)
        
        print("[3/5] 生成预测标签...")
        df = self.label_gen.generate_labels(df)
        
        # 统计标签分布
        self.stats['label_distribution'] = self.label_gen.get_label_distribution(df)
        
        if compute_covariance:
            print("[4/5] 计算动态协方差...")
            index_df = self.cov_calc.load_index_kline(index_code)
            df = self.cov_calc.compute_rolling_covariance(df, index_df)
        else:
            print("[4/5] 跳过动态协方差计算...")
            df['cov_stock_index'] = np.nan
            df['corr_stock_index'] = np.nan
        
        if compute_regime:
            print("[5/5] 检测市场状态...")
            df['market_regime'] = self.regime_detector.detect(df)
            # 统计状态分布
            regime_counts = df['market_regime'].value_counts(normalize=True)
            self.stats['regime_distribution'] = {
                '平稳期(0)': regime_counts.get(0, 0),
                '波动期(1)': regime_counts.get(1, 0),
                '极端期(2)': regime_counts.get(2, 0)
            }
        else:
            print("[5/5] 跳过市场状态检测...")
            df['market_regime'] = 0
        
        # 移除前面无法计算的行（因为diff/rolling需要历史数据）
        df = df.dropna(subset=['ofi_l1'])
        
        self.stats['output_rows'] = len(df)
        
        return df
    
    def print_stats(self):
        """打印特征计算统计"""
        print("\n" + "="*50)
        print("  特征计算统计")
        print("="*50)
        print(f"  输入记录数:     {self.stats['input_rows']:>10,}")
        print(f"  输出记录数:     {self.stats['output_rows']:>10,}")
        print("-"*50)
        print("  标签分布:")
        for label_name, dist in self.stats['label_distribution'].items():
            print(f"    {label_name}:")
            for cat, pct in dist.items():
                print(f"      {cat}: {pct*100:.1f}%")
        print("-"*50)
        print("  市场状态分布:")
        if self.stats['regime_distribution']:
            for regime_name, pct in self.stats['regime_distribution'].items():
                print(f"    {regime_name}: {pct*100:.1f}%")
        print("="*50)


# ============================================================
# 数据导出
# ============================================================

def export_features(
    df: pd.DataFrame, 
    output_path: Path,
    select_columns: bool = True
) -> Path:
    """
    导出特征数据为Parquet文件
    
    Args:
        df: 特征数据
        output_path: 输出路径
        select_columns: 是否只选择特征列（去除原始订单簿列）
        
    Returns:
        输出文件路径
    """
    if select_columns:
        # 选择要导出的列
        feature_cols = [
            # 索引
            'ts', 'code',
            # 价格特征
            'mid_price', 'spread', 'spread_bps', 'return_pct',
            # OFI聚合特征
            'ofi_l1', 'ofi_l5', 'ofi_l10', 'smart_ofi',
            # 分档OFI特征（ofi_level_1 到 ofi_level_10）
            'ofi_level_1', 'ofi_level_2', 'ofi_level_3', 'ofi_level_4', 'ofi_level_5',
            'ofi_level_6', 'ofi_level_7', 'ofi_level_8', 'ofi_level_9', 'ofi_level_10',
            # OFI滚动统计
            'ofi_ma_10', 'ofi_std_10', 'ofi_zscore',
            'smart_ofi_ma_10', 'smart_ofi_std_10', 'smart_ofi_zscore',
            'return_ma_10', 'return_std_10',
            # 深度特征
            'bid_depth_5', 'ask_depth_5', 'depth_imbalance_5',
            'bid_depth_10', 'ask_depth_10', 'depth_imbalance_10',
            # 深度不平衡滚动统计（新增）
            'depth_imb_ma_10', 'depth_imb_std_10', 'depth_imb_zscore',
            # 成交特征
            'buy_volume', 'sell_volume', 'trade_count', 'trade_imbalance',
            # 成交不平衡滚动统计（新增）
            'trade_imb_ma_10', 'trade_imb_std_10', 'trade_imb_zscore',
            # 协方差
            'cov_stock_index', 'corr_stock_index',
            # 市场状态（新增）
            'market_regime',
            # 标签
            'threshold',
            'future_return_20', 'future_return_50', 'future_return_100',
            'label_20', 'label_50', 'label_100'
        ]
        
        # 只保留存在的列
        cols = [c for c in feature_cols if c in df.columns]
        df = df[cols]
    
    # 创建目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存
    df.to_parquet(output_path, index=False, engine='pyarrow')
    
    print(f"  [OK] 导出: {output_path}")
    print(f"       {len(df):,} 行 × {len(df.columns)} 列")
    
    return output_path


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='特征计算模块')
    parser.add_argument('--input', type=str, help='输入Parquet文件路径')
    parser.add_argument('--code', type=str, help='股票代码')
    parser.add_argument('--days', type=int, default=20, help='处理最近N天')
    parser.add_argument('--no-covariance', action='store_true', help='不计算动态协方差')
    parser.add_argument('--index', type=str, default='HK.800000', help='指数代码')
    parser.add_argument('--alpha', type=float, default=LABEL_ALPHA, help='标签阈值系数')
    parser.add_argument('--weight-method', type=str, default='exponential',
                        choices=['exponential', 'linear', 'equal'],
                        help='深度加权方案: exponential(指数衰减) | linear(线性衰减) | equal(等权重)')

    args = parser.parse_args()

    print("="*50)
    print("  OFI论文 - 特征计算模块")
    print("="*50)
    print(f"  深度加权方案: {args.weight_method}")
    if args.weight_method == 'exponential':
        print(f"  深度衰减系数: α = {DECAY_ALPHA}")
    print(f"  滚动窗口: {ROLLING_WINDOW} 步")
    print(f"  标签阈值系数: α = {args.alpha}")
    print(f"  预测步长: k = {PREDICTION_HORIZONS}")

    # 初始化计算器
    calculator = FeatureCalculator(weight_method=args.weight_method)
    calculator.label_gen.alpha = args.alpha
    
    # 打印权重配置详情
    print(calculator.ofi_calc.get_weights_info())
    
    # 确定输入文件
    if args.input:
        input_files = [Path(args.input)]
    elif args.code:
        # 查找指定股票的所有清洗文件
        code_dir = DATA_PROCESSED / args.code.replace('.', '_')
        input_files = sorted(code_dir.glob('cleaned_*.parquet'))[-args.days:]
    else:
        # 查找所有清洗文件
        input_files = sorted(DATA_PROCESSED.glob('*/cleaned_*.parquet'))[-args.days:]
    
    if not input_files:
        print("\n[ERROR] 未找到输入文件")
        return
    
    print(f"\n  找到 {len(input_files)} 个输入文件")
    
    # 处理每个文件
    all_results = []
    
    for input_path in input_files:
        print(f"\n{'='*50}")
        print(f"  处理: {input_path.name}")
        print('='*50)
        
        # 读取数据
        df = pd.read_parquet(input_path)
        print(f"  输入: {len(df):,} 行")
        
        # 计算特征
        result = calculator.process(
            df, 
            compute_covariance=not args.no_covariance,
            index_code=args.index
        )
        
        # 导出结果
        output_name = input_path.name.replace('cleaned_', 'features_')
        output_path = input_path.parent / output_name
        export_features(result, output_path)
        
        all_results.append(result)
    
    # 打印统计
    calculator.print_stats()
    
    # 合并结果统计
    if all_results:
        total_df = pd.concat(all_results, ignore_index=True)
        print(f"\n  总计处理: {len(total_df):,} 条记录")
        
        # 打印OFI统计
        print("\n  OFI特征统计:")
        for col in ['ofi_l1', 'ofi_l5', 'smart_ofi']:
            if col in total_df.columns:
                print(f"    {col}:")
                print(f"      均值: {total_df[col].mean():.2f}")
                print(f"      标准差: {total_df[col].std():.2f}")
                print(f"      偏度: {total_df[col].skew():.2f}")
    
    print("\n[DONE] 特征计算完成！")


if __name__ == "__main__":
    main()
