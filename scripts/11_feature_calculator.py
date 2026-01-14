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
from typing import Optional, List, Dict, Tuple

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
    2. OFI-L5: 5档加权OFI（指数衰减）
    3. OFI-L10: 10档加权OFI
    4. Smart-OFI: 承诺强度修正的OFI
    """
    
    def __init__(self, decay_alpha: float = DECAY_ALPHA):
        """
        初始化OFI计算器
        
        Args:
            decay_alpha: 深度衰减系数，越大衰减越快
        """
        self.decay_alpha = decay_alpha
        self.weights_5 = self._compute_weights(5)
        self.weights_10 = self._compute_weights(10)
        
    def _compute_weights(self, n_levels: int) -> np.ndarray:
        """
        计算深度衰减权重（指数衰减）
        
        w_k = exp(-α × k), 归一化使得 Σw_k = 1
        
        Args:
            n_levels: 档位数量
            
        Returns:
            归一化的权重数组
        """
        weights = np.array([np.exp(-self.decay_alpha * k) for k in range(1, n_levels + 1)])
        return weights / weights.sum()
    
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
    
    def compute_commitment_weight(self, df: pd.DataFrame) -> pd.Series:
        """
        计算"承诺强度"权重（用于Smart-OFI）
        
        基于挂单量变化与订单数变化的关系推断订单质量:
        - 量增+笔数减: 大单进入，高承诺 → weight=1.5
        - 量增+笔数增: 正常挂单，中等承诺 → weight=1.0
        - 量减+笔数减: 正常撤单，低承诺 → weight=0.5
        - 其他: 订单重组 → weight=0.8
        
        Args:
            df: 包含订单簿数据的DataFrame
            
        Returns:
            承诺强度权重序列
        """
        # 检查是否有订单数列（bid1_orders）
        has_orders = 'bid1_orders' in df.columns
        
        if not has_orders:
            # 如果没有订单数数据，使用替代方案：基于深度变化波动率
            return self._compute_volatility_weight(df)
        
        # 计算订单数变化
        df = df.copy()
        df['delta_bid1_orders'] = df['bid1_orders'].diff()
        df['delta_ask1_orders'] = df['ask1_orders'].diff() if 'ask1_orders' in df.columns else 0
        
        # 计算挂单量变化
        delta_vol = df['delta_bid1_vol'].fillna(0) - df['delta_ask1_vol'].fillna(0)
        delta_orders = df['delta_bid1_orders'].fillna(0) - df.get('delta_ask1_orders', 0)
        
        # 计算承诺强度权重
        weight = pd.Series(1.0, index=df.index)
        
        # 量增+笔数减: 大单进入（高承诺）
        mask_big_order = (delta_vol > 0) & (delta_orders < 0)
        weight[mask_big_order] = 1.5
        
        # 量增+笔数增: 正常挂单
        mask_normal = (delta_vol > 0) & (delta_orders >= 0)
        weight[mask_normal] = 1.0
        
        # 量减: 可能是撤单（低承诺）
        mask_withdraw = delta_vol < 0
        weight[mask_withdraw] = 0.5
        
        # 量不变但笔数变化: 订单重组
        mask_reorg = (delta_vol == 0) & (delta_orders != 0)
        weight[mask_reorg] = 0.8
        
        return weight
    
    def _compute_volatility_weight(self, df: pd.DataFrame, window: int = 10) -> pd.Series:
        """
        替代方案：基于深度变化波动率计算权重
        
        高波动率档位可能存在频繁撤单，给予较低权重
        
        Args:
            df: 订单簿数据
            window: 滚动窗口大小
            
        Returns:
            稳定性权重序列
        """
        # 计算买盘深度变化的滚动波动率
        volatility = df['delta_bid1_vol'].rolling(window).std().fillna(1)
        
        # 波动率越高，权重越低
        # 使用 1 / (1 + volatility/median) 进行归一化
        median_vol = volatility.median()
        if median_vol > 0:
            weight = 1.0 / (1.0 + volatility / median_vol)
        else:
            weight = pd.Series(1.0, index=df.index)
        
        # 限制在 [0.3, 1.5] 范围内
        weight = weight.clip(0.3, 1.5)
        
        return weight
    
    def compute_smart_ofi(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Smart-OFI（承诺强度修正的OFI）
        
        Smart-OFI = commitment_weight × OFI_L5
        
        Args:
            df: 订单簿数据
            
        Returns:
            Smart-OFI序列
        """
        ofi_l5 = self.compute_ofi_multi(df, n_levels=5)
        commitment = self.compute_commitment_weight(df)
        
        smart_ofi = commitment * ofi_l5
        
        return smart_ofi
    
    def compute_all_ofi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有OFI特征
        
        Args:
            df: 清洗后的订单簿数据
            
        Returns:
            添加了OFI特征列的DataFrame
        """
        # 先计算各档变化量
        df = self.compute_delta_volumes(df)
        
        # 计算各类OFI
        df['ofi_l1'] = self.compute_ofi_l1(df)
        df['ofi_l5'] = self.compute_ofi_multi(df, n_levels=5)
        df['ofi_l10'] = self.compute_ofi_multi(df, n_levels=10)
        df['smart_ofi'] = self.compute_smart_ofi(df)
        
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
        
        # 深度不平衡的滚动统计
        if 'depth_imbalance_5' in df.columns:
            df['depth_imb_ma_10'] = df['depth_imbalance_5'].rolling(self.window).mean()
        
        # 成交不平衡的滚动统计
        if 'trade_imbalance' in df.columns:
            df['trade_imb_ma_10'] = df['trade_imbalance'].rolling(self.window).mean()
        
        return df


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
    """
    
    def __init__(self):
        self.ofi_calc = OFICalculator()
        self.rolling_calc = RollingFeatureCalculator()
        self.label_gen = LabelGenerator()
        self.cov_calc = CovarianceCalculator()
        
        self.stats = {
            'input_rows': 0,
            'output_rows': 0,
            'ofi_features': 4,
            'rolling_features': 0,
            'label_distribution': {}
        }
    
    def process(
        self, 
        df: pd.DataFrame,
        compute_covariance: bool = True,
        index_code: str = 'HK.800000'
    ) -> pd.DataFrame:
        """
        执行完整的特征计算流程
        
        Args:
            df: 清洗后的订单簿数据
            compute_covariance: 是否计算动态协方差
            index_code: 指数代码
            
        Returns:
            计算完特征的DataFrame
        """
        self.stats['input_rows'] = len(df)
        
        print("\n[1/4] 计算OFI特征...")
        df = self.ofi_calc.compute_all_ofi_features(df)
        
        print("[2/4] 计算滚动统计特征...")
        df = self.rolling_calc.compute_rolling_features(df)
        
        print("[3/4] 生成预测标签...")
        df = self.label_gen.generate_labels(df)
        
        # 统计标签分布
        self.stats['label_distribution'] = self.label_gen.get_label_distribution(df)
        
        if compute_covariance:
            print("[4/4] 计算动态协方差...")
            index_df = self.cov_calc.load_index_kline(index_code)
            df = self.cov_calc.compute_rolling_covariance(df, index_df)
        else:
            print("[4/4] 跳过动态协方差计算...")
            df['cov_stock_index'] = np.nan
            df['corr_stock_index'] = np.nan
        
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
            # OFI特征
            'ofi_l1', 'ofi_l5', 'ofi_l10', 'smart_ofi',
            # 滚动统计
            'ofi_ma_10', 'ofi_std_10', 'ofi_zscore',
            'smart_ofi_ma_10', 'smart_ofi_std_10', 'smart_ofi_zscore',
            'return_ma_10', 'return_std_10',
            # 深度特征
            'bid_depth_5', 'ask_depth_5', 'depth_imbalance_5',
            'bid_depth_10', 'ask_depth_10', 'depth_imbalance_10',
            # 成交特征
            'buy_volume', 'sell_volume', 'trade_count', 'trade_imbalance',
            # 协方差
            'cov_stock_index', 'corr_stock_index',
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
    
    args = parser.parse_args()
    
    print("="*50)
    print("  OFI论文 - 特征计算模块")
    print("="*50)
    print(f"  深度衰减系数: α = {DECAY_ALPHA}")
    print(f"  滚动窗口: {ROLLING_WINDOW} 步")
    print(f"  标签阈值系数: α = {args.alpha}")
    print(f"  预测步长: k = {PREDICTION_HORIZONS}")
    
    # 初始化计算器
    calculator = FeatureCalculator()
    calculator.label_gen.alpha = args.alpha
    
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
