"""
K线特征计算模块
基于分钟级K线数据计算论文所需的多尺度量价特征

特征体系（与论文第三章对齐）:
1. 成交不平衡(TI): 基于K线价格位置推断 - 公式3.2-1
2. 多周期收益率与动量 - 公式3.2-2
3. 相对成交量与量价相关性 - 公式3.2-3, 3.2-4
4. 波动率特征(ATR, Range) - 公式3.2-5, 3.2-6
5. 技术指标(RSI, MACD, 布林带)
6. 市场状态检测 - 公式3.3-1
7. 预测标签生成 - 公式3.1-2

使用方法:
    python 11b_kline_feature_calculator.py --code HK.00700 --ktype 1M
    python 11b_kline_feature_calculator.py --code HK.00700 --multi-scale
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

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": int(os.getenv("DB_PORT", "5433")),
    "database": os.getenv("DB_NAME", "futu_ofi"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "ofi123456")
}

# K线类型配置
KLINE_TYPES = {
    '1M': {'name': '1分钟', 'window_1h': 60, 'input_len': 60},
    '5M': {'name': '5分钟', 'window_1h': 12, 'input_len': 24},
    '60M': {'name': '60分钟', 'window_1h': 1, 'input_len': 12},
    'DAY': {'name': '日K', 'window_1h': None, 'input_len': 20},
}

# 滚动窗口配置
ROLLING_WINDOWS = {
    'short': 10,   # 短期窗口
    'medium': 20,  # 中期窗口
    'long': 60,    # 长期窗口
}

# 预测步长（分钟）
PREDICTION_HORIZONS = [5, 15, 30]

# 标签阈值
LABEL_ALPHA = 0.002  # 涨跌阈值（0.2%）

# 输出路径
DATA_PROCESSED = Path(os.getenv("DATA_PROCESSED", "data/processed"))


# ============================================================
# 数据加载
# ============================================================

class KlineDataLoader:
    """K线数据加载器"""
    
    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)
        print(f"[OK] 数据库连接成功")
    
    def load_kline(
        self,
        code: str,
        ktype: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        加载K线数据
        
        Args:
            code: 股票代码，如 'HK.00700'
            ktype: K线类型，如 '1M', '5M', '60M', 'DAY'
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            K线DataFrame，包含 ts, open, high, low, close, volume
        """
        sql = """
            SELECT 
                ts, code, 
                open_price as open, 
                high_price as high,
                low_price as low, 
                close_price as close,
                volume, turnover
            FROM kline
            WHERE code = %s AND ktype = %s
        """
        params = [code, f'K_{ktype}']
        
        if start_date:
            sql += " AND ts >= %s"
            params.append(start_date)
        if end_date:
            sql += " AND ts < %s"
            params.append(end_date)
        
        sql += " ORDER BY ts"
        
        df = pd.read_sql(sql, self.conn, params=params)
        df['ts'] = pd.to_datetime(df['ts'])
        
        print(f"  加载K线数据: {len(df)} 条 ({code}, {ktype})")
        return df
    
    def get_available_codes(self) -> List[str]:
        """获取可用的股票代码"""
        sql = "SELECT DISTINCT code FROM kline ORDER BY code"
        df = pd.read_sql(sql, self.conn)
        return df['code'].tolist()
    
    def close(self):
        self.conn.close()


# ============================================================
# 成交不平衡(TI)计算 - 论文公式3.2-1
# ============================================================

class TradeImbalanceCalculator:
    """
    成交不平衡(Trade Imbalance)计算器
    
    基于论文公式3.2-1:
    TI_t = (C_t - O_t) / (H_t - L_t) × V_t
    
    边界条件: 当 H_t = L_t 时（无价格波动），TI_t = 0
    """
    
    def compute_ti(self, df: pd.DataFrame) -> pd.Series:
        """
        计算成交不平衡（单根K线）
        
        公式: TI = (Close - Open) / (High - Low) × Volume
        经济含义: 若收盘价接近最高价，买方力量占优；若接近最低价，卖方力量占优
        
        Args:
            df: K线数据，包含 open, high, low, close, volume
            
        Returns:
            TI序列
        """
        range_hl = df['high'] - df['low']
        
        # 边界条件处理：当 H = L 时（无波动），TI = 0
        # 使用 np.where 避免除零
        ti = np.where(
            range_hl > 0,
            (df['close'] - df['open']) / range_hl * df['volume'],
            0.0
        )
        
        return pd.Series(ti, index=df.index, name='ti')
    
    def compute_kline_position(self, df: pd.DataFrame) -> pd.Series:
        """
        计算K线形态特征（不乘成交量）- 论文表3.3-1
        
        公式: kline_position = (C - O) / (H - L)
        经济含义: 收盘价在日内波动区间的相对位置
        - 接近+1: 收盘价接近最高价（强势）
        - 接近-1: 收盘价接近最低价（弱势）
        - 接近0: 收盘价接近开盘价（犹豫）
        
        Args:
            df: K线数据
            
        Returns:
            K线形态特征序列
        """
        range_hl = df['high'] - df['low']
        
        # 边界条件处理：当 H = L 时（无波动），设为0
        kline_pos = np.where(
            range_hl > 0,
            (df['close'] - df['open']) / range_hl,
            0.0
        )
        
        return pd.Series(kline_pos, index=df.index, name='kline_position')
    
    def compute_multi_period_ti(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算多周期成交不平衡与K线形态特征
        
        论文表3.3-1特征:
        - kline_position: K线形态 (C-O)/(H-L)，不乘成交量
        - TI: 成交不平衡 (C-O)/(H-L) × V
        - ti_5: 5分钟累积TI (Σ TI_{t-i}, i=0..4)
        - ti_60: 60分钟累积TI (Σ TI_{t-i}, i=0..59)
        
        Args:
            df: K线数据
            
        Returns:
            添加了TI和K线形态特征的DataFrame
        """
        df = df.copy()
        
        # K线形态特征（不乘成交量）- 论文表3.3-1
        df['kline_position'] = self.compute_kline_position(df)
        
        # 基础TI（乘成交量）
        df['ti'] = self.compute_ti(df)
        
        # 累积TI（滚动求和）- 按论文表3.3-1：5分钟和60分钟
        df['ti_5'] = df['ti'].rolling(5, min_periods=1).sum()
        df['ti_60'] = df['ti'].rolling(60, min_periods=1).sum()
        
        # TI的Z-score标准化
        ti_ma = df['ti'].rolling(ROLLING_WINDOWS['medium']).mean()
        ti_std = df['ti'].rolling(ROLLING_WINDOWS['medium']).std()
        df['ti_zscore'] = (df['ti'] - ti_ma) / (ti_std + 1e-8)
        
        return df


# ============================================================
# 量价动量特征 - 论文公式3.2-2, 3.2-3, 3.2-4
# ============================================================

class MomentumFeatureCalculator:
    """量价动量特征计算器"""
    
    def compute_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算多周期收益率 - 公式3.2-2
        
        r_t^(k) = (C_t - C_{t-k}) / C_{t-k}
        
        Args:
            df: K线数据
            
        Returns:
            添加了收益率特征的DataFrame
        """
        df = df.copy()
        
        # 多周期收益率
        for k in [1, 5, 20, 60]:
            df[f'return_{k}'] = df['close'].pct_change(k)
        
        # 收益率的滚动统计
        df['return_ma_20'] = df['return_1'].rolling(20).mean()
        df['return_std_20'] = df['return_1'].rolling(20).std()
        df['return_zscore'] = (df['return_1'] - df['return_ma_20']) / (df['return_std_20'] + 1e-8)
        
        return df
    
    def compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量特征 - 公式3.2-3, 3.2-4
        
        RV_t = V_t / MA_20(V_t)  -- 相对成交量
        ρ_{PV,t} = Corr_20(r_t, V_t)  -- 量价相关性
        
        Args:
            df: K线数据
            
        Returns:
            添加了成交量特征的DataFrame
        """
        df = df.copy()
        
        # 相对成交量 - 公式3.2-3
        vol_ma = df['volume'].rolling(ROLLING_WINDOWS['medium']).mean()
        df['relative_volume'] = df['volume'] / (vol_ma + 1e-8)
        
        # 成交量变化率
        df['volume_change'] = df['volume'].pct_change()
        
        # 量价相关性 - 公式3.2-4
        df['pv_corr'] = df['return_1'].rolling(ROLLING_WINDOWS['medium']).corr(df['volume'])
        
        return df


# ============================================================
# 波动率特征 - 论文公式3.2-5, 3.2-6
# ============================================================

class VolatilityFeatureCalculator:
    """波动率特征计算器"""
    
    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算真实波幅(ATR) - 公式3.2-5
        
        TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
        ATR_t = EMA_14(TR_t)
        
        Args:
            df: K线数据
            period: ATR周期
            
        Returns:
            添加了ATR特征的DataFrame
        """
        df = df.copy()
        
        # 真实波幅 TR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        df['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # ATR (指数移动平均)
        df['atr'] = df['tr'].ewm(span=period, adjust=False).mean()
        
        # ATR相对于价格的比例（波动率百分比）
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        return df
    
    def compute_range_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算日内波幅特征 - 公式3.2-6
        
        Range_t = (H_t - L_t) / O_t
        
        Args:
            df: K线数据
            
        Returns:
            添加了波幅特征的DataFrame
        """
        df = df.copy()
        
        # 日内波幅比
        df['range_pct'] = (df['high'] - df['low']) / (df['open'] + 1e-8) * 100
        
        # 滚动波动率
        df['volatility_20'] = df['return_1'].rolling(ROLLING_WINDOWS['medium']).std() * 100
        
        return df


# ============================================================
# 技术指标
# ============================================================

class TechnicalIndicatorCalculator:
    """技术指标计算器"""
    
    def compute_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算RSI (相对强弱指数)
        
        RSI = 100 - 100 / (1 + RS)
        RS = 平均上涨幅度 / 平均下跌幅度
        """
        df = df.copy()
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def compute_macd(
        self, 
        df: pd.DataFrame, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> pd.DataFrame:
        """
        计算MACD
        
        DIF = EMA_fast - EMA_slow
        DEA = EMA_signal(DIF)
        MACD = 2 × (DIF - DEA)
        """
        df = df.copy()
        
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        df['macd_dif'] = ema_fast - ema_slow
        df['macd_dea'] = df['macd_dif'].ewm(span=signal, adjust=False).mean()
        df['macd'] = 2 * (df['macd_dif'] - df['macd_dea'])
        
        return df
    
    def compute_bollinger(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算布林带位置
        
        布林带位置 = (C - MA_20) / (2 × STD_20)
        """
        df = df.copy()
        
        ma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        
        df['bb_position'] = (df['close'] - ma) / (2 * std + 1e-8)
        df['bb_upper'] = ma + 2 * std
        df['bb_lower'] = ma - 2 * std
        
        return df


# ============================================================
# 市场状态检测 - 论文公式3.3-1
# ============================================================

class MarketRegimeDetector:
    """
    市场状态检测器 - 公式3.3-1
    
    regime_t = 
        0 (平稳期) if σ_t < Q_50(σ)
        2 (高波动期) if σ_t > Q_90(σ)
        1 (正常期) otherwise
    """
    
    def detect_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        检测市场状态
        
        Args:
            df: 包含波动率的DataFrame
            
        Returns:
            添加了市场状态的DataFrame
        """
        df = df.copy()
        
        if 'volatility_20' not in df.columns:
            df['volatility_20'] = df['return_1'].rolling(20).std() * 100
        
        vol = df['volatility_20']
        q50 = vol.quantile(0.50)
        q90 = vol.quantile(0.90)
        
        # 默认正常期
        df['market_regime'] = 1
        
        # 平稳期
        df.loc[vol < q50, 'market_regime'] = 0
        
        # 高波动期
        df.loc[vol > q90, 'market_regime'] = 2
        
        return df


# ============================================================
# 预测标签生成 - 论文公式3.1-2
# ============================================================

class LabelGenerator:
    """
    预测标签生成器 - 公式3.1-2
    
    y_t = +1 if r_{t,k} > α (上涨)
          -1 if r_{t,k} < -α (下跌)
           0 otherwise (平稳)
    """
    
    def __init__(self, alpha: float = LABEL_ALPHA):
        self.alpha = alpha
    
    def generate_labels(
        self, 
        df: pd.DataFrame, 
        horizons: List[int] = PREDICTION_HORIZONS
    ) -> pd.DataFrame:
        """
        生成预测标签
        
        Args:
            df: K线数据
            horizons: 预测步长列表（分钟）
            
        Returns:
            添加了标签的DataFrame
        """
        df = df.copy()
        
        for k in horizons:
            # 未来收益率
            future_return = df['close'].shift(-k) / df['close'] - 1
            df[f'future_return_{k}'] = future_return
            
            # 三分类标签
            label = pd.Series(0, index=df.index)
            label[future_return > self.alpha] = 1
            label[future_return < -self.alpha] = -1
            df[f'label_{k}'] = label
        
        return df


# ============================================================
# 主特征计算流程
# ============================================================

class KlineFeatureCalculator:
    """K线特征计算主流程"""
    
    def __init__(self, label_alpha: float = LABEL_ALPHA):
        self.ti_calc = TradeImbalanceCalculator()
        self.momentum_calc = MomentumFeatureCalculator()
        self.vol_calc = VolatilityFeatureCalculator()
        self.tech_calc = TechnicalIndicatorCalculator()
        self.regime_detector = MarketRegimeDetector()
        self.label_gen = LabelGenerator(alpha=label_alpha)
        
        self.stats = {
            'input_rows': 0,
            'output_rows': 0,
            'features_count': 0
        }
    
    def process(
        self, 
        df: pd.DataFrame,
        horizons: List[int] = PREDICTION_HORIZONS
    ) -> pd.DataFrame:
        """
        执行完整的特征计算流程
        
        Args:
            df: 原始K线数据
            horizons: 预测步长列表
            
        Returns:
            计算完特征的DataFrame
        """
        self.stats['input_rows'] = len(df)
        
        print("\n[1/7] 计算成交不平衡(TI)特征...")
        df = self.ti_calc.compute_multi_period_ti(df)
        
        print("[2/7] 计算多周期收益率...")
        df = self.momentum_calc.compute_returns(df)
        
        print("[3/7] 计算成交量特征...")
        df = self.momentum_calc.compute_volume_features(df)
        
        print("[4/7] 计算波动率特征...")
        df = self.vol_calc.compute_atr(df)
        df = self.vol_calc.compute_range_features(df)
        
        print("[5/7] 计算技术指标...")
        df = self.tech_calc.compute_rsi(df)
        df = self.tech_calc.compute_macd(df)
        df = self.tech_calc.compute_bollinger(df)
        
        print("[6/7] 检测市场状态...")
        df = self.regime_detector.detect_regime(df)
        
        print("[7/7] 生成预测标签...")
        df = self.label_gen.generate_labels(df, horizons)
        
        # 移除前面NaN行（因rolling需要历史数据）
        df = df.dropna(subset=['ti', 'return_1', 'atr'])
        
        self.stats['output_rows'] = len(df)
        self.stats['features_count'] = len(df.columns)
        
        return df
    
    def print_stats(self, df: pd.DataFrame):
        """打印统计信息"""
        print("\n" + "="*50)
        print("  K线特征计算统计")
        print("="*50)
        print(f"  输入记录数:     {self.stats['input_rows']:>10,}")
        print(f"  输出记录数:     {self.stats['output_rows']:>10,}")
        print(f"  特征数量:       {self.stats['features_count']:>10}")
        print("-"*50)
        print("  标签分布:")
        for col in df.columns:
            if col.startswith('label_'):
                counts = df[col].value_counts(normalize=True)
                print(f"    {col}:")
                print(f"      上涨(+1): {counts.get(1, 0)*100:.1f}%")
                print(f"      平稳(0):  {counts.get(0, 0)*100:.1f}%")
                print(f"      下跌(-1): {counts.get(-1, 0)*100:.1f}%")
        print("-"*50)
        print("  市场状态分布:")
        regime_counts = df['market_regime'].value_counts(normalize=True)
        print(f"    平稳期(0):   {regime_counts.get(0, 0)*100:.1f}%")
        print(f"    正常期(1):   {regime_counts.get(1, 0)*100:.1f}%")
        print(f"    高波动期(2): {regime_counts.get(2, 0)*100:.1f}%")
        print("="*50)


# ============================================================
# 导出
# ============================================================

def export_features(
    df: pd.DataFrame,
    output_path: Path,
    select_columns: bool = True
) -> Path:
    """导出特征数据"""
    
    if select_columns:
        # 选择要导出的特征列（与论文表3.3-1对齐，共22维模型输入特征）
        feature_cols = [
            # 索引
            'ts', 'code',
            # 价格基础（不作为模型输入，仅供参考）
            'open', 'high', 'low', 'close', 'volume',
            # K线形态 - 论文表3.3-1（2维）
            'kline_position',  # (C-O)/(H-L)，不乘成交量
            'range_pct',       # (H-L)/O，日内波幅比
            # 成交不平衡(TI) - 公式3.2-1（4维：ti, ti_5, ti_60, ti_zscore）
            'ti', 'ti_5', 'ti_60', 'ti_zscore',
            # 收益率 - 公式3.2-2（4维：return_1, return_5, return_20, return_60）
            'return_1', 'return_5', 'return_20', 'return_60',
            'return_ma_20', 'return_std_20', 'return_zscore',
            # 成交量 - 公式3.2-3, 3.2-4（2维：relative_volume, volume_change）
            'relative_volume', 'volume_change', 'pv_corr',
            # 波动率 - 公式3.2-5, 3.2-6（2维：atr_pct, volatility_20）
            'tr', 'atr', 'atr_pct', 'volatility_20',
            # 技术指标（5维）
            'rsi', 'macd_dif', 'macd_dea', 'macd', 'bb_position',
            # 市场状态 - 公式3.3-1
            'market_regime',
            # 标签
            'future_return_5', 'future_return_15', 'future_return_30',
            'label_5', 'label_15', 'label_30'
        ]
        
        cols = [c for c in feature_cols if c in df.columns]
        df = df[cols]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine='pyarrow')
    
    print(f"  [OK] 导出: {output_path}")
    print(f"       {len(df):,} 行 × {len(df.columns)} 列")
    
    return output_path


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='K线特征计算模块')
    parser.add_argument('--code', type=str, default='HK.00700', help='股票代码')
    parser.add_argument('--ktype', type=str, default='1M', 
                        choices=['1M', '5M', '60M', 'DAY'], help='K线类型')
    parser.add_argument('--start', type=str, help='开始日期 YYYY-MM-DD')
    parser.add_argument('--end', type=str, help='结束日期 YYYY-MM-DD')
    parser.add_argument('--alpha', type=float, default=LABEL_ALPHA, 
                        help='标签阈值（涨跌幅）')
    parser.add_argument('--multi-scale', action='store_true', 
                        help='计算多尺度特征（1M/5M/60M/DAY）')
    parser.add_argument('--all', action='store_true',
                        help='计算所有股票（从数据库获取列表）')
    
    args = parser.parse_args()
    
    print("="*60)
    print("  论文 - K线特征计算模块")
    print("="*60)
    
    # 初始化
    loader = KlineDataLoader()
    calculator = KlineFeatureCalculator(label_alpha=args.alpha)
    
    try:
        # 确定要处理的股票
        if args.all:
            codes = loader.get_available_codes()
            print(f"  [批量模式] 共 {len(codes)} 只股票")
        else:
            codes = [args.code]
            print(f"  股票代码: {args.code}")
        
        print(f"  标签阈值: α = {args.alpha}")
        print(f"  预测步长: k = {PREDICTION_HORIZONS}")
        
        # 确定K线类型列表
        if args.multi_scale:
            ktypes = ['1M', '5M', '60M', 'DAY']
        else:
            ktypes = [args.ktype]
        
        print(f"  K线类型: {ktypes}")
        
        # 解析日期
        start_date = datetime.strptime(args.start, '%Y-%m-%d') if args.start else None
        end_date = datetime.strptime(args.end, '%Y-%m-%d') if args.end else None
        
        # 处理每只股票的每种K线类型
        total = len(codes) * len(ktypes)
        current = 0
        
        for code in codes:
            for ktype in ktypes:
                current += 1
                print(f"\n{'='*60}")
                print(f"  [{current}/{total}] 处理: {code} - {ktype}")
                print('='*60)
                
                # 加载数据
                df = loader.load_kline(code, ktype, start_date, end_date)
                
                if df.empty:
                    print(f"  [SKIP] 无数据")
                    continue
                
                # 计算特征
                result = calculator.process(df)
                
                # 导出
                code_dir = DATA_PROCESSED / code.replace('.', '_')
                output_path = code_dir / f"kline_features_{ktype}.parquet"
                export_features(result, output_path)
                
                # 打印统计
                calculator.print_stats(result)
        
    finally:
        loader.close()
    
    print("\n[DONE] K线特征计算完成！")


if __name__ == "__main__":
    main()
