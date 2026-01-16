"""
数据清洗模块
将原始订单簿和逐笔成交数据清洗并聚合为10秒窗口的标准化快照

参照论文设定:
- 时间窗口: Δt = 10秒 (Cont et al., 2014)
- 交易时段: 09:35-12:00, 13:00-15:55（排除开盘/收盘5分钟）
- 输入窗口: T = 100个时间步 (约16.7分钟)
- 预测步长: k ∈ {20, 50, 100}

使用方法:
    python 10_data_cleaner.py --start 2026-01-01 --end 2026-01-31
    python 10_data_cleaner.py --code HK.00700 --days 20
"""

import os
import sys
import io
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import time as dt_time

# 解决Windows编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 加载环境变量（override=True 确保覆盖系统环境变量）
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".apikey.env"
load_dotenv(env_path, override=True)

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# ============================================================
# 配置
# ============================================================

# 数据库配置（使用127.0.0.1避免IPv6问题）
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": int(os.getenv("DB_PORT", "5433")),
    "database": os.getenv("DB_NAME", "futu_ofi"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "ofi123456")
}

# 清洗参数
WINDOW_SECONDS = 10          # 聚合窗口大小（秒）
MIN_SPREAD_BPS = 0           # 最小价差（基点）
MAX_SPREAD_BPS = 100         # 最大价差（基点），超过视为异常
MAX_PRICE_JUMP_PCT = 5.0     # 最大价格跳跃（%），超过视为异常
MIN_DEPTH = 0                # 最小深度，为0视为异常

# 港股交易时段配置（排除开盘/收盘5分钟）
# 有效时段: 09:35-12:00, 13:00-15:55
HK_TRADING_SESSIONS = [
    ("09:35", "12:00"),  # 上午（排除开盘后5分钟）
    ("13:00", "15:55"),  # 下午（排除收盘前5分钟）
]

# 数据路径
DATA_PROCESSED = Path(os.getenv("DATA_PROCESSED", "data/processed"))


# ============================================================
# 数据库连接
# ============================================================

def get_db_connection():
    """获取数据库连接"""
    conn = psycopg2.connect(**DB_CONFIG)
    return conn


# ============================================================
# 数据加载
# ============================================================

class DataLoader:
    """从数据库加载原始数据"""
    
    def __init__(self):
        self.conn = get_db_connection()
        print(f"[OK] 数据库连接成功: {DB_CONFIG['database']}")
    
    def load_orderbook(
        self, 
        code: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> pd.DataFrame:
        """
        加载订单簿数据
        
        Args:
            code: 股票代码，如 'HK.00700'
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            DataFrame with columns: ts, code, bid1_price, bid1_vol, ..., ask10_price, ask10_vol
        """
        sql = """
            SELECT 
                ts, code,
                bid1_price, bid1_vol, bid2_price, bid2_vol, bid3_price, bid3_vol,
                bid4_price, bid4_vol, bid5_price, bid5_vol,
                bid6_price, bid6_vol, bid7_price, bid7_vol, bid8_price, bid8_vol,
                bid9_price, bid9_vol, bid10_price, bid10_vol,
                ask1_price, ask1_vol, ask2_price, ask2_vol, ask3_price, ask3_vol,
                ask4_price, ask4_vol, ask5_price, ask5_vol,
                ask6_price, ask6_vol, ask7_price, ask7_vol, ask8_price, ask8_vol,
                ask9_price, ask9_vol, ask10_price, ask10_vol
            FROM orderbook
            WHERE code = %s 
              AND ts >= %s 
              AND ts < %s
            ORDER BY ts
        """
        
        df = pd.read_sql(sql, self.conn, params=[code, start_time, end_time])
        df['ts'] = pd.to_datetime(df['ts'])
        
        print(f"  加载订单簿数据: {len(df)} 条 ({code})")
        return df
    
    def load_ticker(
        self, 
        code: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> pd.DataFrame:
        """
        加载逐笔成交数据
        
        Args:
            code: 股票代码
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            DataFrame with columns: ts, code, price, volume, direction
        """
        sql = """
            SELECT 
                ts, code, trade_time, sequence, 
                price, volume, turnover, direction
            FROM ticker
            WHERE code = %s 
              AND ts >= %s 
              AND ts < %s
            ORDER BY ts, sequence
        """
        
        df = pd.read_sql(sql, self.conn, params=[code, start_time, end_time])
        df['ts'] = pd.to_datetime(df['ts'])
        
        print(f"  加载逐笔数据: {len(df)} 条 ({code})")
        return df
    
    def get_available_codes(self) -> List[str]:
        """获取数据库中可用的股票代码列表"""
        sql = "SELECT DISTINCT code FROM orderbook ORDER BY code"
        df = pd.read_sql(sql, self.conn)
        return df['code'].tolist()
    
    def get_data_range(self, code: str) -> Tuple[datetime, datetime]:
        """获取指定股票的数据时间范围"""
        sql = """
            SELECT MIN(ts) as min_ts, MAX(ts) as max_ts 
            FROM orderbook 
            WHERE code = %s
        """
        df = pd.read_sql(sql, self.conn, params=[code])
        return df['min_ts'].iloc[0], df['max_ts'].iloc[0]
    
    def close(self):
        self.conn.close()


# ============================================================
# 数据清洗器
# ============================================================

class DataCleaner:
    """
    数据清洗器
    
    清洗流程:
    1. 交易时段过滤（排除开盘/收盘5分钟）
    2. 时间窗口聚合（10秒）
    3. 异常过滤
    4. 缺失值填充
    5. 基础特征计算
    """
    
    def __init__(self, window_seconds: int = WINDOW_SECONDS, 
                 trading_sessions: List[Tuple[str, str]] = None):
        self.window_seconds = window_seconds
        self.trading_sessions = trading_sessions or HK_TRADING_SESSIONS
        self.stats = {
            'raw_orderbook': 0,
            'raw_ticker': 0,
            'filtered_trading_session': 0,
            'aggregated_windows': 0,
            'filtered_zero_price': 0,
            'filtered_negative_spread': 0,
            'marked_extreme_spread': 0,  # 改为标记而非过滤
            'filtered_price_jump': 0,
            'filtered_zero_depth': 0,
            'final_windows': 0
        }
    
    def filter_trading_session(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        过滤交易时段，只保留有效时段的数据
        
        港股有效时段: 09:35-12:00, 13:00-15:55
        排除:
        - 开盘后5分钟 (09:30-09:35)
        - 午间休市 (12:00-13:00)
        - 收盘前5分钟 (15:55-16:00)
        
        Args:
            df: 原始数据，必须包含 'ts' 列
            
        Returns:
            过滤后的数据
        """
        if df.empty:
            return df
        
        original_len = len(df)
        df = df.copy()
        
        # 提取时间部分
        df['_time'] = df['ts'].dt.time
        
        # 构建有效时段掩码
        mask = pd.Series(False, index=df.index)
        for start_str, end_str in self.trading_sessions:
            start_time = datetime.strptime(start_str, "%H:%M").time()
            end_time = datetime.strptime(end_str, "%H:%M").time()
            session_mask = (df['_time'] >= start_time) & (df['_time'] < end_time)
            mask = mask | session_mask
        
        # 应用过滤
        df = df[mask]
        df = df.drop(columns=['_time'])
        
        filtered_count = original_len - len(df)
        self.stats['filtered_trading_session'] = filtered_count
        
        if filtered_count > 0:
            print(f"  交易时段过滤: {original_len} -> {len(df)} 条 "
                  f"(移除 {filtered_count} 条非交易时段数据)")
        
        return df
    
    def aggregate_orderbook(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将订单簿数据聚合到固定时间窗口
        
        策略: 每个窗口取最后一条记录（代表窗口结束时的订单簿状态）
        
        Args:
            df: 原始订单簿数据
            
        Returns:
            聚合后的订单簿快照
        """
        if df.empty:
            return pd.DataFrame()
        
        self.stats['raw_orderbook'] = len(df)
        
        # 创建时间窗口标签
        df = df.copy()
        df['window'] = df['ts'].dt.floor(f'{self.window_seconds}s')
        
        # 删除原始ts列，避免后续列名冲突
        df = df.drop(columns=['ts'])
        
        # 每个窗口取最后一条记录
        agg_df = df.groupby(['window', 'code']).last().reset_index()
        
        # 重命名window为ts
        agg_df = agg_df.rename(columns={'window': 'ts'})
        
        # 只保留需要的列
        cols_to_keep = ['ts', 'code'] + [c for c in agg_df.columns 
                                          if c not in ['ts', 'code'] 
                                          and ('bid' in c or 'ask' in c)]
        agg_df = agg_df[cols_to_keep]
        
        self.stats['aggregated_windows'] = len(agg_df)
        
        return agg_df
    
    def aggregate_ticker(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将逐笔成交数据聚合到固定时间窗口
        
        聚合规则:
        - buy_volume: 买方成交量总和
        - sell_volume: 卖方成交量总和
        - trade_count: 成交笔数
        - avg_price: 成交量加权平均价
        - last_price: 最后成交价
        - total_turnover: 总成交额
        
        Args:
            df: 原始逐笔数据
            
        Returns:
            聚合后的成交统计
        """
        if df.empty:
            return pd.DataFrame()
        
        self.stats['raw_ticker'] = len(df)
        
        df = df.copy()
        df['window'] = df['ts'].dt.floor(f'{self.window_seconds}s')
        
        # 计算买卖量
        df['buy_volume'] = df.apply(
            lambda x: x['volume'] if x['direction'] == 'BUY' else 0, axis=1
        )
        df['sell_volume'] = df.apply(
            lambda x: x['volume'] if x['direction'] == 'SELL' else 0, axis=1
        )
        
        # 按窗口聚合
        agg_df = df.groupby(['window', 'code']).agg({
            'buy_volume': 'sum',
            'sell_volume': 'sum',
            'volume': 'sum',
            'turnover': 'sum',
            'price': ['mean', 'last', 'count']
        }).reset_index()
        
        # 扁平化列名
        agg_df.columns = [
            'ts', 'code', 'buy_volume', 'sell_volume', 
            'total_volume', 'total_turnover',
            'avg_price', 'last_price', 'trade_count'
        ]
        
        return agg_df
    
    def filter_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        过滤异常数据并标记特殊状态
        
        过滤类型（直接删除）:
        1. 零价格: bid1_price或ask1_price为0或NULL
        2. 负价差: ask1_price < bid1_price
        4. 价格跳跃: 相邻窗口价格变化 > MAX_PRICE_JUMP_PCT%
        5. 零深度: bid1_vol或ask1_vol为0
        
        标记类型（保留但添加标记）:
        3. 极端价差: spread > mid_price * MAX_SPREAD_BPS / 10000
           标记字段: is_extreme_spread (True/False)
           理由: 大价差本身是订单流不平衡的重要信号，不应直接过滤
        
        Args:
            df: 聚合后的数据
            
        Returns:
            过滤后的数据（含标记字段）
        """
        if df.empty:
            return df
        
        df = df.copy()
        original_len = len(df)
        
        # 1. 过滤零价格
        mask_zero_price = (
            (df['bid1_price'].isna()) | (df['bid1_price'] <= 0) |
            (df['ask1_price'].isna()) | (df['ask1_price'] <= 0)
        )
        self.stats['filtered_zero_price'] = mask_zero_price.sum()
        df = df[~mask_zero_price]
        
        # 2. 过滤负价差
        df['spread'] = df['ask1_price'] - df['bid1_price']
        mask_negative_spread = df['spread'] < 0
        self.stats['filtered_negative_spread'] = mask_negative_spread.sum()
        df = df[~mask_negative_spread]
        
        # 3. 标记极端价差（不过滤，仅标记）
        # 理由：大价差是订单流不平衡的重要信号，可能包含有价值的预测信息
        df['mid_price'] = (df['bid1_price'] + df['ask1_price']) / 2
        df['spread_bps'] = df['spread'] / df['mid_price'] * 10000
        df['is_extreme_spread'] = df['spread_bps'] > MAX_SPREAD_BPS
        self.stats['marked_extreme_spread'] = df['is_extreme_spread'].sum()
        
        # 4. 过滤价格跳跃
        df = df.sort_values('ts')
        df['prev_mid'] = df['mid_price'].shift(1)
        df['price_change_pct'] = abs(df['mid_price'] - df['prev_mid']) / df['prev_mid'] * 100
        
        # 第一条数据无法计算变化率，不过滤
        mask_price_jump = (df['price_change_pct'] > MAX_PRICE_JUMP_PCT) & (df['prev_mid'].notna())
        self.stats['filtered_price_jump'] = mask_price_jump.sum()
        df = df[~mask_price_jump]
        
        # 5. 过滤零深度
        mask_zero_depth = (
            (df['bid1_vol'].isna()) | (df['bid1_vol'] <= MIN_DEPTH) |
            (df['ask1_vol'].isna()) | (df['ask1_vol'] <= MIN_DEPTH)
        )
        self.stats['filtered_zero_depth'] = mask_zero_depth.sum()
        df = df[~mask_zero_depth]
        
        # 清理临时列
        df = df.drop(columns=['prev_mid', 'price_change_pct'], errors='ignore')
        
        self.stats['final_windows'] = len(df)
        
        print(f"  异常过滤: {original_len} -> {len(df)} 条 "
              f"(移除 {original_len - len(df)} 条)")
        
        return df
    
    def fill_missing_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        填充缺失的时间窗口
        
        策略:
        - 对于短期缺失（≤3个窗口，即30秒），使用前向填充
        - 对于长期缺失（>3个窗口），标记为无效
        
        Args:
            df: 过滤后的数据
            
        Returns:
            填充后的数据
        """
        if df.empty:
            return df
        
        df = df.copy()
        df = df.sort_values('ts')
        
        # 获取时间范围
        start_ts = df['ts'].min()
        end_ts = df['ts'].max()
        code = df['code'].iloc[0]
        
        # 创建完整的时间网格
        full_range = pd.date_range(
            start=start_ts, 
            end=end_ts, 
            freq=f'{self.window_seconds}s'
        )
        
        full_df = pd.DataFrame({'ts': full_range})
        full_df['code'] = code
        
        # 合并
        merged = full_df.merge(df, on=['ts', 'code'], how='left')
        
        # 检测缺失段
        merged['is_missing'] = merged['mid_price'].isna()
        merged['gap_group'] = (merged['is_missing'] != merged['is_missing'].shift()).cumsum()
        
        # 计算每个缺失段的长度
        gap_lengths = merged[merged['is_missing']].groupby('gap_group').size()
        
        # 标记长期缺失（>3个窗口）
        long_gaps = gap_lengths[gap_lengths > 3].index
        merged['is_long_gap'] = merged['gap_group'].isin(long_gaps) & merged['is_missing']
        
        # 前向填充短期缺失
        cols_to_fill = [c for c in merged.columns 
                        if c not in ['ts', 'code', 'is_missing', 'gap_group', 'is_long_gap']]
        
        merged[cols_to_fill] = merged[cols_to_fill].ffill()
        
        # 移除长期缺失段
        merged = merged[~merged['is_long_gap']]
        
        # 清理临时列
        merged = merged.drop(columns=['is_missing', 'gap_group', 'is_long_gap'], errors='ignore')
        
        filled_count = len(merged) - len(df)
        if filled_count > 0:
            print(f"  缺失填充: 填充了 {filled_count} 个窗口")
        
        return merged
    
    def calculate_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算基础特征
        
        特征列表:
        - mid_price: 中间价
        - spread: 买卖价差
        - spread_bps: 价差基点
        - return_pct: 收益率
        - bid_depth_5/10: 买盘深度
        - ask_depth_5/10: 卖盘深度
        - depth_imbalance_5/10: 深度不平衡
        
        Args:
            df: 清洗后的数据
            
        Returns:
            添加基础特征后的数据
        """
        if df.empty:
            return df
        
        df = df.copy()
        df = df.sort_values('ts')
        
        # 中间价和价差（如果尚未计算）
        if 'mid_price' not in df.columns:
            df['mid_price'] = (df['bid1_price'] + df['ask1_price']) / 2
        
        if 'spread' not in df.columns:
            df['spread'] = df['ask1_price'] - df['bid1_price']
        
        if 'spread_bps' not in df.columns:
            df['spread_bps'] = df['spread'] / df['mid_price'] * 10000
        
        # 收益率
        df['return_pct'] = df['mid_price'].pct_change() * 100
        
        # 买盘深度
        bid_cols_5 = [f'bid{i}_vol' for i in range(1, 6)]
        bid_cols_10 = [f'bid{i}_vol' for i in range(1, 11)]
        
        df['bid_depth_5'] = df[bid_cols_5].sum(axis=1, skipna=True)
        df['bid_depth_10'] = df[[c for c in bid_cols_10 if c in df.columns]].sum(axis=1, skipna=True)
        
        # 卖盘深度
        ask_cols_5 = [f'ask{i}_vol' for i in range(1, 6)]
        ask_cols_10 = [f'ask{i}_vol' for i in range(1, 11)]
        
        df['ask_depth_5'] = df[ask_cols_5].sum(axis=1, skipna=True)
        df['ask_depth_10'] = df[[c for c in ask_cols_10 if c in df.columns]].sum(axis=1, skipna=True)
        
        # 深度不平衡
        df['depth_imbalance_5'] = (df['bid_depth_5'] - df['ask_depth_5']) / (
            df['bid_depth_5'] + df['ask_depth_5'] + 1e-8
        )
        df['depth_imbalance_10'] = (df['bid_depth_10'] - df['ask_depth_10']) / (
            df['bid_depth_10'] + df['ask_depth_10'] + 1e-8
        )
        
        return df
    
    def clean(
        self, 
        orderbook_df: pd.DataFrame, 
        ticker_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        执行完整的清洗流程
        
        Args:
            orderbook_df: 原始订单簿数据
            ticker_df: 原始逐笔数据
            
        Returns:
            清洗并合并后的数据
        """
        print("\n[1/6] 过滤交易时段...")
        orderbook_df = self.filter_trading_session(orderbook_df)
        ticker_df = self.filter_trading_session(ticker_df)
        
        print("[2/6] 聚合订单簿数据...")
        ob_agg = self.aggregate_orderbook(orderbook_df)
        
        print("[3/6] 聚合逐笔数据...")
        tk_agg = self.aggregate_ticker(ticker_df)
        
        print("[4/6] 过滤异常数据...")
        ob_filtered = self.filter_anomalies(ob_agg)
        
        print("[5/6] 填充缺失窗口...")
        ob_filled = self.fill_missing_windows(ob_filtered)
        
        print("[6/6] 计算基础特征...")
        result = self.calculate_base_features(ob_filled)
        
        # 合并逐笔聚合数据
        if not tk_agg.empty and not result.empty:
            result = result.merge(
                tk_agg[['ts', 'code', 'buy_volume', 'sell_volume', 
                        'trade_count', 'avg_price', 'last_price']],
                on=['ts', 'code'],
                how='left'
            )
            
            # 填充无成交的窗口
            result['buy_volume'] = result['buy_volume'].fillna(0)
            result['sell_volume'] = result['sell_volume'].fillna(0)
            result['trade_count'] = result['trade_count'].fillna(0)
            
            # 计算成交不平衡
            result['trade_imbalance'] = (result['buy_volume'] - result['sell_volume']) / (
                result['buy_volume'] + result['sell_volume'] + 1e-8
            )
        
        return result
    
    def print_stats(self):
        """打印清洗统计信息"""
        print("\n" + "="*50)
        print("  数据清洗统计")
        print("="*50)
        print(f"  原始订单簿记录:     {self.stats['raw_orderbook']:>10,}")
        print(f"  原始逐笔记录:       {self.stats['raw_ticker']:>10,}")
        print(f"  交易时段过滤:       {self.stats['filtered_trading_session']:>10,}")
        print(f"  聚合后窗口数:       {self.stats['aggregated_windows']:>10,}")
        print("-"*50)
        print("  异常过滤:")
        print(f"    - 零价格:         {self.stats['filtered_zero_price']:>10,}")
        print(f"    - 负价差:         {self.stats['filtered_negative_spread']:>10,}")
        print(f"    - 价格跳跃:       {self.stats['filtered_price_jump']:>10,}")
        print(f"    - 零深度:         {self.stats['filtered_zero_depth']:>10,}")
        print("-"*50)
        print("  标记（保留但标记）:")
        print(f"    - 极端价差:       {self.stats['marked_extreme_spread']:>10,}")
        print("-"*50)
        print(f"  最终有效窗口:       {self.stats['final_windows']:>10,}")
        print("="*50)


# ============================================================
# 数据导出
# ============================================================

class DataExporter:
    """数据导出器"""
    
    def __init__(self, output_dir: Path = DATA_PROCESSED):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.conn = get_db_connection()
    
    def to_parquet(
        self, 
        df: pd.DataFrame, 
        code: str, 
        date_str: str
    ) -> Path:
        """
        导出为Parquet文件
        
        Args:
            df: 清洗后的数据
            code: 股票代码
            date_str: 日期字符串 YYYYMMDD
            
        Returns:
            输出文件路径
        """
        # 创建股票代码目录
        code_dir = self.output_dir / code.replace('.', '_')
        code_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        filename = f"cleaned_{date_str}.parquet"
        filepath = code_dir / filename
        
        # 保存
        df.to_parquet(filepath, index=False, engine='pyarrow')
        
        print(f"  [OK] 导出: {filepath}")
        return filepath
    
    def to_database(self, df: pd.DataFrame) -> int:
        """
        保存到数据库 ofi_features 表（仅保存基础清洗结果）
        
        注意: 完整的OFI特征计算在下一步进行
        
        Args:
            df: 清洗后的数据
            
        Returns:
            插入的记录数
        """
        if df.empty:
            return 0
        
        # 选择要保存的列
        cols = [
            'ts', 'code', 'mid_price', 'spread', 'spread_bps', 'return_pct',
            'bid_depth_5', 'ask_depth_5', 'bid_depth_10', 'ask_depth_10',
            'depth_imbalance_5', 'depth_imbalance_10',
            'buy_volume', 'sell_volume', 'trade_count', 'trade_imbalance'
        ]
        
        # 只保留存在的列
        cols = [c for c in cols if c in df.columns]
        save_df = df[cols].copy()
        
        # 转换为记录列表
        records = [tuple(row) for row in save_df.values]
        
        # 构建INSERT语句
        placeholders = ', '.join(['%s'] * len(cols))
        col_names = ', '.join(cols)
        
        sql = f"""
            INSERT INTO ofi_features ({col_names})
            VALUES ({placeholders})
            ON CONFLICT (ts, code) DO UPDATE SET
                mid_price = EXCLUDED.mid_price,
                spread = EXCLUDED.spread,
                spread_bps = EXCLUDED.spread_bps
        """
        
        with self.conn.cursor() as cur:
            cur.executemany(sql, records)
        
        self.conn.commit()
        
        print(f"  [OK] 保存到数据库: {len(records)} 条")
        return len(records)
    
    def close(self):
        self.conn.close()


# ============================================================
# 主流程
# ============================================================

def clean_single_day(
    loader: DataLoader,
    cleaner: DataCleaner,
    exporter: DataExporter,
    code: str,
    date: datetime,
    export_parquet: bool = True,
    export_db: bool = False
) -> Optional[pd.DataFrame]:
    """
    清洗单日数据
    
    Args:
        loader: 数据加载器
        cleaner: 数据清洗器
        exporter: 数据导出器
        code: 股票代码
        date: 日期
        export_parquet: 是否导出Parquet
        export_db: 是否保存到数据库
        
    Returns:
        清洗后的DataFrame
    """
    start_time = datetime.combine(date.date(), datetime.min.time())
    end_time = start_time + timedelta(days=1)
    
    print(f"\n{'='*50}")
    print(f"  处理: {code} - {date.strftime('%Y-%m-%d')}")
    print('='*50)
    
    # 加载数据
    orderbook_df = loader.load_orderbook(code, start_time, end_time)
    ticker_df = loader.load_ticker(code, start_time, end_time)
    
    if orderbook_df.empty:
        print("  [SKIP] 无订单簿数据")
        return None
    
    # 清洗数据
    cleaned_df = cleaner.clean(orderbook_df, ticker_df)
    
    if cleaned_df.empty:
        print("  [SKIP] 清洗后无有效数据")
        return None
    
    # 导出
    date_str = date.strftime('%Y%m%d')
    
    if export_parquet:
        exporter.to_parquet(cleaned_df, code, date_str)
    
    if export_db:
        exporter.to_database(cleaned_df)
    
    return cleaned_df


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description='数据清洗模块')
    parser.add_argument('--code', type=str, help='股票代码，如 HK.00700')
    parser.add_argument('--start', type=str, help='开始日期 YYYY-MM-DD')
    parser.add_argument('--end', type=str, help='结束日期 YYYY-MM-DD')
    parser.add_argument('--days', type=int, default=20, help='处理最近N天的数据')
    parser.add_argument('--export-db', action='store_true', help='保存到数据库')
    parser.add_argument('--no-parquet', action='store_true', help='不导出Parquet文件')
    
    args = parser.parse_args()
    
    print("="*50)
    print("  OFI论文 - 数据清洗模块")
    print("="*50)
    print(f"  时间窗口: {WINDOW_SECONDS} 秒")
    print(f"  有效时段: {HK_TRADING_SESSIONS}")
    print(f"  输出目录: {DATA_PROCESSED}")
    
    # 初始化组件
    loader = DataLoader()
    cleaner = DataCleaner(window_seconds=WINDOW_SECONDS)
    exporter = DataExporter()
    
    try:
        # 确定股票代码
        if args.code:
            codes = [args.code]
        else:
            codes = loader.get_available_codes()
            if not codes:
                print("\n[ERROR] 数据库中无可用数据")
                return
        
        print(f"\n  处理股票: {codes}")
        
        # 确定日期范围
        if args.start and args.end:
            start_date = datetime.strptime(args.start, '%Y-%m-%d')
            end_date = datetime.strptime(args.end, '%Y-%m-%d')
        else:
            # 使用最近N天
            end_date = datetime.now()
            start_date = end_date - timedelta(days=args.days)
        
        print(f"  日期范围: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        # 处理每只股票的每一天
        all_results = []
        
        for code in codes:
            current_date = start_date
            while current_date <= end_date:
                result = clean_single_day(
                    loader=loader,
                    cleaner=cleaner,
                    exporter=exporter,
                    code=code,
                    date=current_date,
                    export_parquet=not args.no_parquet,
                    export_db=args.export_db
                )
                
                if result is not None:
                    all_results.append(result)
                
                current_date += timedelta(days=1)
        
        # 打印统计
        cleaner.print_stats()
        
        if all_results:
            total_df = pd.concat(all_results, ignore_index=True)
            print(f"\n  总计处理: {len(total_df):,} 条有效窗口")
        
    finally:
        loader.close()
        exporter.close()
    
    print("\n[DONE] 数据清洗完成！")


if __name__ == "__main__":
    main()
