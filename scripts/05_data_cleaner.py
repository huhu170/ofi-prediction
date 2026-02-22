"""
K线数据清洗模块
从数据库读取K线数据，清洗异常并输出为标准格式

清洗规则（与论文第三章对齐）:
1. 交易时段过滤：排除开盘/收盘5分钟异常时段
2. 价格异常检测：单根K线涨跌幅超过阈值
3. 成交量异常检测：成交量为0或异常放大
4. 缺失K线处理：停牌日、数据缺失
5. 数据连续性校验：时间戳是否连续

使用方法:
    python 10b_kline_data_cleaner.py --code HK.00700 --ktype 1M
    python 10b_kline_data_cleaner.py --code HK.00700 --multi-scale
    python 10b_kline_data_cleaner.py --all  # 清洗所有股票
"""

import os
import sys
import io
import argparse
from datetime import datetime, timedelta, time as dt_time
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

# 清洗参数
CLEAN_CONFIG = {
    'max_price_change_pct': 10.0,   # 单根K线最大涨跌幅（%）
    'min_volume': 0,                 # 最小成交量（0表示允许无成交）
    'max_volume_zscore': 10.0,       # 成交量Z-score异常阈值
    'max_gap_minutes': 5,            # 分钟K线最大允许缺失（根）
}

# 港股交易时段（排除开盘/收盘5分钟）
HK_TRADING_SESSIONS = {
    '1M': [
        (dt_time(9, 35), dt_time(12, 0)),   # 上午
        (dt_time(13, 0), dt_time(15, 55)),  # 下午
    ],
    '5M': [
        (dt_time(9, 35), dt_time(12, 0)),
        (dt_time(13, 0), dt_time(15, 55)),
    ],
    '60M': [
        (dt_time(9, 30), dt_time(12, 0)),
        (dt_time(13, 0), dt_time(16, 0)),
    ],
    'DAY': None,  # 日K无需时段过滤
}

# K线类型映射
KTYPE_MAP = {
    '1M': 'K_1M',
    '5M': 'K_5M',
    '60M': 'K_60M',
    'DAY': 'K_DAY',
}

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
        从数据库加载K线数据
        
        Args:
            code: 股票代码
            ktype: K线类型（1M, 5M, 60M, DAY）
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: ts, code, open, high, low, close, volume, turnover
        """
        ktype_db = KTYPE_MAP.get(ktype, f'K_{ktype}')
        
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
        params = [code, ktype_db]
        
        if start_date:
            sql += " AND ts >= %s"
            params.append(start_date)
        if end_date:
            sql += " AND ts < %s"
            params.append(end_date)
        
        sql += " ORDER BY ts"
        
        df = pd.read_sql(sql, self.conn, params=params)
        df['ts'] = pd.to_datetime(df['ts'])
        
        print(f"  加载原始数据: {len(df):,} 条 ({code}, {ktype})")
        return df
    
    def get_available_codes(self) -> List[str]:
        """获取数据库中可用的股票代码"""
        sql = "SELECT DISTINCT code FROM kline WHERE ktype = 'K_1M' ORDER BY code"
        df = pd.read_sql(sql, self.conn)
        return df['code'].tolist()
    
    def close(self):
        self.conn.close()


# ============================================================
# 数据清洗器
# ============================================================

class KlineDataCleaner:
    """K线数据清洗器"""
    
    def __init__(self, config: dict = None):
        self.cfg = config or CLEAN_CONFIG
        self.stats = {
            'input_rows': 0,
            'output_rows': 0,
            'removed_session': 0,
            'removed_price_anomaly': 0,
            'removed_volume_anomaly': 0,
            'filled_missing': 0,
        }
    
    def clean(self, df: pd.DataFrame, ktype: str) -> pd.DataFrame:
        """
        执行完整清洗流程
        
        Args:
            df: 原始K线数据
            ktype: K线类型
            
        Returns:
            清洗后的DataFrame
        """
        self.stats['input_rows'] = len(df)
        
        if df.empty:
            return df
        
        print(f"\n[1/5] 交易时段过滤...")
        df = self._filter_trading_session(df, ktype)
        
        print(f"[2/5] 价格异常检测...")
        df = self._detect_price_anomaly(df)
        
        print(f"[3/5] 成交量异常检测...")
        df = self._detect_volume_anomaly(df)
        
        print(f"[4/5] 缺失数据处理...")
        df = self._handle_missing(df, ktype)
        
        print(f"[5/5] 数据连续性校验...")
        df = self._validate_continuity(df, ktype)
        
        self.stats['output_rows'] = len(df)
        
        return df
    
    def _filter_trading_session(self, df: pd.DataFrame, ktype: str) -> pd.DataFrame:
        """过滤非交易时段"""
        sessions = HK_TRADING_SESSIONS.get(ktype)
        
        if sessions is None:  # 日K无需过滤
            return df
        
        df = df.copy()
        df['time'] = df['ts'].dt.time
        
        mask = pd.Series(False, index=df.index)
        for start, end in sessions:
            session_mask = (df['time'] >= start) & (df['time'] <= end)
            mask |= session_mask
        
        removed = (~mask).sum()
        self.stats['removed_session'] = removed
        
        df = df[mask].drop(columns=['time'])
        print(f"    移除非交易时段: {removed:,} 条")
        
        return df
    
    def _detect_price_anomaly(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测价格异常"""
        df = df.copy()
        
        # 计算收益率
        df['return'] = df['close'].pct_change() * 100
        
        # 标记异常（涨跌幅超过阈值）
        threshold = self.cfg['max_price_change_pct']
        anomaly_mask = df['return'].abs() > threshold
        
        # 首行不算异常
        anomaly_mask.iloc[0] = False
        
        removed = anomaly_mask.sum()
        self.stats['removed_price_anomaly'] = removed
        
        if removed > 0:
            print(f"    价格异常（涨跌幅>{threshold}%）: {removed:,} 条")
            # 不直接删除，标记为NaN后续处理
            df.loc[anomaly_mask, ['open', 'high', 'low', 'close']] = np.nan
        
        df = df.drop(columns=['return'])
        return df
    
    def _detect_volume_anomaly(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测成交量异常"""
        df = df.copy()
        
        # 计算成交量Z-score
        vol_mean = df['volume'].rolling(100, min_periods=20).mean()
        vol_std = df['volume'].rolling(100, min_periods=20).std()
        df['vol_zscore'] = (df['volume'] - vol_mean) / (vol_std + 1e-8)
        
        # 标记异常
        threshold = self.cfg['max_volume_zscore']
        anomaly_mask = df['vol_zscore'].abs() > threshold
        
        removed = anomaly_mask.sum()
        self.stats['removed_volume_anomaly'] = removed
        
        if removed > 0:
            print(f"    成交量异常（Z-score>{threshold}）: {removed:,} 条")
            # 异常成交量用前值填充
            df.loc[anomaly_mask, 'volume'] = np.nan
        
        df = df.drop(columns=['vol_zscore'])
        return df
    
    def _handle_missing(self, df: pd.DataFrame, ktype: str) -> pd.DataFrame:
        """处理缺失数据"""
        df = df.copy()
        
        # 统计缺失
        missing_before = df.isna().sum().sum()
        
        # 价格缺失：前向填充
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = df[price_cols].ffill()
        
        # 成交量缺失：用0填充
        df['volume'] = df['volume'].fillna(0)
        df['turnover'] = df['turnover'].fillna(0)
        
        # 删除仍有缺失的行（主要是开头几行）
        df = df.dropna(subset=['close'])
        
        filled = missing_before - df.isna().sum().sum()
        self.stats['filled_missing'] = filled
        
        if filled > 0:
            print(f"    填充缺失值: {filled:,} 个")
        
        return df
    
    def _validate_continuity(self, df: pd.DataFrame, ktype: str) -> pd.DataFrame:
        """校验数据连续性"""
        if len(df) < 2:
            return df
        
        df = df.copy()
        
        # 计算时间差
        df['time_diff'] = df['ts'].diff()
        
        # 根据K线类型确定预期间隔
        expected_intervals = {
            '1M': timedelta(minutes=1),
            '5M': timedelta(minutes=5),
            '60M': timedelta(hours=1),
            'DAY': timedelta(days=1),
        }
        expected = expected_intervals.get(ktype, timedelta(minutes=1))
        
        # 检测异常间隔（超过预期的N倍）
        max_gap = self.cfg['max_gap_minutes']
        if ktype in ['1M', '5M']:
            gap_threshold = expected * max_gap
        else:
            gap_threshold = expected * 3  # 日K允许周末/节假日
        
        # 统计跳跃
        jumps = (df['time_diff'] > gap_threshold).sum()
        if jumps > 0:
            print(f"    时间跳跃（间隔>{gap_threshold}）: {jumps:,} 处")
        
        df = df.drop(columns=['time_diff'])
        return df
    
    def print_stats(self):
        """打印清洗统计"""
        print("\n" + "="*50)
        print("  清洗统计")
        print("="*50)
        print(f"  输入记录数:       {self.stats['input_rows']:>10,}")
        print(f"  输出记录数:       {self.stats['output_rows']:>10,}")
        print(f"  保留率:           {self.stats['output_rows']/max(self.stats['input_rows'],1)*100:>10.1f}%")
        print("-"*50)
        print(f"  移除（非交易时段）: {self.stats['removed_session']:>10,}")
        print(f"  移除（价格异常）:   {self.stats['removed_price_anomaly']:>10,}")
        print(f"  移除（成交量异常）: {self.stats['removed_volume_anomaly']:>10,}")
        print(f"  填充缺失值:       {self.stats['filled_missing']:>10,}")
        print("="*50)


# ============================================================
# 数据质量报告
# ============================================================

def generate_quality_report(df: pd.DataFrame, code: str, ktype: str) -> Dict:
    """生成数据质量报告"""
    report = {
        'code': code,
        'ktype': ktype,
        'total_rows': len(df),
        'date_range': f"{df['ts'].min()} ~ {df['ts'].max()}" if len(df) > 0 else "N/A",
        'trading_days': df['ts'].dt.date.nunique() if len(df) > 0 else 0,
    }
    
    if len(df) > 0:
        # 价格统计
        report['price_min'] = df['close'].min()
        report['price_max'] = df['close'].max()
        report['price_mean'] = df['close'].mean()
        
        # 成交量统计
        report['volume_mean'] = df['volume'].mean()
        report['volume_zero_pct'] = (df['volume'] == 0).mean() * 100
        
        # 收益率统计
        returns = df['close'].pct_change().dropna()
        report['return_mean'] = returns.mean() * 100
        report['return_std'] = returns.std() * 100
        report['return_max'] = returns.max() * 100
        report['return_min'] = returns.min() * 100
    
    return report


def print_quality_report(report: Dict):
    """打印质量报告"""
    print("\n" + "="*50)
    print(f"  数据质量报告 ({report['code']}, {report['ktype']})")
    print("="*50)
    print(f"  记录数:     {report['total_rows']:,}")
    print(f"  日期范围:   {report['date_range']}")
    print(f"  交易日数:   {report['trading_days']}")
    
    if report['total_rows'] > 0:
        print("-"*50)
        print(f"  价格区间:   {report['price_min']:.2f} ~ {report['price_max']:.2f}")
        print(f"  平均价格:   {report['price_mean']:.2f}")
        print(f"  平均成交量: {report['volume_mean']:,.0f}")
        print(f"  零成交占比: {report['volume_zero_pct']:.2f}%")
        print("-"*50)
        print(f"  收益率均值: {report['return_mean']:.4f}%")
        print(f"  收益率标准差: {report['return_std']:.4f}%")
        print(f"  收益率极值: [{report['return_min']:.2f}%, {report['return_max']:.2f}%]")
    print("="*50)


# ============================================================
# 导出
# ============================================================

def export_cleaned_data(
    df: pd.DataFrame,
    code: str,
    ktype: str,
    output_dir: Path = DATA_PROCESSED
) -> Path:
    """导出清洗后的数据"""
    code_dir = output_dir / code.replace('.', '_')
    code_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = code_dir / f"kline_cleaned_{ktype}.parquet"
    df.to_parquet(output_path, index=False, engine='pyarrow')
    
    print(f"  [OK] 导出: {output_path}")
    print(f"       {len(df):,} 行 × {len(df.columns)} 列")
    
    return output_path


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='K线数据清洗')
    parser.add_argument('--code', type=str, default='HK.00700', help='股票代码')
    parser.add_argument('--ktype', type=str, default='1M', 
                        choices=['1M', '5M', '60M', 'DAY'], help='K线类型')
    parser.add_argument('--start', type=str, help='开始日期 YYYY-MM-DD')
    parser.add_argument('--end', type=str, help='结束日期 YYYY-MM-DD')
    parser.add_argument('--multi-scale', action='store_true', 
                        help='清洗所有时间尺度（1M/5M/60M/DAY）')
    parser.add_argument('--all', action='store_true', 
                        help='清洗所有股票')
    
    args = parser.parse_args()
    
    print("="*60)
    print("  K线数据清洗模块")
    print("="*60)
    
    # 初始化
    loader = KlineDataLoader()
    cleaner = KlineDataCleaner()
    
    try:
        # 确定要处理的股票
        if args.all:
            codes = loader.get_available_codes()
        else:
            codes = [args.code]
        
        # 确定要处理的K线类型
        if args.multi_scale:
            ktypes = ['1M', '5M', '60M', 'DAY']
        else:
            ktypes = [args.ktype]
        
        # 解析日期
        start_date = datetime.strptime(args.start, '%Y-%m-%d') if args.start else None
        end_date = datetime.strptime(args.end, '%Y-%m-%d') if args.end else None
        
        print(f"  股票数量: {len(codes)}")
        print(f"  K线类型: {ktypes}")
        
        # 处理每只股票的每种K线
        for code in codes:
            for ktype in ktypes:
                print(f"\n{'='*60}")
                print(f"  处理: {code} - {ktype}")
                print('='*60)
                
                # 加载数据
                df = loader.load_kline(code, ktype, start_date, end_date)
                
                if df.empty:
                    print(f"  [SKIP] 无数据")
                    continue
                
                # 清洗
                df_clean = cleaner.clean(df, ktype)
                
                # 统计
                cleaner.print_stats()
                
                # 质量报告
                report = generate_quality_report(df_clean, code, ktype)
                print_quality_report(report)
                
                # 导出
                export_cleaned_data(df_clean, code, ktype)
                
                # 重置统计
                cleaner.stats = {k: 0 for k in cleaner.stats}
        
    finally:
        loader.close()
    
    print("\n[DONE] 数据清洗完成！")


if __name__ == "__main__":
    main()
