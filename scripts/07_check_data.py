"""
数据检查脚本
查看采集到的数据统计和样本
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine

# 加载环境变量
env_path = Path(__file__).parent.parent / ".apikey.env"
load_dotenv(env_path)

# 数据库配置
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5433"),
    "database": os.getenv("DB_NAME", "futu_ofi"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "")
}

def get_engine():
    conn_str = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(conn_str)


def check_data():
    engine = get_engine()
    
    print("="*60)
    print("  数据采集统计")
    print("="*60)
    
    # 各表数据量
    tables = ['orderbook', 'ticker', 'quote', 'ofi_features']
    
    print("\n📊 各表数据量:")
    print("-"*40)
    
    for table in tables:
        try:
            df = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table}", engine)
            count = df['cnt'].values[0]
            print(f"  {table:15} : {count:>10,} 行")
        except Exception as e:
            print(f"  {table:15} : 查询失败 ({e})")
    
    # 按股票统计
    print("\n📈 订单簿数据（按股票）:")
    print("-"*40)
    
    try:
        df = pd.read_sql("""
            SELECT code, 
                   COUNT(*) as records,
                   MIN(ts) as first_time,
                   MAX(ts) as last_time
            FROM orderbook 
            GROUP BY code 
            ORDER BY records DESC
        """, engine)
        
        if len(df) > 0:
            for _, row in df.iterrows():
                print(f"  {row['code']:12} : {row['records']:>8,} 条  "
                      f"({row['first_time']} ~ {row['last_time']})")
        else:
            print("  暂无数据")
    except Exception as e:
        print(f"  查询失败: {e}")
    
    # 逐笔成交统计
    print("\n📉 逐笔成交数据（按股票）:")
    print("-"*40)
    
    try:
        df = pd.read_sql("""
            SELECT code, 
                   COUNT(*) as records,
                   SUM(volume) as total_volume
            FROM ticker 
            GROUP BY code 
            ORDER BY records DESC
        """, engine)
        
        if len(df) > 0:
            for _, row in df.iterrows():
                vol = row['total_volume'] or 0
                print(f"  {row['code']:12} : {row['records']:>8,} 笔  "
                      f"总量: {vol:>12,}")
        else:
            print("  暂无数据")
    except Exception as e:
        print(f"  查询失败: {e}")
    
    # 最新数据样本
    print("\n📋 最新订单簿样本:")
    print("-"*40)
    
    try:
        df = pd.read_sql("""
            SELECT ts, code, 
                   bid1_price, bid1_vol, 
                   ask1_price, ask1_vol
            FROM orderbook 
            ORDER BY ts DESC 
            LIMIT 5
        """, engine)
        
        if len(df) > 0:
            print(df.to_string(index=False))
        else:
            print("  暂无数据")
    except Exception as e:
        print(f"  查询失败: {e}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    check_data()
