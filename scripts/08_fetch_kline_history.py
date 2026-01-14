"""
K线历史数据拉取脚本
从富途API获取历史K线数据并存入数据库

支持的K线类型:
- K_1M: 1分钟K线
- K_5M: 5分钟K线  
- K_15M: 15分钟K线
- K_60M: 60分钟K线
- K_DAY: 日K线
- K_WEEK: 周K线
"""

import os
import sys
import io
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".apikey.env"
load_dotenv(env_path)

import psycopg2
from psycopg2.extras import execute_values
from futu import *

# ============================================================
# 配置
# ============================================================

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5433,
    "database": "futu_ofi",
    "user": "postgres",
    "password": "ofi123456"
}

# 要拉取的股票/指数
STOCK_LIST = [
    'HK.00700',  # 腾讯控股
    'HK.HSI',    # 恒生指数
]

# K线类型及对应天数
KLINE_CONFIG = [
    (KLType.K_1M, 'K_1M', 30),     # 1分钟K，拉取30天
    (KLType.K_5M, 'K_5M', 60),     # 5分钟K，拉取60天
    (KLType.K_60M, 'K_60M', 180),  # 60分钟K，拉取180天
    (KLType.K_DAY, 'K_DAY', 365),  # 日K，拉取1年
]

# ============================================================
# 数据库操作
# ============================================================

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def insert_kline(conn, data_list):
    """批量插入K线数据"""
    if not data_list:
        return 0
    
    sql = """
        INSERT INTO kline (
            ts, code, ktype, open_price, high_price, low_price, close_price,
            volume, turnover, turnover_rate, pe_ratio, change_rate, last_close
        ) VALUES %s
        ON CONFLICT (ts, code, ktype) DO NOTHING
    """
    
    with conn.cursor() as cur:
        execute_values(cur, sql, data_list)
        conn.commit()
    
    return len(data_list)

# ============================================================
# K线拉取
# ============================================================

def fetch_kline(quote_ctx, code, ktype, ktype_name, days):
    """拉取单只股票的K线数据"""
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    print(f"  拉取 {code} {ktype_name} ({start_date} ~ {end_date})...")
    
    all_data = []
    
    # 分页拉取（每次最多1000条）
    ret, data, page_req_key = quote_ctx.request_history_kline(
        code, 
        start=start_date, 
        end=end_date, 
        ktype=ktype,
        max_count=1000
    )
    
    while ret == RET_OK:
        for _, row in data.iterrows():
            try:
                ts = datetime.strptime(row['time_key'], '%Y-%m-%d %H:%M:%S')
            except:
                ts = datetime.strptime(row['time_key'], '%Y-%m-%d')
            
            all_data.append((
                ts,
                code,
                ktype_name,
                row.get('open'),
                row.get('high'),
                row.get('low'),
                row.get('close'),
                row.get('volume'),
                row.get('turnover'),
                row.get('turnover_rate'),
                row.get('pe_ratio'),
                row.get('change_rate'),
                row.get('last_close')
            ))
        
        if page_req_key is None:
            break
        
        # 继续拉取下一页
        ret, data, page_req_key = quote_ctx.request_history_kline(
            code, 
            start=start_date, 
            end=end_date, 
            ktype=ktype,
            max_count=1000,
            page_req_key=page_req_key
        )
        
        time.sleep(0.5)  # 避免请求过快
    
    if ret != RET_OK and len(all_data) == 0:
        print(f"    [FAIL] {data}")
        return []
    
    print(f"    [OK] {len(all_data)} bars")
    return all_data

# ============================================================
# 主函数
# ============================================================

def main():
    print("="*60)
    print("  K线历史数据拉取")
    print("="*60)
    
    # 连接数据库
    conn = get_db_connection()
    print(f"\n[OK] Database connected: {DB_CONFIG['database']}")
    
    # 连接OpenD
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    print("[OK] OpenD connected")
    
    total_count = 0
    
    try:
        for code in STOCK_LIST:
            print(f"\n{'='*40}")
            print(f"Processing: {code}")
            print('='*40)
            
            for ktype, ktype_name, days in KLINE_CONFIG:
                data = fetch_kline(quote_ctx, code, ktype, ktype_name, days)
                
                if data:
                    count = insert_kline(conn, data)
                    total_count += count
                
                time.sleep(1)  # API频率限制
    
    finally:
        quote_ctx.close()
        conn.close()
    
    print("\n" + "="*60)
    print(f"  Completed! Total: {total_count} kline bars inserted")
    print("="*60)


if __name__ == "__main__":
    print("\n[Note] Make sure OpenD is running and logged in.\n")
    main()
