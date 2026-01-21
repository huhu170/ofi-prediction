"""
K线历史数据拉取脚本 - 5年版本
分批拉取股票近5年的K线数据（1分钟/5分钟/60分钟/日K/周K/月K）

策略：分批次拉取，每次拉取一段时间，避免超出API单次请求限制
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

# 要拉取的标的列表（恒生指数权重前10 + 恒生指数）
STOCK_LIST = [
    ('HK.00700', '腾讯控股'),
    ('HK.00005', '汇丰控股'),
    ('HK.09988', '阿里巴巴'),
    ('HK.01810', '小米集团'),
    ('HK.00939', '建设银行'),
    ('HK.01299', '友邦保险'),
    ('HK.00941', '中国移动'),
    ('HK.03690', '美团'),
    ('HK.01211', '比亚迪'),
    ('HK.00388', '香港交易所'),
    ('HK.800000', '恒生指数'),  # 指数
]

# K线配置：(K线类型, 类型名称, 单次拉取天数, 总共拉取年数)
# 统一拉取5年数据
KLINE_CONFIG = [
    (KLType.K_1M, 'K_1M', 30, 5),       # 1分钟K：每次30天，共拉5年
    (KLType.K_5M, 'K_5M', 60, 5),       # 5分钟K：每次60天，共拉5年
    (KLType.K_60M, 'K_60M', 180, 5),    # 60分钟K：每次180天，共拉5年
    (KLType.K_DAY, 'K_DAY', 365, 5),    # 日K：每次365天，共拉5年
    (KLType.K_WEEK, 'K_WEEK', 365, 5),  # 周K：每次365天，共拉5年
    (KLType.K_MON, 'K_MON', 365, 5),    # 月K：每次365天，共拉5年
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

def get_existing_date_range(conn, code, ktype):
    """查询数据库中已有的数据时间范围"""
    sql = """
        SELECT MIN(ts), MAX(ts), COUNT(*) 
        FROM kline 
        WHERE code = %s AND ktype = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (code, ktype))
        result = cur.fetchone()
        return result

# ============================================================
# K线拉取
# ============================================================

def fetch_kline_batch(quote_ctx, code, ktype, ktype_name, start_date, end_date):
    """拉取指定日期范围的K线数据"""
    
    print(f"    拉取 {start_date} ~ {end_date} ...")
    
    all_data = []
    
    # 分页拉取（每次最多1000条）
    ret, data, page_req_key = quote_ctx.request_history_kline(
        code, 
        start=start_date, 
        end=end_date, 
        ktype=ktype,
        max_count=1000
    )
    
    page_count = 0
    while ret == RET_OK:
        page_count += 1
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
        
        time.sleep(0.3)  # 避免请求过快
    
    if ret != RET_OK and len(all_data) == 0:
        print(f"      [FAIL] {data}")
        return []
    
    print(f"      [OK] {len(all_data)} bars (pages: {page_count})")
    return all_data


def fetch_kline_years(quote_ctx, conn, code, ktype, ktype_name, batch_days, total_years):
    """分批拉取多年的K线数据"""
    
    print(f"\n{'='*50}")
    print(f"  {code} - {ktype_name}")
    print(f"  策略: 每次{batch_days}天, 共拉取{total_years}年")
    print('='*50)
    
    # 查询已有数据
    existing = get_existing_date_range(conn, code, ktype_name)
    if existing[2] > 0:
        print(f"  已有数据: {existing[2]:,} 条 ({existing[0]} ~ {existing[1]})")
    
    total_count = 0
    end_date = datetime.now()
    start_limit = end_date - timedelta(days=total_years * 365)
    
    batch_num = 0
    current_end = end_date
    
    while current_end > start_limit:
        batch_num += 1
        current_start = current_end - timedelta(days=batch_days)
        
        # 确保不超过起始限制
        if current_start < start_limit:
            current_start = start_limit
        
        start_str = current_start.strftime('%Y-%m-%d')
        end_str = current_end.strftime('%Y-%m-%d')
        
        print(f"\n  Batch {batch_num}:")
        
        data = fetch_kline_batch(quote_ctx, code, ktype, ktype_name, start_str, end_str)
        
        if data:
            count = insert_kline(conn, data)
            total_count += count
            print(f"      Inserted: {count} rows")
        
        # 移动到下一个批次
        current_end = current_start - timedelta(days=1)
        
        # API频率限制 - 每个批次之间暂停
        time.sleep(1)
    
    print(f"\n  {ktype_name} 完成! 共插入 {total_count:,} 条")
    return total_count


# ============================================================
# 主函数
# ============================================================

def main():
    print("="*60)
    print("  恒生指数成分股 + 恒生指数 5年K线数据拉取")
    print(f"  标的数量: {len(STOCK_LIST)} 个")
    print("  K线类型: 1分钟/5分钟/60分钟/日K/周K/月K")
    print("="*60)
    print()
    
    # 显示标的列表
    print("标的列表:")
    for code, name in STOCK_LIST:
        print(f"  - {code} {name}")
    print()
    
    # 连接数据库
    conn = get_db_connection()
    print(f"[OK] Database connected: {DB_CONFIG['database']}")
    
    # 连接OpenD
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    print("[OK] OpenD connected")
    
    total_count = 0
    
    try:
        for stock_idx, (stock_code, stock_name) in enumerate(STOCK_LIST, 1):
            print("\n" + "#"*60)
            print(f"  [{stock_idx}/{len(STOCK_LIST)}] {stock_code} {stock_name}")
            print("#"*60)
            
            for ktype, ktype_name, batch_days, total_years in KLINE_CONFIG:
                count = fetch_kline_years(
                    quote_ctx, conn, stock_code, 
                    ktype, ktype_name, batch_days, total_years
                )
                total_count += count
                
                # K线类型之间暂停
                time.sleep(2)
            
            # 标的之间暂停
            time.sleep(3)
    
    except KeyboardInterrupt:
        print("\n\n[!] 用户中断")
    
    finally:
        quote_ctx.close()
        conn.close()
    
    print("\n" + "="*60)
    print(f"  全部完成! 共插入 {total_count:,} 条K线数据")
    print("="*60)
    
    # 显示最终统计
    show_data_summary()


def show_data_summary():
    """显示所有数据的汇总统计"""
    print("\n" + "="*70)
    print("  数据汇总统计")
    print("="*70)
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    # 按标的和K线类型统计
    cur.execute("""
        SELECT code, ktype, COUNT(*) as cnt, 
               MIN(ts)::date as min_date, MAX(ts)::date as max_date
        FROM kline 
        GROUP BY code, ktype
        ORDER BY code, ktype
    """)
    
    results = cur.fetchall()
    
    # 整理数据
    from collections import defaultdict
    data = defaultdict(dict)
    for code, ktype, cnt, min_date, max_date in results:
        data[code][ktype] = (cnt, min_date, max_date)
    
    # 打印表头
    ktypes = ['K_1M', 'K_5M', 'K_60M', 'K_DAY', 'K_WEEK', 'K_MON']
    print(f"\n{'标的代码':<12} | {'1分钟':>10} | {'5分钟':>10} | {'60分钟':>10} | {'日K':>8} | {'周K':>6} | {'月K':>5}")
    print("-"*80)
    
    total_bars = 0
    for code, name in STOCK_LIST:
        row_data = []
        for kt in ktypes:
            if kt in data[code]:
                cnt = data[code][kt][0]
                row_data.append(f"{cnt:,}")
                total_bars += cnt
            else:
                row_data.append("-")
        print(f"{code:<12} | {row_data[0]:>10} | {row_data[1]:>10} | {row_data[2]:>10} | {row_data[3]:>8} | {row_data[4]:>6} | {row_data[5]:>5}")
    
    print("-"*80)
    print(f"{'总计':<12} | {total_bars:,} 条K线数据")
    
    # 时间范围
    cur.execute("SELECT MIN(ts)::date, MAX(ts)::date FROM kline")
    date_range = cur.fetchone()
    print(f"\n数据时间范围: {date_range[0]} ~ {date_range[1]}")
    
    conn.close()


if __name__ == "__main__":
    print("\n" + "!"*60)
    print("  请确保 OpenD 已启动并登录!")
    print("!"*60 + "\n")
    main()
