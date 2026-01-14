"""
腾讯控股实时数据采集
订阅后持续接收推送，存入数据库
按 Ctrl+C 停止
"""

import os
import sys
import io
import time
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".apikey.env"
load_dotenv(env_path)

import psycopg2
from psycopg2.extras import execute_values
from futu import *

# 数据库配置
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5433,
    "database": "futu_ofi",
    "user": "postgres",
    "password": "ofi123456"
}

# 只采集腾讯
STOCK_CODE = 'HK.00700'

# 全局计数器
ticker_count = 0
orderbook_count = 0

# 数据库连接
conn = None

def init_db():
    global conn
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    print(f"[OK] Database connected")

def insert_ticker(records):
    global conn, ticker_count
    if not records:
        return
    sql = """
        INSERT INTO ticker (ts, code, name, trade_time, sequence, price, volume, turnover, direction)
        VALUES %s ON CONFLICT DO NOTHING
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, records)
    ticker_count += len(records)

def insert_orderbook(record):
    global conn, orderbook_count
    sql = """
        INSERT INTO orderbook (ts, code, bid1_price, bid1_vol, bid2_price, bid2_vol, 
            bid3_price, bid3_vol, bid4_price, bid4_vol, bid5_price, bid5_vol,
            ask1_price, ask1_vol, ask2_price, ask2_vol, ask3_price, ask3_vol,
            ask4_price, ask4_vol, ask5_price, ask5_vol)
        VALUES %s ON CONFLICT DO NOTHING
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, [record])
    orderbook_count += 1


class MyTickerHandler(TickerHandlerBase):
    """逐笔成交回调 - 每笔成交自动推送"""
    
    def on_recv_rsp(self, rsp_pb):
        ret, data = super().on_recv_rsp(rsp_pb)
        if ret == RET_OK and len(data) > 0:
            ts = datetime.now()
            records = []
            for _, row in data.iterrows():
                records.append((
                    ts, row['code'], row.get('name', ''),
                    row.get('time', ts), row.get('sequence', 0),
                    row.get('price', 0), row.get('volume', 0),
                    row.get('turnover', 0), str(row.get('ticker_direction', 'NEUTRAL'))
                ))
            insert_ticker(records)
            print(f"[TICKER] +{len(records)} | Price: {records[-1][5]} | Vol: {records[-1][6]}")
        return ret, data


class MyOrderBookHandler(OrderBookHandlerBase):
    """订单簿回调 - 盘口变化自动推送"""
    
    def on_recv_rsp(self, rsp_pb):
        ret, data = super().on_recv_rsp(rsp_pb)
        if ret == RET_OK:
            ts = datetime.now()
            code = data.get('code', '')
            bids = data.get('Bid', [])
            asks = data.get('Ask', [])
            
            record = [ts, code]
            for i in range(5):
                if i < len(bids):
                    record.extend([bids[i][0], bids[i][1]])
                else:
                    record.extend([None, None])
            for i in range(5):
                if i < len(asks):
                    record.extend([asks[i][0], asks[i][1]])
                else:
                    record.extend([None, None])
            
            insert_orderbook(tuple(record))
            
            bid1 = bids[0] if bids else ['N/A', 0]
            ask1 = asks[0] if asks else ['N/A', 0]
            print(f"[ORDERBOOK] Bid1: {bid1[0]}x{bid1[1]} | Ask1: {ask1[0]}x{ask1[1]}")
        return ret, data


def main():
    print("="*60)
    print(f"  Tencent (HK.00700) Real-time Data Collection")
    print("="*60)
    
    init_db()
    
    # 连接OpenD
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    print("[OK] OpenD connected")
    
    # 设置回调处理器
    quote_ctx.set_handler(MyTickerHandler())
    quote_ctx.set_handler(MyOrderBookHandler())
    
    # 订阅腾讯的逐笔和订单簿
    print(f"\nSubscribing to {STOCK_CODE}...")
    ret, err = quote_ctx.subscribe(
        [STOCK_CODE], 
        [SubType.TICKER, SubType.ORDER_BOOK],
        subscribe_push=True
    )
    
    if ret == RET_OK:
        print("[OK] Subscribed! Waiting for data...\n")
        print("-"*60)
    else:
        print(f"[FAIL] Subscribe failed: {err}")
        quote_ctx.close()
        return
    
    # 持续运行，等待推送
    try:
        while True:
            time.sleep(10)
            print(f"\n--- Stats: Ticker={ticker_count}, OrderBook={orderbook_count} ---\n")
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        quote_ctx.close()
        if conn:
            conn.close()
        print(f"\n[DONE] Total: Ticker={ticker_count}, OrderBook={orderbook_count}")


if __name__ == "__main__":
    main()
