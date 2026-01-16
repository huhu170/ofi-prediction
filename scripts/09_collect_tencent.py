"""
腾讯控股实时数据采集
订阅后持续接收推送，存入数据库
按 Ctrl+C 停止
"""

import os
import sys
import io
import re
import time
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

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

# 全局计数器（只统计实际写入数据库的数据）
ticker_count = 0          # 实际写入的 ticker 条数
ticker_received = 0       # 收到的 ticker 条数（含重复）
orderbook_count = 0       # 实际写入的 orderbook 条数
total_turnover = 0.0      # 实际写入的累计成交额
total_volume = 0          # 实际写入的累计成交量

# 数据库连接
conn = None

def get_session_tag(t):
    """根据时间返回交易时段标记"""
    try:
        if hasattr(t, 'hour'):
            h, m = t.hour, t.minute
        else:
            # 如果是字符串，尝试多种格式
            time_str = str(t)
            # 找 HH:MM:SS 格式（可能在字符串中间）
            match = re.search(r'(\d{1,2}):(\d{2}):\d{2}', time_str)
            if match:
                h, m = int(match.group(1)), int(match.group(2))
            else:
                return ""  # 无法解析，返回空
        
        time_val = h * 100 + m  # 转换为 HHMM 格式比较
        
        if time_val < 930:       # 9:00-9:30 盘前竞价
            return "[PRE]"
        elif time_val >= 1600:   # 16:00-16:10 收市竞价
            return "[CLS]"
        else:                    # 9:30-12:00, 13:00-16:00 连续交易
            return ""
    except:
        return ""  # 出错时返回空

def init_db():
    global conn
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    print(f"[OK] Database connected", flush=True)

def insert_ticker(records):
    """插入ticker数据，返回实际插入的行数"""
    global conn, ticker_count
    if not records:
        return 0
    sql = """
        INSERT INTO ticker (ts, code, name, trade_time, sequence, price, volume, turnover, direction, ticker_type, push_data_type)
        VALUES %s ON CONFLICT DO NOTHING
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, records)
        inserted = cur.rowcount  # 实际插入的行数
    ticker_count += inserted
    return inserted

def insert_orderbook(record):
    """插入orderbook数据，返回是否成功插入"""
    global conn, orderbook_count
    sql = """
        INSERT INTO orderbook (ts, code, name, svr_recv_time_bid, svr_recv_time_ask,
            bid1_price, bid1_vol, bid1_orders, bid2_price, bid2_vol, bid2_orders,
            bid3_price, bid3_vol, bid3_orders, bid4_price, bid4_vol, bid4_orders,
            bid5_price, bid5_vol, bid5_orders, bid6_price, bid6_vol, bid6_orders,
            bid7_price, bid7_vol, bid7_orders, bid8_price, bid8_vol, bid8_orders,
            bid9_price, bid9_vol, bid9_orders, bid10_price, bid10_vol, bid10_orders,
            ask1_price, ask1_vol, ask1_orders, ask2_price, ask2_vol, ask2_orders,
            ask3_price, ask3_vol, ask3_orders, ask4_price, ask4_vol, ask4_orders,
            ask5_price, ask5_vol, ask5_orders, ask6_price, ask6_vol, ask6_orders,
            ask7_price, ask7_vol, ask7_orders, ask8_price, ask8_vol, ask8_orders,
            ask9_price, ask9_vol, ask9_orders, ask10_price, ask10_vol, ask10_orders)
        VALUES %s ON CONFLICT DO NOTHING
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, [record])
        inserted = cur.rowcount
    if inserted > 0:
        orderbook_count += 1
    return inserted > 0


class MyTickerHandler(TickerHandlerBase):
    """逐笔成交回调 - 每笔成交自动推送"""
    
    def on_recv_rsp(self, rsp_pb):
        global total_turnover, total_volume
        ret, data = super().on_recv_rsp(rsp_pb)
        try:
            if ret == RET_OK and len(data) > 0:
                now = datetime.now()
                records = []
                batch_turnover = 0.0
                batch_volume = 0
                code = ""
                
                for _, row in data.iterrows():
                    code = row['code']
                    trade_time = row.get('time', now)
                    sequence = row.get('sequence', 0)
                    price = row.get('price', 0)
                    volume = row.get('volume', 0)
                    turnover = row.get('turnover', 0)
                    direction = str(row.get('ticker_direction', 'NEUTRAL'))
                    ticker_type = str(row.get('type', ''))
                    push_data_type = str(row.get('push_data_type', ''))
                    
                    batch_turnover += turnover
                    batch_volume += volume
                    
                    ts = trade_time if trade_time else now
                    records.append((
                        ts, code, row.get('name', ''),
                        trade_time, sequence, price, volume, turnover, direction,
                        ticker_type, push_data_type
                    ))
                
                # 插入数据库
                inserted = insert_ticker(records)
                
                # 统计
                if inserted > 0 and len(records) > 0:
                    ratio = inserted / len(records)
                    total_turnover += batch_turnover * ratio
                    total_volume += int(batch_volume * ratio)
                
                # 打印详细格式
                ts_str = now.strftime('%H:%M:%S')
                first_row = data.iloc[0]
                last_row = data.iloc[-1]
                first_time = first_row.get('time', now)
                last_time = last_row.get('time', now)
                
                def fmt_time(t):
                    """格式化时间为 HH:MM:SS.mmm"""
                    if hasattr(t, 'strftime'):
                        # 带毫秒
                        return t.strftime('%H:%M:%S.') + f"{t.microsecond // 1000:03d}"
                    # 从字符串中提取 HH:MM:SS.mmm
                    match = re.search(r'(\d{1,2}:\d{2}:\d{2})\.?(\d{0,3})?', str(t))
                    if match:
                        time_part = match.group(1)
                        ms_part = match.group(2) or "000"
                        return f"{time_part}.{ms_part[:3]:0<3}"
                    return str(t)
                
                first_time_str = fmt_time(first_time)
                last_time_str = fmt_time(last_time)
                session_tag = get_session_tag(first_time)
                
                print(f"", flush=True)
                print(f"{'='*60}", flush=True)
                print(f"[TICKER]{session_tag} ts={ts_str} | {code} | recv={len(records)}, inserted={inserted}", flush=True)
                print(f"  time: {first_time_str} ~ {last_time_str}", flush=True)
                print(f"  turnover: {batch_turnover/1_000_000:.2f}M | vol: {batch_volume:,}", flush=True)
                if len(records) <= 5:
                    for _, row in data.iterrows():
                        trade_time = row.get('time', now)
                        time_str = fmt_time(trade_time)
                        print(f"  -> #{row.get('sequence',0)} | {time_str} | {row.get('price',0):.2f} x {row.get('volume',0)} | {row.get('ticker_direction','')}", flush=True)
                print(f"{'='*60}", flush=True)
                
        except Exception as e:
            print(f"[TICKER ERROR] {e}", flush=True)
            import traceback
            traceback.print_exc()
        return ret, data


class MyOrderBookHandler(OrderBookHandlerBase):
    """订单簿回调 - 盘口变化自动推送"""
    _seq = 0  # 序列计数器，确保同一秒内时间戳唯一
    _print_counter = 0  # 打印计数器，减少输出频率
    
    def on_recv_rsp(self, rsp_pb):
        ret, data = super().on_recv_rsp(rsp_pb)
        if ret == RET_OK:
            # 使用微秒级时间戳 + 序列偏移，避免主键冲突
            MyOrderBookHandler._seq = (MyOrderBookHandler._seq + 1) % 1000
            MyOrderBookHandler._print_counter += 1
            ts = datetime.now().replace(microsecond=MyOrderBookHandler._seq * 1000)
            code = data.get('code', '')
            name = data.get('name', '')
            svr_recv_time_bid = data.get('svr_recv_time_bid', None)
            svr_recv_time_ask = data.get('svr_recv_time_ask', None)
            bids = data.get('Bid', [])
            asks = data.get('Ask', [])
            
            # 采集10档数据存入数据库（每档包含 price, vol, orders）
            record = [ts, code, name, svr_recv_time_bid, svr_recv_time_ask]
            for i in range(10):
                if i < len(bids):
                    # bids[i] 格式: [price, volume, order_num, ...]
                    price = bids[i][0] if len(bids[i]) > 0 else None
                    vol = bids[i][1] if len(bids[i]) > 1 else None
                    orders = bids[i][2] if len(bids[i]) > 2 else None
                    record.extend([price, vol, orders])
                else:
                    record.extend([None, None, None])
            for i in range(10):
                if i < len(asks):
                    price = asks[i][0] if len(asks[i]) > 0 else None
                    vol = asks[i][1] if len(asks[i]) > 1 else None
                    orders = asks[i][2] if len(asks[i]) > 2 else None
                    record.extend([price, vol, orders])
                else:
                    record.extend([None, None, None])
            
            insert_orderbook(tuple(record))
            
            # 每 20 条打印一次，减少输出（避免淹没 TICKER 日志）
            if MyOrderBookHandler._print_counter % 20 == 0:
                ts_str = ts.strftime('%H:%M:%S')
                session_tag = get_session_tag(ts)
                print(f"[ORDERBOOK]{session_tag} ts={ts_str} | {code} | #{orderbook_count}", flush=True)
                
                # 买盘：B5 -> B1（从低到高）
                bid_parts = []
                for i in range(4, -1, -1):  # 5,4,3,2,1
                    if i < len(bids):
                        bid_parts.append(f"B{i+1}: {bids[i][0]}x{bids[i][1]}")
                    else:
                        bid_parts.append(f"B{i+1}: --")
                print(f"  {' | '.join(bid_parts)}", flush=True)
                
                # 卖盘：A1 -> A5（从低到高）
                ask_parts = []
                for i in range(5):
                    if i < len(asks):
                        ask_parts.append(f"A{i+1}: {asks[i][0]}x{asks[i][1]}")
                    else:
                        ask_parts.append(f"A{i+1}: --")
                print(f"  {' | '.join(ask_parts)}", flush=True)
        return ret, data


def main():
    print("="*60)
    print(f"  Tencent (HK.00700) Real-time Data Collection")
    print("="*60)
    
    init_db()
    
    # 连接OpenD
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    print("[OK] OpenD connected", flush=True)
    
    # 设置回调处理器
    quote_ctx.set_handler(MyTickerHandler())
    quote_ctx.set_handler(MyOrderBookHandler())
    
    # 订阅腾讯的逐笔和订单簿
    print(f"\nSubscribing to {STOCK_CODE}...", flush=True)
    ret, err = quote_ctx.subscribe(
        [STOCK_CODE], 
        [SubType.TICKER, SubType.ORDER_BOOK],
        subscribe_push=True
    )
    
    if ret == RET_OK:
        print("[OK] Subscribed! Waiting for data...\n", flush=True)
        print("-"*60, flush=True)
    else:
        print(f"[FAIL] Subscribe failed: {err}", flush=True)
        quote_ctx.close()
        return
    
    # 持续运行，等待推送
    try:
        while True:
            time.sleep(10)
            turnover_m = total_turnover / 1_000_000  # 转换为百万单位
            print(f"\n--- Stats: Ticker={ticker_count}, OrderBook={orderbook_count}, Turnover={turnover_m:.2f}M, Vol={total_volume:,} ---\n", flush=True)
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        quote_ctx.close()
        if conn:
            conn.close()
        print(f"\n[DONE] Total: Ticker={ticker_count}, OrderBook={orderbook_count}")


if __name__ == "__main__":
    main()
