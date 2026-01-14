"""
富途API数据采集脚本
实时采集订单簿、逐笔成交、报价数据并存入数据库

使用方法:
    python 06_data_collector.py

按 Ctrl+C 停止采集
"""

import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from decimal import Decimal

# 加载环境变量
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".apikey.env"
load_dotenv(env_path)

import psycopg2
from psycopg2.extras import execute_values
from futu import *

# ============================================================
# 配置
# ============================================================

# 数据库配置
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5433")),
    "database": os.getenv("DB_NAME", "futu_ofi"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "")
}

# 采集配置
STOCK_LIST = [
    'HK.00700',  # 腾讯
    'HK.09988',  # 阿里巴巴
    'HK.00005',  # 汇丰
    'HK.01810',  # 小米
    'HK.09999',  # 网易
]

# 采集间隔（秒）
ORDERBOOK_INTERVAL = 1.0   # 订单簿快照间隔
QUOTE_INTERVAL = 5.0       # 报价间隔

# ============================================================
# 数据库操作
# ============================================================

class DatabaseWriter:
    """数据库写入器"""
    
    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.conn.autocommit = True
        print(f"✅ 数据库连接成功: {DB_CONFIG['database']}")
    
    def insert_orderbook(self, data_list):
        """批量插入订单簿数据"""
        if not data_list:
            return
        
        sql = """
            INSERT INTO orderbook (
                ts, code, name,
                bid1_price, bid1_vol, bid2_price, bid2_vol, bid3_price, bid3_vol,
                bid4_price, bid4_vol, bid5_price, bid5_vol,
                ask1_price, ask1_vol, ask2_price, ask2_vol, ask3_price, ask3_vol,
                ask4_price, ask4_vol, ask5_price, ask5_vol
            ) VALUES %s
            ON CONFLICT (ts, code) DO NOTHING
        """
        
        with self.conn.cursor() as cur:
            execute_values(cur, sql, data_list)
        
    def insert_ticker(self, data_list):
        """批量插入逐笔数据"""
        if not data_list:
            return
        
        sql = """
            INSERT INTO ticker (
                ts, code, name, trade_time, sequence, price, volume, turnover, direction
            ) VALUES %s
            ON CONFLICT (ts, code, sequence) DO NOTHING
        """
        
        with self.conn.cursor() as cur:
            execute_values(cur, sql, data_list)
    
    def insert_quote(self, data_list):
        """批量插入报价数据"""
        if not data_list:
            return
        
        sql = """
            INSERT INTO quote (
                ts, code, name, last_price, open_price, high_price, low_price,
                prev_close_price, volume, turnover
            ) VALUES %s
            ON CONFLICT (ts, code) DO NOTHING
        """
        
        with self.conn.cursor() as cur:
            execute_values(cur, sql, data_list)
    
    def close(self):
        self.conn.close()


# ============================================================
# 行情回调处理器
# ============================================================

class TickerHandler(TickerHandlerBase):
    """逐笔成交回调"""
    
    def __init__(self, db_writer):
        super().__init__()
        self.db_writer = db_writer
        self.count = 0
    
    def on_recv_rsp(self, rsp_pb):
        ret, data = super().on_recv_rsp(rsp_pb)
        if ret == RET_OK and len(data) > 0:
            ts = datetime.now()
            records = []
            for _, row in data.iterrows():
                records.append((
                    ts,
                    row['code'],
                    row.get('name', ''),
                    row.get('time', ts),
                    row.get('sequence', 0),
                    row.get('price', 0),
                    row.get('volume', 0),
                    row.get('turnover', 0),
                    row.get('ticker_direction', 'NEUTRAL')
                ))
            
            self.db_writer.insert_ticker(records)
            self.count += len(records)
            
        return ret, data


class OrderBookHandler(OrderBookHandlerBase):
    """订单簿回调"""
    
    def __init__(self, db_writer):
        super().__init__()
        self.db_writer = db_writer
        self.count = 0
    
    def on_recv_rsp(self, rsp_pb):
        ret, data = super().on_recv_rsp(rsp_pb)
        if ret == RET_OK:
            ts = datetime.now()
            code = data.get('code', '')
            
            bids = data.get('Bid', [])
            asks = data.get('Ask', [])
            
            # 提取买卖盘各5档
            record = [ts, code, '']
            
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
            
            self.db_writer.insert_orderbook([tuple(record)])
            self.count += 1
            
        return ret, data


# ============================================================
# 主采集逻辑
# ============================================================

class DataCollector:
    """数据采集器"""
    
    def __init__(self, stock_list):
        self.stock_list = stock_list
        self.running = False
        self.db_writer = DatabaseWriter()
        
        # 创建行情上下文
        self.quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
        
        # 设置回调
        self.ticker_handler = TickerHandler(self.db_writer)
        self.orderbook_handler = OrderBookHandler(self.db_writer)
        
        self.quote_ctx.set_handler(self.ticker_handler)
        self.quote_ctx.set_handler(self.orderbook_handler)
    
    def subscribe(self):
        """订阅行情"""
        print(f"\n📡 订阅股票行情: {self.stock_list}")
        
        # 订阅逐笔成交
        ret, err = self.quote_ctx.subscribe(
            self.stock_list, 
            [SubType.TICKER, SubType.ORDER_BOOK, SubType.QUOTE],
            subscribe_push=True
        )
        
        if ret == RET_OK:
            print("✅ 订阅成功！")
            return True
        else:
            print(f"❌ 订阅失败: {err}")
            return False
    
    def poll_orderbook(self):
        """轮询订单簿（补充推送）"""
        while self.running:
            for code in self.stock_list:
                try:
                    ret, data = self.quote_ctx.get_order_book(code, num=10)
                    if ret == RET_OK:
                        ts = datetime.now()
                        bids = data.get('Bid', [])
                        asks = data.get('Ask', [])
                        
                        record = [ts, code, '']
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
                        
                        self.db_writer.insert_orderbook([tuple(record)])
                        
                except Exception as e:
                    print(f"⚠️ 获取订单簿失败 {code}: {e}")
            
            time.sleep(ORDERBOOK_INTERVAL)
    
    def poll_quote(self):
        """轮询报价"""
        while self.running:
            try:
                ret, data = self.quote_ctx.get_market_snapshot(self.stock_list)
                if ret == RET_OK:
                    ts = datetime.now()
                    records = []
                    for _, row in data.iterrows():
                        records.append((
                            ts,
                            row['code'],
                            row.get('name', ''),
                            row.get('last_price'),
                            row.get('open_price'),
                            row.get('high_price'),
                            row.get('low_price'),
                            row.get('prev_close_price'),
                            row.get('volume'),
                            row.get('turnover')
                        ))
                    self.db_writer.insert_quote(records)
                    
            except Exception as e:
                print(f"⚠️ 获取报价失败: {e}")
            
            time.sleep(QUOTE_INTERVAL)
    
    def start(self):
        """开始采集"""
        if not self.subscribe():
            return
        
        self.running = True
        
        # 启动轮询线程
        threading.Thread(target=self.poll_orderbook, daemon=True).start()
        threading.Thread(target=self.poll_quote, daemon=True).start()
        
        print("\n" + "="*50)
        print("  🚀 数据采集已启动！")
        print("  按 Ctrl+C 停止采集")
        print("="*50 + "\n")
        
        # 状态显示
        try:
            while self.running:
                time.sleep(10)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"逐笔: {self.ticker_handler.count} | "
                      f"订单簿: {self.orderbook_handler.count}")
        except KeyboardInterrupt:
            print("\n\n⏹️ 停止采集...")
            self.stop()
    
    def stop(self):
        """停止采集"""
        self.running = False
        self.quote_ctx.close()
        self.db_writer.close()
        
        print("\n" + "="*50)
        print("  📊 采集统计")
        print(f"  逐笔成交: {self.ticker_handler.count} 条")
        print(f"  订单簿:   {self.orderbook_handler.count} 条")
        print("="*50)


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("="*50)
    print("  富途API数据采集器")
    print("="*50)
    print(f"\n采集股票: {STOCK_LIST}")
    print(f"数据库: {DB_CONFIG['database']} @ {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    
    print("\n⚠️  请确保：")
    print("   1. OpenD 已启动并登录")
    print("   2. 港股市场已开市")
    print("   3. Docker数据库容器已运行\n")
    
    input("按 Enter 键开始采集...")
    
    collector = DataCollector(STOCK_LIST)
    collector.start()
