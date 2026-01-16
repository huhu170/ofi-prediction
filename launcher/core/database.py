"""数据库管理"""
import psycopg2
from core.config import CONFIG


class DatabaseManager:
    """数据库连接管理器"""
    
    def __init__(self):
        self._conn = None
        self._detail_cache = {}  # 缓存日期详情
    
    def get_connection(self):
        """获取新的数据库连接"""
        db_config = CONFIG.get("database", {})
        return psycopg2.connect(
            host=db_config.get("host", "localhost"),
            port=db_config.get("port", 5433),
            database=db_config.get("database", "futu_ofi"),
            user=db_config.get("user", "postgres"),
            password=db_config.get("password", "")
        )
    
    def test_connection(self):
        """测试连接，返回 (成功, 信息)"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            return True, version[:60]
        except Exception as e:
            return False, str(e)
    
    def get_stats(self):
        """获取数据统计"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM ticker")
            ticker_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM orderbook")
            orderbook_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT MAX(ts::date) FROM ticker")
            latest_date = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            return {
                "ticker_count": ticker_count,
                "orderbook_count": orderbook_count,
                "latest_date": latest_date
            }
        except:
            return None
    
    def get_daily_counts(self):
        """获取每日数据量"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT trade_time::date as day, COUNT(*) as cnt FROM ticker GROUP BY trade_time::date")
            result = {row[0].strftime("%Y-%m-%d"): row[1] for row in cursor.fetchall()}
            cursor.close()
            conn.close()
            return result
        except:
            return {}
    
    def get_date_detail(self, date_str):
        """获取指定日期的详细数据（带缓存）"""
        # 检查缓存
        if date_str in self._detail_cache:
            return self._detail_cache[date_str]
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 使用范围查询（按 trade_time，能用索引和分区）
            start = f"{date_str} 00:00:00"
            end = f"{date_str} 23:59:59"
            
            # 统计 ticker（新主键确保 sequence 唯一，无需去重）
            cursor.execute("""
                SELECT COUNT(*), MIN(trade_time), MAX(trade_time),
                       COALESCE(SUM(volume), 0), COALESCE(SUM(turnover), 0)
                FROM ticker WHERE trade_time BETWEEN %s AND %s
            """, (start, end))
            row = cursor.fetchone()
            ticker_count = row[0]
            time_range = (row[1], row[2])
            total_volume = int(row[3]) if row[3] else 0
            total_turnover = float(row[4]) if row[4] else 0.0
            
            # orderbook 计数
            cursor.execute("""
                SELECT COUNT(*) FROM orderbook WHERE ts BETWEEN %s AND %s
            """, (start, end))
            orderbook_count = cursor.fetchone()[0]
            
            # 按股票统计（只在有数据时查询）
            stock_details = []
            if ticker_count > 0:
                cursor.execute("""
                    SELECT code, COUNT(*) as cnt FROM ticker 
                    WHERE trade_time BETWEEN %s AND %s GROUP BY code ORDER BY cnt DESC
                """, (start, end))
                stock_details = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            result = {
                "ticker_count": ticker_count,
                "orderbook_count": orderbook_count,
                "stock_details": stock_details,
                "time_range": time_range,
                "total_volume": total_volume,
                "total_turnover": total_turnover
            }
            
            # 缓存结果
            self._detail_cache[date_str] = result
            return result
            
        except Exception as e:
            return {"error": str(e)}


# 全局实例
db_manager = DatabaseManager()
