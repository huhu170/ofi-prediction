"""数据库连接处理"""
import threading
from core.database import db_manager


class DatabaseHandler:
    """数据库连接处理器"""
    
    def __init__(self, app):
        self.app = app
    
    def connect(self, log_view, on_done=None):
        """异步执行连接"""
        log_view.log("Connecting to database...")
        
        def do_connect():
            success, msg = db_manager.test_connection()
            status = "✓ OK" if success else "✗ Failed"
            self.app.after(0, lambda: log_view.log(f"{status}: {msg}"))
            
            if on_done:
                self.app.after(0, lambda: on_done(success))
        
        threading.Thread(target=do_connect, daemon=True).start()
