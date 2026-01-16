"""富途连接处理"""
import threading
from core.config import CONFIG


class FutuHandler:
    """富途连接处理器"""
    
    def __init__(self, app):
        self.app = app
    
    def connect(self, log_view, on_done=None):
        """异步执行连接"""
        log_view.log("Connecting to Futu...")
        
        def do_connect():
            success = False
            try:
                from futu import OpenQuoteContext, RET_OK
                host = CONFIG.get("futu", {}).get("host", "127.0.0.1")
                port = CONFIG.get("futu", {}).get("port", 11111)
                
                self.app.after(0, lambda h=host, p=port: log_view.log(f"Connecting {h}:{p} ..."))
                ctx = OpenQuoteContext(host=host, port=port)
                ret, data = ctx.get_global_state()
                ctx.close()
                
                if ret == RET_OK:
                    self.app.after(0, lambda d=str(data): log_view.log(f"✓ OK: {d}"))
                    success = True
                else:
                    raise Exception(data)
                    
            except ImportError as e:
                self.app.after(0, lambda: log_view.log("✗ Failed: futu library not installed"))
                self.app.after(0, lambda: log_view.log("Run: pip install futu-api"))
            except Exception as e:
                err_msg = str(e)
                self.app.after(0, lambda m=err_msg: log_view.log(f"✗ Failed: {m}"))
                self.app.after(0, lambda: log_view.log("Please ensure FutuOpenD is running"))
            
            if on_done:
                self.app.after(0, lambda s=success: on_done(s))
        
        threading.Thread(target=do_connect, daemon=True).start()
