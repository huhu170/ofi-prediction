"""富途连接处理"""
import threading
from core.config import CONFIG


class FutuHandler:
    """富途连接处理器"""
    
    def __init__(self, app):
        self.app = app
    
    def connect(self, log_view, on_done=None):
        """异步执行连接"""
        log_view.log("正在连接富途...")
        
        def do_connect():
            success = False
            try:
                from futu import OpenQuoteContext, RET_OK
                host = CONFIG.get("futu", {}).get("host", "127.0.0.1")
                port = CONFIG.get("futu", {}).get("port", 11111)
                
                self.app.after(0, lambda h=host, p=port: log_view.log(f"连接 {h}:{p} ..."))
                ctx = OpenQuoteContext(host=host, port=port)
                ret, data = ctx.get_global_state()
                ctx.close()
                
                if ret == RET_OK:
                    self.app.after(0, lambda d=str(data): log_view.log(f"✓ 成功: {d}"))
                    success = True
                else:
                    raise Exception(data)
                    
            except ImportError as e:
                self.app.after(0, lambda: log_view.log("✗ 失败: futu 库未安装"))
                self.app.after(0, lambda: log_view.log("运行: pip install futu-api"))
            except Exception as e:
                err_msg = str(e)
                self.app.after(0, lambda m=err_msg: log_view.log(f"✗ 失败: {m}"))
                self.app.after(0, lambda: log_view.log("请确保 FutuOpenD 已启动"))
            
            if on_done:
                self.app.after(0, lambda s=success: on_done(s))
        
        threading.Thread(target=do_connect, daemon=True).start()
