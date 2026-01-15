"""
论文项目脚本启动器 v3.1
模块化架构
"""

import sys
from tkinter import ttk
from pathlib import Path

# 打包后支持
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
    LAUNCHER_DIR = BASE_DIR
    SCRIPTS_DIR = BASE_DIR / "scripts"
else:
    LAUNCHER_DIR = Path(__file__).parent
    PROJECT_DIR = LAUNCHER_DIR.parent
    SCRIPTS_DIR = PROJECT_DIR / "scripts"

if str(LAUNCHER_DIR) not in sys.path:
    sys.path.insert(0, str(LAUNCHER_DIR))

import ttkbootstrap as ttkb
from ttkbootstrap.constants import *

from components import CalendarView, Sidebar
from views import MainView, LogView
from handlers import DatabaseHandler, FutuHandler, CollectorHandler
from core.database import db_manager


class LauncherApp(ttkb.Window):
    """主应用"""
    
    def __init__(self):
        super().__init__(themename="superhero")
        
        self.title("论文项目脚本启动器")
        self.geometry("1100x750")
        self.minsize(900, 550)
        
        # 状态
        self.db_connected = False
        self.futu_connected = False
        self.current_view = None
        self._current_log_view = None
        
        # 处理器
        self._db_handler = DatabaseHandler(self)
        self._futu_handler = FutuHandler(self)
        self._collector = CollectorHandler(self, SCRIPTS_DIR)
        
        # 界面
        self._configure_styles()
        self._create_ui()
        self.after(50, lambda: self.show_view("main"))
    
    def _configure_styles(self):
        """配置样式"""
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Microsoft YaHei", 18, "bold"))
        style.configure("Subtitle.TLabel", font=("Microsoft YaHei", 14, "bold"))
    
    def _create_ui(self):
        """创建界面"""
        main = ttk.Frame(self, padding=12)
        main.pack(fill=BOTH, expand=YES)
        
        # 左侧内容区
        self.content = ttk.Frame(main)
        self.content.pack(side=LEFT, fill=BOTH, expand=YES, padx=(0, 12))
        
        # 右侧菜单栏
        self.sidebar = Sidebar(main, self)
        self.sidebar.pack(side=RIGHT, fill=Y)
    
    # ============================================================
    # 公共接口（供 Sidebar 调用）
    # ============================================================
    
    def show_view(self, name):
        """切换视图"""
        if self.current_view == name:
            return
        
        for w in self.content.winfo_children():
            w.destroy()
        
        self._current_log_view = None
        
        if name == "main":
            view = MainView(self.content, self)
            view.pack(fill=BOTH, expand=YES)
            view.log("启动器已就绪")
            self._main_view = view
            
        elif name == "calendar":
            ttk.Label(self.content, text="📅  数据采集日历", 
                      font=("Microsoft YaHei", 18, "bold")).pack(pady=(10, 8))
            CalendarView(self.content, db_manager).pack(fill=BOTH, expand=YES, padx=5, pady=5)
            
        elif name == "db":
            view = LogView(self.content, "数据库连接", "📊")
            view.pack(fill=BOTH, expand=YES)
            self._current_log_view = view
            
        elif name == "futu":
            view = LogView(self.content, "富途连接", "📈")
            view.pack(fill=BOTH, expand=YES)
            self._current_log_view = view
            
        elif name == "collect":
            view = LogView(self.content, "腾讯数据采集", "▶")
            view.pack(fill=BOTH, expand=YES)
            self._current_log_view = view
        
        self.current_view = name
    
    def on_db_connect(self):
        """数据库连接"""
        self.show_view("db")
        # 延迟执行，让 UI 先刷新
        self.after(50, self._do_db_connect)
    
    def _do_db_connect(self):
        """执行数据库连接（异步）"""
        def on_done(success):
            self.db_connected = success
            self.sidebar.update_db_status(success)
        
        self._db_handler.connect(self._current_log_view, on_done=on_done)
    
    def on_futu_connect(self):
        """富途连接"""
        self.show_view("futu")
        # 延迟执行，让 UI 先刷新
        self.after(50, self._do_futu_connect)
    
    def _do_futu_connect(self):
        """执行富途连接（异步）"""
        def on_done(success):
            self.futu_connected = success
            self.sidebar.update_futu_status(success)
        
        self._futu_handler.connect(self._current_log_view, on_done=on_done)
    
    def on_collect(self):
        """开始采集"""
        if self._collector.is_running:
            return
        self.show_view("collect")
        self.sidebar.set_collecting(True)
        self._collector.start("09_collect_tencent.py", self._current_log_view, 
                               on_done=lambda: self.sidebar.set_collecting(False))
    
    def on_stop(self):
        """停止采集"""
        self._collector.stop()
        self.sidebar.set_collecting(False)


if __name__ == "__main__":
    app = LauncherApp()
    app.mainloop()
