"""主页视图"""
import tkinter as tk
from tkinter import ttk
from ttkbootstrap.constants import *

from core.database import db_manager


COLORS = {
    "success": "#2d6a4f",
    "text_dim": "#8888aa",
}


class MainView(ttk.Frame):
    """主页视图"""
    
    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self._create_widgets()
    
    def _create_widgets(self):
        """创建界面"""
        # 标题
        header = ttk.Frame(self)
        header.pack(fill=X, pady=(10, 20))
        ttk.Label(header, text="论文项目脚本启动器", 
                  font=("Microsoft YaHei", 18, "bold")).pack()
        ttk.Label(header, text="点击右侧按钮执行相应功能", 
                  foreground=COLORS["text_dim"]).pack(pady=(5, 0))
        
        # 连接状态卡片
        card1 = ttk.LabelFrame(self, text="  连接状态  ", padding=15)
        card1.pack(fill=X, padx=15, pady=8)
        
        status_frame = ttk.Frame(card1)
        status_frame.pack(fill=X)
        
        # 数据库状态
        db_color = COLORS["success"] if self.app.db_connected else COLORS["text_dim"]
        db_text = "● 数据库: 已连接" if self.app.db_connected else "● 数据库: 未连接"
        self._db_label = ttk.Label(status_frame, text=db_text, foreground=db_color,
                                    font=("Microsoft YaHei", 11))
        self._db_label.pack(anchor=W, pady=2)
        
        # 富途状态
        futu_color = COLORS["success"] if self.app.futu_connected else COLORS["text_dim"]
        futu_text = "● 富途: 已连接" if self.app.futu_connected else "● 富途: 未连接"
        self._futu_label = ttk.Label(status_frame, text=futu_text, foreground=futu_color,
                                      font=("Microsoft YaHei", 11))
        self._futu_label.pack(anchor=W, pady=2)
        
        # 数据统计卡片
        card2 = ttk.LabelFrame(self, text="  数据统计  ", padding=15)
        card2.pack(fill=X, padx=15, pady=8)
        
        stats = db_manager.get_stats()
        if stats:
            ticker = stats['ticker_count']
            orderbook = stats['orderbook_count']
            latest = stats['latest_date']
            
            ttk.Label(card2, text=f"Ticker 记录: {ticker:,}", 
                      font=("Microsoft YaHei", 11)).pack(anchor=W, pady=1)
            ttk.Label(card2, text=f"OrderBook 记录: {orderbook:,}", 
                      font=("Microsoft YaHei", 11)).pack(anchor=W, pady=1)
            if latest:
                ttk.Label(card2, text=f"最新数据: {latest.strftime('%Y-%m-%d')}", 
                          font=("Microsoft YaHei", 11)).pack(anchor=W, pady=1)
        else:
            ttk.Label(card2, text="暂无数据", foreground=COLORS["text_dim"]).pack(anchor=W)
        
        # 系统日志
        ttk.Label(self, text="  系统日志", 
                  font=("Microsoft YaHei", 14, "bold")).pack(anchor=W, padx=15, pady=(15, 8))
        
        log_frame = ttk.Frame(self)
        log_frame.pack(fill=BOTH, expand=YES, padx=15, pady=(0, 12))
        
        self.log_widget = tk.Text(log_frame, font=("Consolas", 10), wrap="word",
                                   bg="#0d1b2a", fg="#e0e0e0", insertbackground="#e0e0e0",
                                   relief="flat", padx=10, pady=8)
        self.log_widget.pack(fill=BOTH, expand=YES)
    
    def log(self, msg):
        """写入日志"""
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_widget.insert("end", f"[{ts}] {msg}\n")
        self.log_widget.see("end")
