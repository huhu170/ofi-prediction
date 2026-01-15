"""日志视图"""
import tkinter as tk
from tkinter import ttk
from ttkbootstrap.constants import *


class LogView(ttk.Frame):
    """通用日志视图"""
    
    def __init__(self, master, title, icon="📋", **kwargs):
        super().__init__(master, **kwargs)
        self.title_text = title
        self.icon = icon
        self._create_widgets()
    
    def _create_widgets(self):
        """创建界面"""
        ttk.Label(self, text=f"{self.icon}  {self.title_text}", 
                  font=("Microsoft YaHei", 18, "bold")).pack(pady=(10, 15))
        
        log_frame = ttk.Frame(self)
        log_frame.pack(fill=BOTH, expand=YES, padx=15, pady=(0, 12))
        
        self.log_widget = tk.Text(log_frame, font=("Consolas", 10), wrap="word",
                                   bg="#0d1b2a", fg="#e0e0e0", insertbackground="#e0e0e0",
                                   relief="flat", padx=10, pady=8)
        self.log_widget.pack(fill=BOTH, expand=YES)
    
    def log(self, msg):
        """写入日志"""
        self.log_widget.insert("end", msg + "\n")
        self.log_widget.see("end")
    
    def clear(self):
        """清空日志"""
        self.log_widget.delete("1.0", "end")
