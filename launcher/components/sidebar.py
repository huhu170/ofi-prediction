"""侧边栏组件"""
from tkinter import ttk
from ttkbootstrap.constants import *


class Sidebar(ttk.Frame):
    """侧边栏菜单"""
    
    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self._create_widgets()
    
    def _create_widgets(self):
        """创建按钮"""
        # ttkbootstrap 的 rounded 样式按钮
        btn_style = {"width": 15, "padding": (10, 8)}
        
        # 导航
        self._section("导航")
        ttk.Button(self, text="🏠  主页", bootstyle="secondary-outline",
                   command=lambda: self.app.show_view("main"), **btn_style).pack(pady=4)
        
        # 数据连接
        self._section("数据连接")
        self.db_btn = ttk.Button(self, text="📊  数据库", bootstyle="secondary-outline",
                                  command=self.app.on_db_connect, **btn_style)
        self.db_btn.pack(pady=4)
        
        self.futu_btn = ttk.Button(self, text="📈  富途", bootstyle="secondary-outline",
                                    command=self.app.on_futu_connect, **btn_style)
        self.futu_btn.pack(pady=4)
        
        # 数据查看
        self._section("数据查看")
        ttk.Button(self, text="📅  查看数据", bootstyle="success",
                   command=lambda: self.app.show_view("calendar"), **btn_style).pack(pady=4)
        
        # 数据采集
        self._section("数据采集")
        self.collect_btn = ttk.Button(self, text="▶  腾讯数据", bootstyle="info",
                                       command=self.app.on_collect, **btn_style)
        self.collect_btn.pack(pady=4)
        
        self.stop_btn = ttk.Button(self, text="⬛  停止采集", bootstyle="danger",
                                    command=self.app.on_stop, state=DISABLED, **btn_style)
        self.stop_btn.pack(pady=4)
        
        # 其他功能
        self._section("其他功能")
        for name in ["数据检查", "特征计算", "模型训练", "策略回测"]:
            ttk.Button(self, text=name, bootstyle="secondary-outline",
                       state=DISABLED, **btn_style).pack(pady=3)
    
    def _section(self, title):
        """添加分组标题"""
        ttk.Label(self, text=f"── {title} ──", 
                  font=("Microsoft YaHei", 10),
                  foreground="#8888aa").pack(pady=(16, 6) if title != "导航" else (8, 6))
    
    def update_db_status(self, connected):
        """更新数据库按钮状态"""
        style = "success" if connected else "danger"
        text = "📊  数据库 ✓" if connected else "📊  数据库 ✗"
        self.db_btn.config(bootstyle=style, text=text)
    
    def update_futu_status(self, connected):
        """更新富途按钮状态"""
        style = "success" if connected else "danger"
        text = "📈  富途 ✓" if connected else "📈  富途 ✗"
        self.futu_btn.config(bootstyle=style, text=text)
    
    def set_collecting(self, collecting):
        """设置采集状态"""
        if collecting:
            # 采集中：按钮仍可点击（用于查看日志），只改变样式
            self.collect_btn.config(text="⏳  采集中...", bootstyle="warning")
            self.stop_btn.config(state=NORMAL)
        else:
            self.collect_btn.config(text="▶  腾讯数据", bootstyle="info")
            self.stop_btn.config(state=DISABLED)
