"""
日历组件 - ttkbootstrap 版本
简洁可靠 + 缓存优化
"""
import calendar
from datetime import datetime
import tkinter as tk
from tkinter import ttk
import threading


class CalendarView(ttk.Frame):
    """数据日历视图"""
    
    def __init__(self, master, db_manager, **kwargs):
        super().__init__(master, **kwargs)
        
        self.db_manager = db_manager
        self.current_year = datetime.now().year
        self.current_month = datetime.now().month
        self.data_counts = {}
        self.selected_date = None
        self._detail_cache = {}
        
        # 防抖控制
        self._resize_job = None
        self._last_size = (0, 0)
        
        self._create_widgets()
        self.after(100, self._load_data)
    
    def _create_widgets(self):
        """创建界面"""
        # 导航栏
        nav = ttk.Frame(self)
        nav.pack(fill="x", padx=15, pady=(0, 8))
        
        ttk.Button(nav, text="◀ 上月", width=8, style="info-outline.TButton",
                   command=self._prev_month).pack(side="left")
        
        self._month_label = ttk.Label(nav, text="", 
                                       font=("Microsoft YaHei", 16, "bold"))
        self._month_label.pack(side="left", expand=True)
        
        ttk.Button(nav, text="下月 ▶", width=8, style="info-outline.TButton",
                   command=self._next_month).pack(side="right")
        
        # 日历画布
        self._canvas = tk.Canvas(self, height=280, bg="#1a1d21",
                                  highlightthickness=0, relief="flat")
        self._canvas.pack(fill="both", expand=True, padx=10)
        self._canvas.bind("<Button-1>", self._on_click)
        self._canvas.bind("<Configure>", self._on_resize)
        
        # 详情区
        detail_label = ttk.Label(self, text="📋 日期详情", 
                                  font=("Microsoft YaHei", 11, "bold"))
        detail_label.pack(anchor="w", padx=15, pady=(12, 6))
        
        self._detail = tk.Text(self, font=("Microsoft YaHei", 10), wrap="word",
                                height=8, bg="#1a1d21", fg="#c8d0d8", 
                                relief="flat", padx=12, pady=10)
        self._detail.pack(fill="x", padx=10, pady=(0, 10))
        self._detail.insert("end", "👆 点击日历日期查看采集详情")
    
    def _load_data(self):
        """异步加载日历数据"""
        def load():
            counts = self.db_manager.get_daily_counts()
            self.after(0, lambda: self._on_data_loaded(counts))
            
            # 后台预加载详情
            for date_str in counts.keys():
                if date_str not in self._detail_cache:
                    detail = self.db_manager.get_date_detail(date_str)
                    self._detail_cache[date_str] = detail
        
        threading.Thread(target=load, daemon=True).start()
    
    def _on_data_loaded(self, data):
        """数据加载完成"""
        self.data_counts = data
        self._draw()
    
    def _on_resize(self, event):
        """窗口大小变化 - 防抖处理"""
        new_size = (event.width, event.height)
        if abs(new_size[0] - self._last_size[0]) < 30:
            return
        
        self._last_size = new_size
        
        # 取消之前的重绘任务
        if self._resize_job:
            self.after_cancel(self._resize_job)
        
        # 延迟 100ms 重绘
        self._resize_job = self.after(100, self._draw)
    
    def _get_color(self, count, selected=False):
        """获取单元格颜色"""
        if selected:
            return "#3d8bfd", "#ffffff"
        elif count >= 10000:
            return "#198754", "#ffffff"
        elif count >= 1000:
            return "#20c997", "#ffffff"
        elif count > 0:
            return "#0d6efd", "#ffffff"
        else:
            return "#2b3035", "#6c757d"
    
    def _draw(self):
        """绘制日历"""
        if not self._canvas.winfo_exists():
            return
        
        self._canvas.delete("all")
        self._month_label.config(text=f"{self.current_year} 年 {self.current_month} 月")
        
        w = self._canvas.winfo_width()
        h = self._canvas.winfo_height()
        if w < 100:
            w = 700
        if h < 100:
            h = 280
        
        padding = 10
        cell_w = (w - padding * 2) / 7
        header_h = 32
        cell_h = (h - header_h - padding) / 6
        
        # 星期标题
        days = ["一", "二", "三", "四", "五", "六", "日"]
        for i, d in enumerate(days):
            x = padding + i * cell_w + cell_w / 2
            color = "#dc3545" if i >= 5 else "#adb5bd"
            self._canvas.create_text(x, header_h / 2, text=d, fill=color,
                                      font=("Microsoft YaHei", 11, "bold"))
        
        # 分隔线
        self._canvas.create_line(padding, header_h, w - padding, header_h, 
                                  fill="#3a3f44", width=1)
        
        # 日期格子
        cal = calendar.Calendar(firstweekday=0)
        weeks = cal.monthdayscalendar(self.current_year, self.current_month)
        
        # 存储日期位置用于点击检测
        self._date_rects = {}
        
        for wi, week in enumerate(weeks):
            for di, day in enumerate(week):
                if day == 0:
                    continue
                
                x = padding + di * cell_w
                y = header_h + 4 + wi * cell_h
                
                date_str = f"{self.current_year}-{self.current_month:02d}-{day:02d}"
                count = self.data_counts.get(date_str, 0)
                selected = (date_str == self.selected_date)
                
                bg, fg = self._get_color(count, selected)
                
                # 圆角矩形
                r = 10
                x1, y1 = x + 4, y + 3
                x2, y2 = x + cell_w - 4, y + cell_h - 3
                
                # 存储日期边界
                self._date_rects[date_str] = (x1, y1, x2, y2)
                
                self._canvas.create_polygon(
                    x1+r, y1, x2-r, y1, x2, y1, x2, y1+r,
                    x2, y2-r, x2, y2, x2-r, y2, x1+r, y2,
                    x1, y2, x1, y2-r, x1, y1+r, x1, y1,
                    fill=bg, outline="", smooth=True
                )
                
                # 日期文字
                self._canvas.create_text(x + cell_w/2, y + cell_h*0.35, text=str(day),
                                          fill=fg, font=("Microsoft YaHei", 12, "bold"))
                
                # 数据量
                if count > 0:
                    count_str = f"{count//1000}k" if count >= 1000 else str(count)
                    self._canvas.create_text(x + cell_w/2, y + cell_h*0.72, text=count_str,
                                              fill=fg, font=("Microsoft YaHei", 9))
    
    def _on_click(self, event):
        """点击日期"""
        # 通过坐标查找日期
        clicked_date = None
        for date_str, (x1, y1, x2, y2) in self._date_rects.items():
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                clicked_date = date_str
                break
        
        if not clicked_date or clicked_date == self.selected_date:
            return
        
        # 更新选中状态并重绘
        self.selected_date = clicked_date
        self._draw()
        
        # 显示详情
        self._show_detail(clicked_date)
    
    def _show_detail(self, date_str):
        """显示日期详情"""
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        
        # 当日数据不使用缓存（实时查询），其他日期用缓存
        if date_str != today and date_str in self._detail_cache:
            self._render_detail(date_str, self._detail_cache[date_str])
            return
        
        # 实时加载
        self._detail.delete("1.0", "end")
        self._detail.insert("end", f"⏳ 正在加载 {date_str} ...")
        
        def load():
            # 清除 DatabaseManager 中该日期的缓存
            if date_str in self.db_manager._detail_cache:
                del self.db_manager._detail_cache[date_str]
            
            data = self.db_manager.get_date_detail(date_str)
            self._detail_cache[date_str] = data
            # 检查是否仍是当前选中
            if self.selected_date == date_str:
                self.after(0, lambda: self._render_detail(date_str, data))
        
        threading.Thread(target=load, daemon=True).start()
    
    def _render_detail(self, date_str, data):
        """渲染详情"""
        if not self._detail.winfo_exists():
            return
        
        self._detail.delete("1.0", "end")
        
        if "error" in data:
            self._detail.insert("end", f"❌ 查询失败: {data['error']}")
            return
        
        ticker = data["ticker_count"]
        orderbook = data["orderbook_count"]
        
        if ticker + orderbook > 0:
            vol = data.get("total_volume", 0)
            turnover = data.get("total_turnover", 0)
            vol_str = f"{vol:,}" if vol < 100000000 else f"{vol/100000000:.2f}亿"
            turnover_str = f"{turnover/100000000:.2f}亿" if turnover >= 100000000 else f"{turnover/10000:.2f}万"
            
            tr = data["time_range"]
            time_str = f"{tr[0].strftime('%H:%M:%S')} ~ {tr[1].strftime('%H:%M:%S')}" if tr[0] and tr[1] else "--"
            
            stocks = ", ".join([f"{code}: {cnt:,}" for code, cnt in data["stock_details"]]) if data["stock_details"] else "--"
            
            # 第1行：日期 + 时段 + 股票 + 合计
            self._detail.insert("end", f"📅 {date_str}    ⏰ {time_str}    📋 按股票: {stocks}    📦 合计: {ticker + orderbook:,}\n\n")
            
            # 第2行：详细统计
            self._detail.insert("end", f"📊 Ticker: {ticker:,}    📈 OrderBook: {orderbook:,}    💹 总成交量: {vol_str}    💰 总成交额: {turnover_str}\n")
        else:
            self._detail.insert("end", "📭 该日期暂无数据")
    
    def _prev_month(self):
        """上个月"""
        if self.current_month == 1:
            self.current_month = 12
            self.current_year -= 1
        else:
            self.current_month -= 1
        self.selected_date = None
        self._draw()
    
    def _next_month(self):
        """下个月"""
        if self.current_month == 12:
            self.current_month = 1
            self.current_year += 1
        else:
            self.current_month += 1
        self.selected_date = None
        self._draw()
    
    def refresh(self):
        """刷新数据"""
        self._load_data()
