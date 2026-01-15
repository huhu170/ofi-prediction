"""数据采集处理"""
import os
import sys
import subprocess
import threading
import shutil
from pathlib import Path


def get_python_executable():
    """获取 Python 解释器路径"""
    if getattr(sys, 'frozen', False):
        # 打包后：使用系统 Python
        python = shutil.which('python')
        if python:
            return python
        # 备选路径
        for path in [
            r'C:\Users\12995\AppData\Local\Programs\Python\Python312\python.exe',
            r'C:\Python312\python.exe',
            r'C:\Python311\python.exe',
            r'C:\Python310\python.exe',
        ]:
            if os.path.exists(path):
                return path
        return 'python'  # 尝试 PATH 中的 python
    else:
        # 开发环境：使用当前 Python
        return sys.executable


class CollectorHandler:
    """数据采集处理器"""
    
    def __init__(self, app, scripts_dir):
        self.app = app
        self.scripts_dir = Path(scripts_dir)
        self._process = None
    
    @property
    def is_running(self):
        return self._process is not None
    
    def start(self, script_name, log_view, on_done=None):
        """开始采集"""
        if self._process:
            return False
        
        script = self.scripts_dir / script_name
        if not script.exists():
            log_view.log(f"脚本不存在: {script}")
            return False
        
        python_exe = get_python_executable()
        log_view.log(f"Python: {python_exe}")
        log_view.log(f"脚本: {script}")
        log_view.log("=" * 50)
        
        self._on_done = on_done
        
        def run():
            # CREATE_NO_WINDOW 隐藏 Python 控制台窗口
            self._process = subprocess.Popen(
                [python_exe, str(script)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(self.scripts_dir),
                encoding='utf-8', errors='replace',
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            for line in self._process.stdout:
                line = line.strip()
                if line:
                    self.app.after(0, lambda l=line: log_view.log(l))
            
            self._process.wait()
            self.app.after(0, lambda: self._finish(log_view))
        
        threading.Thread(target=run, daemon=True).start()
        return True
    
    def stop(self):
        """停止采集"""
        if self._process:
            self._process.terminate()
            self._process = None
    
    def _finish(self, log_view):
        """采集完成"""
        self._process = None
        log_view.log("=" * 50)
        log_view.log("采集已停止")
        if self._on_done:
            self._on_done()
