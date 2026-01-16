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
        self._stopped = False  # 手动停止标记
    
    @property
    def is_running(self):
        return self._process is not None and self._process.poll() is None
    
    def start(self, script_name, log_view, on_done=None):
        """开始采集"""
        if self._process and self._process.poll() is None:
            return False
        
        script = self.scripts_dir / script_name
        if not script.exists():
            log_view.log(f"Script not found: {script}")
            return False
        
        python_exe = get_python_executable()
        log_view.log(f"Python: {python_exe}")
        log_view.log(f"Script: {script}")
        log_view.log("=" * 50)
        
        self._on_done = on_done
        self._stopped = False
        self._log_view = log_view  # 保存引用
        
        def run():
            try:
                # CREATE_NO_WINDOW 隐藏 Python 控制台窗口
                self._process = subprocess.Popen(
                    [python_exe, str(script)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, cwd=str(self.scripts_dir),
                    encoding='utf-8', errors='replace',
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                
                # 读取输出
                while True:
                    proc = self._process  # 本地引用，避免竞态
                    if proc is None:
                        break
                    line = proc.stdout.readline()
                    if not line:
                        if proc.poll() is not None:
                            break
                        continue
                    line = line.strip()
                    if line:
                        self.app.after(0, lambda l=line: log_view.log(l))
                
                # 等待进程结束
                proc = self._process
                if proc:
                    try:
                        proc.wait()
                    except Exception:
                        pass
            except Exception as e:
                self.app.after(0, lambda: log_view.log(f"Error: {e}"))
            finally:
                self.app.after(0, lambda: self._finish(log_view))
        
        threading.Thread(target=run, daemon=True).start()
        return True
    
    def stop(self, log_view=None):
        """停止采集"""
        self._stopped = True
        if self._process:
            pid = self._process.pid
            try:
                # 直接强制杀死进程树（包括子进程）
                os.system(f'taskkill /PID {pid} /F /T >nul 2>&1')
            except Exception:
                pass
            finally:
                self._process = None
        
        # 显示停止信息
        view = log_view or self._log_view
        if view:
            try:
                view.log("=" * 50)
                view.log("[STOPPED] Collection stopped by user")
            except Exception:
                pass
        
        if self._on_done:
            self._on_done()
    
    def _finish(self, log_view):
        """采集完成（自然结束时调用）"""
        self._process = None
        # 手动停止时，stop() 已经显示了信息，这里只处理自然结束
        if not self._stopped:
            try:
                log_view.log("=" * 50)
                log_view.log("[DONE] Collection completed")
            except Exception:
                pass
            if self._on_done:
                self._on_done()
