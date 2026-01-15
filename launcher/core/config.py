"""配置管理 - ttkbootstrap 版本"""
import json
import sys
from pathlib import Path

# 路径配置（支持打包后运行）
if getattr(sys, 'frozen', False):
    # 打包后的路径
    BASE_DIR = Path(sys.executable).parent
    LAUNCHER_DIR = BASE_DIR
    PROJECT_DIR = BASE_DIR
    SCRIPTS_DIR = BASE_DIR / "scripts"
    CONFIG_FILE = BASE_DIR / "config.json"
else:
    # 开发环境
    LAUNCHER_DIR = Path(__file__).parent.parent
    PROJECT_DIR = LAUNCHER_DIR.parent
    SCRIPTS_DIR = PROJECT_DIR / "scripts"
    CONFIG_FILE = LAUNCHER_DIR / "config.json"

# 加载配置文件
def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

CONFIG = load_config()

# 字体配置
FONT_FAMILY = "Microsoft YaHei"
