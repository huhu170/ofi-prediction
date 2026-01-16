# -*- mode: python ; coding: utf-8 -*-
"""
Paper Project 脚本启动器 - PyInstaller 打包配置
"""

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

# 项目路径
PROJECT_DIR = Path(SPECPATH)
LAUNCHER_DIR = PROJECT_DIR / 'launcher'

# 收集 futu 包的所有内容
futu_datas, futu_binaries, futu_hiddenimports = collect_all('futu')

a = Analysis(
    [str(LAUNCHER_DIR / 'launcher.py')],
    pathex=[str(LAUNCHER_DIR)],
    binaries=futu_binaries,
    datas=[
        # 配置文件
        (str(LAUNCHER_DIR / 'config.json'), '.'),
    ] + futu_datas,
    hiddenimports=[
        # ttkbootstrap 相关
        'ttkbootstrap',
        'ttkbootstrap.constants',
        'ttkbootstrap.style',
        'ttkbootstrap.themes',
        'ttkbootstrap.themes.standard',
        # 本地模块
        'components',
        'components.calendar',
        'components.sidebar',
        'views',
        'views.main_view',
        'views.log_view',
        'handlers',
        'handlers.database',
        'handlers.futu',
        'handlers.collector',
        'core',
        'core.config',
        'core.database',
        # 数据库
        'psycopg2',
        'psycopg2.extensions',
        'psycopg2.extras',
        # futu-api 相关
        'futu',
        'futu.common',
        'futu.quote',
        'futu.trade',
        'google.protobuf',
        'google.protobuf.descriptor',
        'google.protobuf.message',
        'google.protobuf.reflection',
        'google.protobuf.descriptor_pb2',
        'google.protobuf.internal',
        'google.protobuf.internal.builder',
        'google.protobuf.internal.containers',
        'google.protobuf.internal.decoder',
        'google.protobuf.internal.encoder',
        'google.protobuf.internal.enum_type_wrapper',
        'google.protobuf.internal.message_listener',
        'google.protobuf.internal.python_message',
        'google.protobuf.internal.type_checkers',
        'google.protobuf.internal.well_known_types',
        'google.protobuf.internal.wire_format',
        # 其他
        'PIL',
        'PIL._tkinter_finder',
    ] + futu_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'scipy',
        'pytest',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Paper Project 启动器',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # 无控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 可以添加图标: icon='icon.ico'
)
