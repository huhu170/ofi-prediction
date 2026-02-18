# -*- coding: utf-8 -*-
"""
批量重新训练multi_scale模型
1. 生成多尺度数据集
2. 训练模型
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

STOCKS = [
    'HK.00700',  # 腾讯 - 已完成
    'HK.00005',  # 汇丰
    'HK.09988',  # 阿里
    'HK.01810',  # 小米
    'HK.00939',  # 建行
    'HK.01299',  # 友邦
    'HK.00941',  # 中移动
    'HK.03690',  # 美团
    'HK.01211',  # 比亚迪
    'HK.00388',  # 港交所
]

def run_cmd(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    print(f"CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0

def main():
    # 处理所有股票（强制重新生成）
    stocks_to_process = STOCKS  # 全部重做
    
    print(f"待处理股票: {len(stocks_to_process)}")
    
    success = []
    failed = []
    
    for code in stocks_to_process:
        print(f"\n\n{'#'*70}")
        print(f"  处理: {code}")
        print(f"{'#'*70}")
        
        # Step 1: 生成数据集（强制重新生成）
        dataset_path = PROJECT_ROOT / 'data' / 'datasets' / f"dataset_{code.replace('.', '_')}_multi_scale.pkl"
        
        # 删除旧数据集
        if dataset_path.exists():
            dataset_path.unlink()
            print(f"[DELETE] 删除旧数据集: {dataset_path}")
        
        cmd1 = [
            sys.executable,
            str(PROJECT_ROOT / 'scripts' / '12b_kline_dataset_builder.py'),
            '--code', code,
            '--multi-scale',
            '--output', str(PROJECT_ROOT / 'data' / 'datasets')
        ]
        if not run_cmd(cmd1, f"生成数据集: {code}"):
            print(f"[FAILED] 数据集生成失败: {code}")
            failed.append(code)
            continue
        
        # Step 2: 训练模型
        cmd2 = [
            sys.executable,
            str(PROJECT_ROOT / 'scripts' / '13b_kline_model_trainer.py'),
            '--dataset', str(dataset_path),
            '--model', 'multi_scale',
            '--code', code,
            '--epochs', '20'
        ]
        if run_cmd(cmd2, f"训练模型: {code}"):
            success.append(code)
        else:
            failed.append(code)
    
    # 汇总
    print(f"\n\n{'='*70}")
    print(f"  训练完成")
    print(f"{'='*70}")
    print(f"成功: {len(success)} - {success}")
    print(f"失败: {len(failed)} - {failed}")

if __name__ == "__main__":
    main()
