#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键运行所有实验脚本
====================
生成论文第四章所需的全部表格和图表

使用方法:
    python run_all_experiments.py [--section 4.1] [--dry-run]
"""

import argparse
import subprocess
import sys
from pathlib import Path

# 实验脚本列表
EXPERIMENTS = {
    '4.1': [
        ('exp_4_1_1_sample_stats.py', '表4-1: 样本描述性统计'),
        ('exp_4_1_2_ofi_distribution.py', '图4-1/表4-2: OFI分布分析'),
        ('exp_4_1_3_label_balance.py', '表4-3: 标签分布'),
        ('exp_4_1_4_correlation.py', '表4-4: 相关性检验'),
        ('exp_4_1_5_ols_regression.py', '表4-5: OLS回归'),
        ('exp_4_1_6_intraday_impact.py', '图4-2: 日内冲击系数'),
        ('exp_4_1_7_depth_comparison.py', '表4-6/图4-3: 深度对比'),
    ],
    '4.2': [
        ('exp_4_2_1_baseline_models.py', '表4-7: 基准模型'),
        ('exp_4_2_2_deep_models.py', '表4-8: 深度学习模型'),
        ('exp_4_2_3_model_comparison_plot.py', '图4-4: 模型对比图'),
        ('exp_4_2_4_ablation.py', '表4-9: 消融实验'),
        ('exp_4_2_5_threshold_sensitivity.py', '表4-10: 阈值敏感性'),
    ],
    '4.3': [
        ('exp_4_3_1_backtest_config.py', '表4-11: 回测参数'),
        ('exp_4_3_2_backtest.py', '表4-12/图4-5: 策略回测'),
        ('exp_4_3_3_ofi_comparison.py', '表4-13: OFI经济价值'),
    ],
    '4.4': [
        ('exp_4_4_1_shap_analysis.py', '图4-6/图4-7: SHAP分析'),
        ('exp_4_4_2_regime_split.py', '表4-14: 市场状态分组'),
        ('exp_4_4_3_asset_split.py', '表4-15: 资产类型分组'),
        ('exp_4_4_5_control_group.py', '表4-16: 对照组检验'),
    ],
}

def run_experiment(script_name, description, dry_run=False):
    """运行单个实验"""
    script_path = Path(__file__).parent / script_name
    
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  脚本: {script_name}")
    print(f"{'='*60}")
    
    if dry_run:
        print("  [DRY-RUN] 跳过执行")
        return True
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        if result.returncode == 0:
            print(result.stdout)
            print(f"  [OK] 完成")
            return True
        else:
            print(f"  [FAIL] 失败")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] 超时")
        return False
    except Exception as e:
        print(f"  [ERROR] 错误: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='运行论文实验脚本')
    parser.add_argument('--section', type=str, default='all',
                       help='运行特定章节 (4.1, 4.2, 4.3, 4.4, all)')
    parser.add_argument('--dry-run', action='store_true',
                       help='仅显示将要运行的脚本，不实际执行')
    args = parser.parse_args()
    
    print("="*60)
    print("  论文第四章实验脚本运行器")
    print("="*60)
    
    # 确定要运行的章节
    if args.section == 'all':
        sections = ['4.1', '4.2', '4.3', '4.4']
    else:
        sections = [args.section]
    
    # 统计
    total = 0
    success = 0
    failed = []
    
    for section in sections:
        if section not in EXPERIMENTS:
            print(f"\n[WARN] 未知章节: {section}")
            continue
        
        print(f"\n\n{'#'*60}")
        print(f"  第 {section} 节实验")
        print(f"{'#'*60}")
        
        for script, desc in EXPERIMENTS[section]:
            total += 1
            if run_experiment(script, desc, args.dry_run):
                success += 1
            else:
                failed.append(script)
    
    # 汇总
    print(f"\n\n{'='*60}")
    print(f"  运行完成")
    print(f"{'='*60}")
    print(f"  总计: {total}")
    print(f"  成功: {success}")
    print(f"  失败: {len(failed)}")
    
    if failed:
        print(f"\n  失败脚本:")
        for f in failed:
            print(f"    - {f}")
    
    print(f"\n  结果保存至: experiment_results/")
    print(f"    - tables/: 表格文件 (.csv)")
    print(f"    - figures/: 图表文件 (.png)")

if __name__ == '__main__':
    main()
