"""
实验主运行脚本
按顺序运行所有实验

使用方法:
    python run_all_experiments.py              # 运行所有实验
    python run_all_experiments.py --section 4.1  # 只运行4.1节
    python run_all_experiments.py --exp 4.1.1    # 只运行特定实验
"""

import sys
import argparse
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from exp_config import EXPERIMENT_SCRIPTS, log_experiment, set_seed

def run_experiment(exp_id: str):
    """运行单个实验"""
    if exp_id not in EXPERIMENT_SCRIPTS:
        print(f"[ERROR] 未知实验ID: {exp_id}")
        return False
    
    script_name, description = EXPERIMENT_SCRIPTS[exp_id]
    module_name = script_name.replace('.py', '')
    
    print(f"\n{'='*60}")
    print(f"  实验 {exp_id}: {description}")
    print(f"  脚本: {script_name}")
    print('='*60)
    
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, 'run_experiment'):
            module.run_experiment()
            return True
        else:
            print(f"[ERROR] 脚本 {script_name} 没有 run_experiment 函数")
            return False
    except Exception as e:
        print(f"[ERROR] 运行实验 {exp_id} 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_section(section: str):
    """运行某一节的所有实验"""
    exp_ids = [eid for eid in EXPERIMENT_SCRIPTS.keys() if eid.startswith(section)]
    
    if not exp_ids:
        print(f"[ERROR] 未找到第 {section} 节的实验")
        return
    
    print(f"\n运行第 {section} 节的 {len(exp_ids)} 个实验")
    
    success = 0
    for exp_id in sorted(exp_ids):
        if run_experiment(exp_id):
            success += 1
    
    print(f"\n{'='*60}")
    print(f"  第 {section} 节实验完成: {success}/{len(exp_ids)} 成功")
    print('='*60)

def run_all():
    """运行所有实验"""
    print("\n" + "="*60)
    print("  论文第四章全部实验")
    print("="*60)
    
    sections = ['4.1', '4.2', '4.3', '4.4']
    
    for section in sections:
        run_section(section)
    
    print("\n" + "="*60)
    print("  所有实验完成!")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='运行论文实验')
    parser.add_argument('--exp', type=str, help='运行特定实验，如 4.1.1')
    parser.add_argument('--section', type=str, help='运行特定章节，如 4.1')
    parser.add_argument('--list', action='store_true', help='列出所有实验')
    
    args = parser.parse_args()
    
    set_seed()
    
    if args.list:
        print("\n" + "="*60)
        print("  可用实验列表")
        print("="*60)
        for exp_id, (script, desc) in sorted(EXPERIMENT_SCRIPTS.items()):
            print(f"  {exp_id}: {desc}")
            print(f"          ({script})")
        return
    
    if args.exp:
        run_experiment(args.exp)
    elif args.section:
        run_section(args.section)
    else:
        run_all()


if __name__ == "__main__":
    main()
