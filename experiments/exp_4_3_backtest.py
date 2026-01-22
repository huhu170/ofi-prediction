# -*- coding: utf-8 -*-
"""
Experiment 4.3.1: Model Economic Value Comparison (Merged Script)

Merged experiments:
- exp_4_3_1_backtest_config.py: Backtest parameter configuration
- exp_4_3_2_backtest.py: Model backtest
- exp_4_3_3_scale_comparison.py: Multi-scale backtest comparison

Paper outputs:
- Table 4.3-1: Model economic value comparison
- Figure 4.3-1: Strategy equity curves

Output files:
- tables/table_4.3-1_backtest.csv
- figures/fig_4.3-1_equity_curves.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from exp_config import *

def run_all():
    """Run all sub-experiments"""
    print("=" * 60)
    print("Experiment 4.3.1: Model Economic Value Comparison")
    print("=" * 60)
    
    # 1. Backtest parameter configuration
    print("\n[1/3] Configuring backtest parameters...")
    try:
        from exp_4_3_1_backtest_config import main as config_main
        config_main()
    except Exception as e:
        print(f"  Warning: {e}")
    
    # 2. Model backtest
    print("\n[2/3] Running model backtest...")
    try:
        from exp_4_3_2_backtest import main as backtest_main
        backtest_main()
    except Exception as e:
        print(f"  Warning: {e}")
    
    # 3. Multi-scale backtest comparison
    print("\n[3/3] Running multi-scale backtest comparison...")
    try:
        from exp_4_3_3_scale_comparison import main as scale_main
        scale_main()
    except Exception as e:
        print(f"  Warning: {e}")
    
    print("\n" + "=" * 60)
    print("Experiment 4.3.1 Complete")
    print("=" * 60)

if __name__ == "__main__":
    run_all()
