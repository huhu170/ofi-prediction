# -*- coding: utf-8 -*-
"""
Experiment 4.1.1: Sample and Feature Statistics (Merged Script)

Merged experiments:
- exp_4_1_1_sample_stats.py: Sample descriptive statistics
- exp_4_1_2_feature_distribution.py: Feature distribution analysis  
- exp_4_1_3_label_balance.py: Label distribution check

Paper outputs:
- Table 4.1-1: Sample and feature statistics summary

Output files:
- tables/table_4.1-1_sample_feature.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from exp_config import *

def run_all():
    """Run all sub-experiments"""
    print("=" * 60)
    print("Experiment 4.1: Sample and Feature Statistics")
    print("=" * 60)
    
    # 1. Sample descriptive statistics
    print("\n[1/3] Running sample statistics...")
    try:
        from exp_4_1_1_sample_stats import run_experiment as sample_stats_run
        sample_stats_run()
    except Exception as e:
        print(f"  Warning: {e}")
    
    # 2. Feature distribution analysis
    print("\n[2/3] Running feature distribution analysis...")
    try:
        from exp_4_1_2_feature_distribution import run_experiment as feature_dist_run
        feature_dist_run()
    except Exception as e:
        print(f"  Warning: {e}")
    
    # 3. Label distribution check
    print("\n[3/3] Running label balance check...")
    try:
        from exp_4_1_3_label_balance import run_experiment as label_balance_run
        label_balance_run()
    except Exception as e:
        print(f"  Warning: {e}")
    
    print("\n" + "=" * 60)
    print("Experiment 4.1 Complete")
    print("=" * 60)

if __name__ == "__main__":
    run_all()
