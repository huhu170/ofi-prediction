# -*- coding: utf-8 -*-
"""
Experiment 4.1.2: Feature Explanatory Power Analysis (Merged Script)

Merged experiments:
- exp_4_1_4_correlation.py: Correlation analysis
- exp_4_1_5_ols_regression.py: OLS regression analysis
- exp_4_1_6_scale_comparison.py: Multi-scale explanatory power comparison

Paper outputs:
- Table 4.1-2: Feature explanatory power analysis
- Figure 4.1-1: Feature distribution and explanatory power visualization

Output files:
- tables/table_4.1-2_feature_analysis.csv
- figures/fig_4.1-1_feature_analysis.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from exp_config import *

def run_all():
    """Run all sub-experiments"""
    print("=" * 60)
    print("Experiment 4.1.2: Feature Explanatory Power Analysis")
    print("=" * 60)
    
    # 1. Correlation analysis
    print("\n[1/3] Running correlation analysis...")
    try:
        from exp_4_1_4_correlation import run_experiment as correlation_run
        correlation_run()
    except Exception as e:
        print(f"  Warning: {e}")
    
    # 2. OLS regression analysis
    print("\n[2/3] Running OLS regression analysis...")
    try:
        from exp_4_1_5_ols_regression import run_experiment as ols_run
        ols_run()
    except Exception as e:
        print(f"  Warning: {e}")
    
    # 3. Multi-scale explanatory power comparison
    print("\n[3/3] Running multi-scale comparison...")
    try:
        from exp_4_1_6_scale_comparison import run_experiment as scale_run
        scale_run()
    except Exception as e:
        print(f"  Warning: {e}")
    
    print("\n" + "=" * 60)
    print("Experiment 4.1.2 Complete")
    print("=" * 60)

if __name__ == "__main__":
    run_all()
