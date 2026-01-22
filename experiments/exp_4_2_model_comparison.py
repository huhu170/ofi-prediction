# -*- coding: utf-8 -*-
"""
Experiment 4.2.1: Full Model Performance Comparison (Merged Script)

Merged experiments:
- exp_4_2_1_baseline_models.py: Baseline model evaluation
- exp_4_2_2_deep_models.py: Deep learning model evaluation

Paper outputs:
- Table 4.2-1: Full model performance comparison
- Figure 4.2-1: Model performance visualization

Output files:
- tables/table_4.2-1_model_comparison.csv
- figures/fig_4.2-1_model_comparison.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from exp_config import *

def run_all():
    """Run all sub-experiments"""
    print("=" * 60)
    print("Experiment 4.2.1: Full Model Performance Comparison")
    print("=" * 60)
    
    # 1. Baseline model evaluation
    print("\n[1/2] Running baseline model evaluation...")
    try:
        from exp_4_2_1_baseline_models import main as baseline_main
        baseline_main()
    except Exception as e:
        print(f"  Warning: {e}")
    
    # 2. Deep learning model evaluation
    print("\n[2/2] Running deep learning model evaluation...")
    try:
        from exp_4_2_2_deep_models import main as deep_main
        deep_main()
    except Exception as e:
        print(f"  Warning: {e}")
    
    print("\n" + "=" * 60)
    print("Experiment 4.2.1 Complete")
    print("=" * 60)

if __name__ == "__main__":
    run_all()
