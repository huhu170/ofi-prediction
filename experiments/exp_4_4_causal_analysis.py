# -*- coding: utf-8 -*-
"""
Experiment 4.4.2: Causal Inference Analysis (Merged Script)

Merged experiments:
- exp_4_4_5_granger_causality.py: Granger causality test
- exp_4_4_6_causal_feature_comparison.py: Causal feature subset validation
- exp_4_4_7_counterfactual.py: Counterfactual analysis
- exp_4_4_11_shap_vs_causal.py: SHAP vs Granger comparison

Paper outputs:
- Table 4.4-1: Causal inference analysis
- Figure 4.4-3: Counterfactual effect curves

Output files:
- tables/table_4.4-1_causal.csv
- figures/fig_4.4-3_causal.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from exp_config import *

def run_all():
    """Run all sub-experiments"""
    print("=" * 60)
    print("Experiment 4.4.2: Causal Inference Analysis")
    print("=" * 60)
    
    # 1. Granger causality test
    print("\n[1/4] Running Granger causality test...")
    try:
        from exp_4_4_5_granger_causality import main as granger_main
        granger_main()
    except Exception as e:
        print(f"  Warning: {e}")
    
    # 2. Causal feature subset validation
    print("\n[2/4] Running causal feature subset validation...")
    try:
        from exp_4_4_6_causal_feature_comparison import main as causal_feature_main
        causal_feature_main()
    except Exception as e:
        print(f"  Warning: {e}")
    
    # 3. Counterfactual analysis
    print("\n[3/4] Running counterfactual analysis...")
    try:
        from exp_4_4_7_counterfactual import main as counterfactual_main
        counterfactual_main()
    except Exception as e:
        print(f"  Warning: {e}")
    
    # 4. SHAP vs Granger comparison
    print("\n[4/4] Running SHAP vs Granger comparison...")
    try:
        from exp_4_4_11_shap_vs_causal import main as shap_vs_causal_main
        shap_vs_causal_main()
    except Exception as e:
        print(f"  Warning: {e}")
    
    print("\n" + "=" * 60)
    print("Experiment 4.4.2 Complete")
    print("=" * 60)

if __name__ == "__main__":
    run_all()
