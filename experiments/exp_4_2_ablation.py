# -*- coding: utf-8 -*-
"""
Experiment 4.2.2: Architecture Ablation Study (Merged Script)

Merged experiments:
- exp_4_2_3a_pv_crossattn_ablation.py: PV-CrossAttention ablation
- exp_4_2_3b_lsf_ablation.py: LSF ablation study

Paper outputs:
- Table 4.2-2: Architecture ablation results
- Figure 4.2-2a: Attention weight heatmap
- Figure 4.2-2b: LSF gating weight time series

Output files:
- tables/table_4.2-2_ablation.csv
- figures/fig_4.2-2a_attention_heatmap.png
- figures/fig_4.2-2b_scale_weights.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from exp_config import *

def run_all():
    """Run all sub-experiments"""
    print("=" * 60)
    print("Experiment 4.2.2: Architecture Ablation Study")
    print("=" * 60)
    
    # 1. PV-CrossAttention ablation
    print("\n[1/2] Running PV-CrossAttention ablation...")
    try:
        from exp_4_2_3a_pv_crossattn_ablation import main as pv_ablation_main
        pv_ablation_main()
    except Exception as e:
        print(f"  Warning: {e}")
    
    # 2. LSF ablation study
    print("\n[2/2] Running LSF ablation study...")
    try:
        from exp_4_2_3b_lsf_ablation import main as lsf_ablation_main
        lsf_ablation_main()
    except Exception as e:
        print(f"  Warning: {e}")
    
    print("\n" + "=" * 60)
    print("Experiment 4.2.2 Complete")
    print("=" * 60)

if __name__ == "__main__":
    run_all()
