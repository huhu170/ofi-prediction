# -*- coding: utf-8 -*-
"""
Experiment 4.4.3: Heterogeneity Test (Merged Script)

Merged experiments:
- exp_4_4_2_regime_split.py: Market state grouping
- exp_4_4_2a_event_study.py: Financial event case study
- exp_4_4_3_asset_split.py: Asset type grouping

Paper outputs:
- Table 4.4-2: Heterogeneity test
- Figure 4.4-4: Heterogeneity test visualization

Output files:
- tables/table_4.4-2_heterogeneity.csv
- figures/fig_4.4-4_heterogeneity.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from exp_config import *

def run_all():
    """Run all sub-experiments"""
    print("=" * 60)
    print("Experiment 4.4.3: Heterogeneity Test")
    print("=" * 60)
    
    # 1. Market state grouping
    print("\n[1/3] Running market state grouping test...")
    try:
        from exp_4_4_2_regime_split import main as regime_main
        regime_main()
    except Exception as e:
        print(f"  Warning: {e}")
    
    # 2. Financial event case study
    print("\n[2/3] Running financial event case study...")
    try:
        from exp_4_4_2a_event_study import main as event_main
        event_main()
    except Exception as e:
        print(f"  Warning: {e}")
    
    # 3. Asset type grouping
    print("\n[3/3] Running asset type grouping test...")
    try:
        from exp_4_4_3_asset_split import main as asset_main
        asset_main()
    except Exception as e:
        print(f"  Warning: {e}")
    
    print("\n" + "=" * 60)
    print("Experiment 4.4.3 Complete")
    print("=" * 60)

if __name__ == "__main__":
    run_all()
