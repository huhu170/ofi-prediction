"""
实验 4.3.1: 回测参数配置

对应论文:
- 表 4.3-1: 回测参数设置

输出:
- table_4_3_1_backtest_config.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from exp_config import *
import pandas as pd

def run_experiment():
    """运行实验"""
    log_experiment('4.3.1', '输出回测参数配置')
    
    # 参数配置表
    config_data = [
        ('交易成本', '基准情景', '单边0.05%', '含手续费与估算滑点'),
        ('交易成本', '敏感性分析', '0.03%、0.05%、0.10%', '三档对比验证稳健性'),
        ('交易成本', '压力测试', '0.10%、0.15%', '高成本极端情景'),
        ('仓位策略', '主策略', '信号强度映射', '预测概率>0.6做多，<0.4做空'),
        ('仓位策略', '对比策略', '全仓切换', '作为上界参考'),
        ('滑点模型', '固定滑点', '已含在交易成本中', '基准情景'),
        ('滑点模型', '成交量依赖滑点', '按Kyle模型估算', '敏感性扩展'),
        ('风控参数', '止损阈值', '2%', '单笔最大亏损'),
        ('风控参数', '止盈阈值', '5%', '单笔最大盈利'),
        ('风控参数', '最大仓位', '30%', '单一标的最大持仓'),
        ('其他', '初始资金', '100万港币', '回测起始资金'),
        ('其他', '货币单位', '港币(HKD)', '所有收益与成本'),
    ]
    
    df_results = pd.DataFrame(config_data, columns=['参数类别', '配置项', '取值', '说明'])
    
    # 保存
    output_path = get_output_path('table_4_3_1_backtest_config', 'csv')
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    log_experiment('4.3.1', f'结果已保存: {output_path}')
    
    print("\n" + "="*60)
    print("  表 4.3-1: 回测参数设置")
    print("="*60)
    print(df_results.to_string(index=False))
    
    return df_results


if __name__ == "__main__":
    set_seed()
    run_experiment()
