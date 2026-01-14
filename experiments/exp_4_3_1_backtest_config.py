#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 4.3.1: 回测参数配置
========================
对应论文: 表4-11 回测参数设置

输出:
- tables/table_4_11_backtest_config.csv
"""

import pandas as pd
from exp_config import (
    save_table, print_section
)

def document_backtest_config():
    """记录回测参数配置"""
    print_section("实验 4.3.1: 回测参数配置")
    
    print("[1] 定义回测参数...")
    
    # 回测参数配置
    config = [
        {'参数类别': '初始资金', '参数名': 'initial_capital', '取值': '1,000,000 HKD', '说明': '初始资金100万港币'},
        {'参数类别': '交易成本', '参数名': 'commission_rate', '取值': '0.03%', '说明': '单边佣金率'},
        {'参数类别': '交易成本', '参数名': 'slippage_bps', '取值': '1 bps', '说明': '滑点（基点）'},
        {'参数类别': '交易成本', '参数名': 'total_cost', '取值': '~0.05%', '说明': '综合交易成本'},
        {'参数类别': '仓位管理', '参数名': 'position_size', '取值': '30%', '说明': '单次交易仓位'},
        {'参数类别': '仓位管理', '参数名': 'max_position', '取值': '100%', '说明': '最大持仓比例'},
        {'参数类别': '风控参数', '参数名': 'stop_loss_pct', '取值': '2%', '说明': '止损比例'},
        {'参数类别': '风控参数', '参数名': 'take_profit_pct', '取值': '5%', '说明': '止盈比例'},
        {'参数类别': '信号阈值', '参数名': 'min_confidence', '取值': '50%', '说明': '最小预测置信度'},
        {'参数类别': '信号阈值', '参数名': 'min_trade_interval', '取值': '60秒', '说明': '最小交易间隔'},
        {'参数类别': '执行延迟', '参数名': 'execution_delay', '取值': '300ms', '说明': 'API延迟+计算延迟'},
        {'参数类别': '回测周期', '参数名': 'test_period', '取值': '5天', '说明': '样本外测试期'},
    ]
    
    result_df = pd.DataFrame(config)
    
    print("\n[2] 回测参数汇总:")
    print(result_df.to_string(index=False))
    
    print("\n[3] 参数选择依据:")
    print("  - 交易成本: 参考美股大盘股实际成本")
    print("  - 仓位管理: 单次30%避免过度集中")
    print("  - 止损止盈: 参考高频交易常见设置")
    print("  - 执行延迟: API推送50-200ms + 特征计算100-300ms")
    
    save_table(result_df, 'table_4_11_backtest_config')
    
    return result_df

if __name__ == '__main__':
    result = document_backtest_config()
    print("\n✓ 实验 4.3.1 完成")
