"""
实验配置文件
与论文第四章实验设计对齐

命名规则: exp_章_节_序号_描述.py
示例: exp_4_1_1_sample_stats.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# ============================================================
# 路径配置
# ============================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据路径
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"

# 输出路径
EXPERIMENT_RESULTS = PROJECT_ROOT / "experiment_results"
FIGURES_DIR = EXPERIMENT_RESULTS / "figures"
TABLES_DIR = EXPERIMENT_RESULTS / "tables"
MODELS_DIR = PROJECT_ROOT / "models"
BACKTEST_RESULTS = PROJECT_ROOT / "backtest_results"

# 确保目录存在
for d in [FIGURES_DIR, TABLES_DIR, MODELS_DIR, BACKTEST_RESULTS]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 数据配置（与论文第三章对齐）
# ============================================================

# 研究样本
STOCK_LIST = [
    ('HK.00700', '腾讯控股', '科技'),
    ('HK.00005', '汇丰控股', '金融'),
    ('HK.09988', '阿里巴巴', '科技'),
    ('HK.01810', '小米集团', '科技'),
    ('HK.00939', '建设银行', '金融'),
    ('HK.01299', '友邦保险', '金融'),
    ('HK.00941', '中国移动', '通信'),
    ('HK.03690', '美团', '科技'),
    ('HK.01211', '比亚迪', '汽车'),
    ('HK.00388', '香港交易所', '金融'),
]

# 恒生指数
HSI_CODE = 'HK.800000'

# 样本期间
SAMPLE_START = '2021-01-01'
SAMPLE_END = '2025-12-31'

# K线类型配置
KLINE_TYPES = ['1M', '5M', '60M', 'DAY']

# ============================================================
# 特征配置（与论文公式对齐）
# ============================================================

# 特征分组（与11b_kline_feature_calculator.py对齐）
FEATURE_GROUPS = {
    '价格动量': ['return_1', 'return_5', 'return_20', 'return_zscore'],
    '波动率': ['atr_pct', 'range_pct', 'volatility_20'],
    '成交不平衡': ['ti', 'ti_5', 'ti_20', 'ti_zscore'],
    '成交量': ['relative_volume', 'volume_change', 'pv_corr'],
    '技术指标': ['rsi', 'bb_position', 'macd_dif', 'macd_dea', 'macd'],
    '市场状态': ['market_regime'],
}

# 全部特征列表
ALL_FEATURES = [f for features in FEATURE_GROUPS.values() for f in features]

# 特征中文名
FEATURE_NAMES_CN = {
    'return_1': '1分钟收益率',
    'return_5': '5分钟收益率',
    'return_20': '20分钟收益率',
    'return_zscore': '收益率Z-score',
    'atr_pct': 'ATR百分比',
    'range_pct': '日内波幅',
    'volatility_20': '20期波动率',
    'ti': '成交不平衡(TI)',
    'ti_5': '5期累积TI',
    'ti_20': '20期累积TI',
    'ti_zscore': 'TI Z-score',
    'relative_volume': '相对成交量',
    'volume_change': '成交量变化',
    'pv_corr': '量价相关性',
    'rsi': 'RSI(14)',
    'bb_position': '布林带位置',
    'macd_dif': 'MACD DIF',
    'macd_dea': 'MACD DEA',
    'macd': 'MACD柱',
    'market_regime': '市场状态',
}

# ============================================================
# 模型配置（与论文第三章第四节对齐）
# ============================================================

# 预测步长（分钟）
PREDICTION_HORIZONS = [5, 15, 30]

# 标签阈值
LABEL_ALPHA = 0.002  # 0.2%

# 基准模型
BASELINE_MODELS = ['ARIMA', 'LogisticRegression', 'RandomForest', 'XGBoost']

# 深度学习模型
DEEP_MODELS = ['LSTM', 'GRU', 'CNN-LSTM', 'Transformer']

# 近年强基线（论文第三章）
SOTA_BASELINES = ['Crossformer', 'Stockformer', 'MBO-Attention', 'LSTM-CNN-CBAM']

# 本研究模型
OUR_MODELS = ['PV-Transformer', 'PV-Transformer+LSF']

# 模型超参数
MODEL_HYPERPARAMS = {
    'pv_transformer': {
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 256,
        'dropout': 0.1,
    },
    'lstm': {
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.2,
    }
}

# 训练超参数
TRAIN_CONFIG = {
    'batch_size': 64,
    'learning_rate': 1e-4,
    'max_epochs': 100,
    'early_stopping_patience': 15,
    'weight_decay': 1e-5,
    'seed': 42,  # 固定随机种子确保可复现
}

# ============================================================
# 回测配置（与论文表4-6对齐）
# ============================================================

BACKTEST_CONFIG = {
    # 交易成本
    'base_cost': 0.0005,          # 基准单边成本 0.05%
    'cost_levels': [0.0003, 0.0005, 0.001, 0.0015],  # 敏感性分析
    
    # 仓位策略
    'position_threshold_long': 0.6,   # 预测概率>0.6做多
    'position_threshold_short': 0.4,  # 预测概率<0.4做空
    
    # 风控
    'stop_loss_pct': 0.02,        # 止损 2%
    'take_profit_pct': 0.05,      # 止盈 5%
    'max_position_pct': 0.3,      # 最大仓位 30%
    
    # 初始资金
    'initial_capital': 1_000_000,  # 港币
}

# ============================================================
# 市场状态定义（与论文公式3.3-1对齐）
# ============================================================

MARKET_REGIME = {
    'low_vol': {'name': '平稳期', 'quantile': 0.50},
    'normal': {'name': '正常期', 'quantile': (0.50, 0.90)},
    'high_vol': {'name': '高波动期', 'quantile': 0.90},
}

# 牛熊市定义（20日收益率）
MARKET_STATE = {
    'bull': {'name': '牛市', 'threshold': 0.05},      # 20日收益>5%
    'bear': {'name': '熊市', 'threshold': -0.05},     # 20日收益<-5%
    'sideways': {'name': '震荡市', 'threshold': None},
}

# ============================================================
# 金融事件（Case Study）
# ============================================================

# 事件类型
EVENT_TYPES = {
    'earnings': '财报发布日',
    'fomc': '美联储议息日',
    'index_rebalance': '恒指成分股调整日',
}

# ============================================================
# 可视化配置
# ============================================================

# matplotlib中文字体
PLOT_CONFIG = {
    'font.sans-serif': ['SimHei', 'Microsoft YaHei'],
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
}

# 图表颜色方案
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#bcbd22',
    'info': '#17becf',
}

# ============================================================
# 辅助函数
# ============================================================

def get_stock_by_sector(sector: str):
    """按行业筛选股票"""
    return [(code, name) for code, name, s in STOCK_LIST if s == sector]

def get_output_path(exp_name: str, ext: str = 'csv') -> Path:
    """获取实验输出路径"""
    if ext in ['png', 'jpg', 'pdf']:
        return FIGURES_DIR / f"{exp_name}.{ext}"
    else:
        return TABLES_DIR / f"{exp_name}.{ext}"

def setup_plot():
    """配置matplotlib"""
    import matplotlib.pyplot as plt
    for key, value in PLOT_CONFIG.items():
        plt.rcParams[key] = value

def set_seed(seed: int = TRAIN_CONFIG['seed']):
    """设置随机种子确保可复现"""
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def log_experiment(exp_name: str, message: str):
    """记录实验日志"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{exp_name}] {message}")


# ============================================================
# 实验脚本列表（与论文第四章对齐）
# ============================================================

EXPERIMENT_SCRIPTS = {
    # 4.1节 数据与特征的统计分析
    '4.1.1': ('exp_4_1_1_sample_stats.py', '样本描述性统计'),
    '4.1.2': ('exp_4_1_2_feature_distribution.py', '特征分布分析'),
    '4.1.3': ('exp_4_1_3_label_balance.py', '标签分布检验'),
    '4.1.4': ('exp_4_1_4_correlation.py', '相关性检验'),
    '4.1.5': ('exp_4_1_5_ols_regression.py', 'OLS回归分析'),
    '4.1.6': ('exp_4_1_6_scale_comparison.py', '多尺度解释力对比'),
    
    # 4.2节 模型性能评估
    '4.2.1': ('exp_4_2_1_baseline_models.py', '基准模型评估'),
    '4.2.2': ('exp_4_2_2_deep_models.py', '深度学习模型评估'),
    '4.2.3a': ('exp_4_2_3a_pv_crossattn_ablation.py', 'PV-CrossAttention消融'),
    '4.2.3b': ('exp_4_2_3b_lsf_ablation.py', 'LSF消融实验'),
    '4.2.4': ('exp_4_2_4_feature_ablation.py', '特征消融实验'),
    '4.2.5': ('exp_4_2_5_threshold_sensitivity.py', '标签阈值敏感性'),
    
    # 4.3节 策略回测
    '4.3.1': ('exp_4_3_1_backtest_config.py', '回测参数配置'),
    '4.3.2': ('exp_4_3_2_backtest.py', '模型回测'),
    '4.3.3': ('exp_4_3_3_scale_comparison.py', '多尺度回测对比'),
    '4.3.4': ('exp_4_3_4_cost_sensitivity.py', '交易成本敏感性'),
    
    # 4.4节 可解释性与稳健性
    '4.4.1': ('exp_4_4_1_shap_analysis.py', 'SHAP归因分析'),
    '4.4.2': ('exp_4_4_2_regime_split.py', '市场状态分组检验'),
    '4.4.2a': ('exp_4_4_2a_event_study.py', '金融事件案例研究'),
    '4.4.3': ('exp_4_4_3_asset_split.py', '资产类型分组检验'),
    '4.4.5': ('exp_4_4_5_granger_causality.py', 'Granger因果检验'),
    '4.4.6': ('exp_4_4_6_causal_feature_comparison.py', '因果特征子集验证'),
    '4.4.7': ('exp_4_4_7_counterfactual.py', '反事实分析'),
    '4.4.8': ('exp_4_4_8_decay_analysis.py', '预测能力衰减分析'),
    '4.4.9': ('exp_4_4_9_market_state.py', '市场状态预测对比'),
    '4.4.10': ('exp_4_4_10_rolling_training.py', '滚动训练有效性'),
    '4.4.11': ('exp_4_4_11_shap_vs_causal.py', 'SHAP与Granger对比'),
}


if __name__ == "__main__":
    print("实验配置加载成功")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"研究样本: {len(STOCK_LIST)} 只股票")
    print(f"实验脚本: {len(EXPERIMENT_SCRIPTS)} 个")
