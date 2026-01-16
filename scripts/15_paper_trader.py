"""
模拟交易模块
基于模型预测信号进行自动化模拟交易

功能:
1. 加载训练好的模型
2. 实时获取行情数据
3. 计算特征 + 模型预测
4. 根据信号进行模拟下单
5. 风控管理（止损、仓位限制）
6. 在富途PC端查看交易记录

使用方法:
    python 15_paper_trader.py --code HK.00700 --model models/transformer/model.pt
    python 15_paper_trader.py --code HK.00700 --dry-run  # 只预测不下单
"""

import os
import sys
import io
import time
import json
import argparse
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from collections import deque
import threading

# 解决Windows编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 加载环境变量
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".apikey.env"
load_dotenv(env_path, override=True)

import numpy as np
import pandas as pd

# 富途API
from futu import (
    OpenQuoteContext, OpenSecTradeContext,
    RET_OK, RET_ERROR,
    TrdMarket, TrdEnv, TrdSide, OrderType,
    SubType, KLType
)

# PyTorch
import torch
import torch.nn.functional as F

# ============================================================
# 配置
# ============================================================

# 富途API配置
FUTU_HOST = '127.0.0.1'
FUTU_PORT = 11111

# 交易配置
TRADE_CONFIG = {
    'market': TrdMarket.HK,           # 港股市场
    'env': TrdEnv.SIMULATE,           # 模拟环境
    'order_type': OrderType.NORMAL,   # 普通订单
    'max_position_pct': 0.3,          # 单只股票最大仓位30%
    'stop_loss_pct': 0.02,            # 止损比例2%
    'take_profit_pct': 0.05,          # 止盈比例5%
    'min_confidence': 0.6,            # 最小预测置信度
    'cooldown_seconds': 60,           # 同方向交易冷却时间
}

# 特征配置（与11_feature_calculator.py对齐）
FEATURE_COLS = [
    'spread_bps', 'return_pct',
    'ofi_l1', 'ofi_l5', 'ofi_l10', 'smart_ofi',
    'ofi_ma_10', 'ofi_std_10', 'ofi_zscore',
    'smart_ofi_ma_10', 'smart_ofi_std_10', 'smart_ofi_zscore',
    'return_ma_10', 'return_std_10',
    'bid_depth_5', 'ask_depth_5', 'depth_imbalance_5',
    'bid_depth_10', 'ask_depth_10', 'depth_imbalance_10',
    'buy_volume', 'sell_volume', 'trade_count', 'trade_imbalance',
    'corr_stock_index'
]

# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据路径
MODEL_DIR = Path("models")
DATA_PROCESSED = Path(os.getenv("DATA_PROCESSED", "data/processed"))


# ============================================================
# 模型加载
# ============================================================

def load_model(model_path: Path, input_dim: int = 25, seq_len: int = 100):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        input_dim: 输入特征维度
        seq_len: 序列长度
        
    Returns:
        加载好的模型
    """
    # 动态导入模型类（使用 importlib 处理带数字前缀的文件名）
    import importlib.util
    model_trainer_path = Path(__file__).parent / "13_model_trainer.py"
    spec = importlib.util.spec_from_file_location("model_trainer", model_trainer_path)
    model_trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_trainer)
    create_model = model_trainer.create_model
    
    # 从路径推断模型类型
    model_name = model_path.parent.name
    if model_name not in ['lstm', 'gru', 'deeplob', 'transformer', 'smart_trans']:
        model_name = 'transformer'  # 默认
    
    # 创建模型
    model = create_model(model_name, input_dim, seq_len)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print(f"  模型加载成功: {model_name}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def load_scaler(scaler_path: Path):
    """加载标准化器"""
    with open(scaler_path, 'rb') as f:
        params = pickle.load(f)
    return params


# ============================================================
# 实时特征计算
# ============================================================

class RealtimeFeatureCalculator:
    """
    实时特征计算器
    
    维护一个滑动窗口的特征历史，用于模型预测
    """
    
    def __init__(self, seq_len: int = 100, scaler_params: dict = None):
        self.seq_len = seq_len
        self.scaler_params = scaler_params
        
        # 滑动窗口缓冲区
        self.feature_buffer = deque(maxlen=seq_len)
        self.orderbook_buffer = deque(maxlen=20)  # 用于计算OFI
        
        # 上一次的订单簿数据（用于计算差分）
        self.last_orderbook = None
        
        # 构建特征名到索引的映射（避免硬编码）
        self.feature_idx = {col: i for i, col in enumerate(FEATURE_COLS)}
        
    def update_orderbook(self, orderbook: dict):
        """
        更新订单簿数据并计算特征
        
        Args:
            orderbook: 订单簿数据字典
        """
        self.orderbook_buffer.append(orderbook)
        
        # 计算特征
        features = self._compute_features(orderbook)
        
        if features is not None:
            self.feature_buffer.append(features)
        
        self.last_orderbook = orderbook
    
    def _compute_features(self, orderbook: dict) -> Optional[np.ndarray]:
        """计算单个时间点的特征"""
        if self.last_orderbook is None:
            return None
        
        features = {}
        
        # 价格特征
        bid1 = orderbook.get('bid1_price', 0)
        ask1 = orderbook.get('ask1_price', 0)
        mid_price = (bid1 + ask1) / 2 if bid1 > 0 and ask1 > 0 else 0
        spread = ask1 - bid1
        
        last_mid = (self.last_orderbook.get('bid1_price', 0) + 
                    self.last_orderbook.get('ask1_price', 0)) / 2
        
        features['spread_bps'] = (spread / mid_price * 10000) if mid_price > 0 else 0
        features['return_pct'] = ((mid_price - last_mid) / last_mid * 100) if last_mid > 0 else 0
        
        # OFI特征
        delta_bid1 = orderbook.get('bid1_vol', 0) - self.last_orderbook.get('bid1_vol', 0)
        delta_ask1 = orderbook.get('ask1_vol', 0) - self.last_orderbook.get('ask1_vol', 0)
        
        features['ofi_l1'] = delta_bid1 - delta_ask1
        
        # 多档OFI（简化版）
        ofi_l5 = 0
        weights = [0.394, 0.239, 0.145, 0.088, 0.053]
        for i, w in enumerate(weights, 1):
            delta_bid = orderbook.get(f'bid{i}_vol', 0) - self.last_orderbook.get(f'bid{i}_vol', 0)
            delta_ask = orderbook.get(f'ask{i}_vol', 0) - self.last_orderbook.get(f'ask{i}_vol', 0)
            ofi_l5 += w * (delta_bid - delta_ask)
        
        features['ofi_l5'] = ofi_l5
        features['ofi_l10'] = ofi_l5  # 简化：只有5档数据
        features['smart_ofi'] = ofi_l5  # 简化
        
        # 滚动统计（使用缓冲区历史，通过特征名索引避免硬编码）
        if len(self.feature_buffer) >= 10:
            idx_ofi = self.feature_idx['ofi_l1']
            idx_smart = self.feature_idx['smart_ofi']
            idx_ret = self.feature_idx['return_pct']
            
            recent_ofi = [f[idx_ofi] for f in list(self.feature_buffer)[-10:]]
            features['ofi_ma_10'] = np.mean(recent_ofi)
            features['ofi_std_10'] = np.std(recent_ofi) + 1e-8
            features['ofi_zscore'] = (features['ofi_l1'] - features['ofi_ma_10']) / features['ofi_std_10']
            
            recent_smart = [f[idx_smart] for f in list(self.feature_buffer)[-10:]]
            features['smart_ofi_ma_10'] = np.mean(recent_smart)
            features['smart_ofi_std_10'] = np.std(recent_smart) + 1e-8
            features['smart_ofi_zscore'] = (features['smart_ofi'] - features['smart_ofi_ma_10']) / features['smart_ofi_std_10']
            
            recent_ret = [f[idx_ret] for f in list(self.feature_buffer)[-10:]]
            features['return_ma_10'] = np.mean(recent_ret)
            features['return_std_10'] = np.std(recent_ret) + 1e-8
        else:
            features['ofi_ma_10'] = 0
            features['ofi_std_10'] = 1
            features['ofi_zscore'] = 0
            features['smart_ofi_ma_10'] = 0
            features['smart_ofi_std_10'] = 1
            features['smart_ofi_zscore'] = 0
            features['return_ma_10'] = 0
            features['return_std_10'] = 1
        
        # 深度特征
        bid_depth_5 = sum(orderbook.get(f'bid{i}_vol', 0) for i in range(1, 6))
        ask_depth_5 = sum(orderbook.get(f'ask{i}_vol', 0) for i in range(1, 6))
        
        features['bid_depth_5'] = bid_depth_5
        features['ask_depth_5'] = ask_depth_5
        features['depth_imbalance_5'] = (bid_depth_5 - ask_depth_5) / (bid_depth_5 + ask_depth_5 + 1)
        
        features['bid_depth_10'] = bid_depth_5  # 简化
        features['ask_depth_10'] = ask_depth_5
        features['depth_imbalance_10'] = features['depth_imbalance_5']
        
        # 成交特征（需要ticker数据，这里简化）
        features['buy_volume'] = 0
        features['sell_volume'] = 0
        features['trade_count'] = 0
        features['trade_imbalance'] = 0
        
        # 协方差（简化为0）
        features['corr_stock_index'] = 0
        
        # 转为数组
        feature_array = np.array([features.get(col, 0) for col in FEATURE_COLS], dtype=np.float32)
        
        return feature_array
    
    def get_sequence(self) -> Optional[np.ndarray]:
        """
        获取模型输入序列
        
        Returns:
            (seq_len, num_features) 的数组，或None（数据不足）
        """
        if len(self.feature_buffer) < self.seq_len:
            return None
        
        sequence = np.array(list(self.feature_buffer), dtype=np.float32)
        
        # 标准化
        if self.scaler_params is not None:
            mean = self.scaler_params['mean']
            std = self.scaler_params['std']
            sequence = (sequence - mean) / std
        
        return sequence
    
    def is_ready(self) -> bool:
        """检查是否有足够的数据"""
        return len(self.feature_buffer) >= self.seq_len


# ============================================================
# 交易信号生成
# ============================================================

class SignalGenerator:
    """
    交易信号生成器
    
    信号类型:
    - BUY: 买入信号（预测上涨）
    - SELL: 卖出信号（预测下跌）
    - HOLD: 持有信号（预测平稳或置信度不足）
    """
    
    def __init__(self, model, min_confidence: float = 0.6):
        self.model = model
        self.min_confidence = min_confidence
        
        # 标签映射: 0=下跌, 1=平稳, 2=上涨
        self.label_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    
    def generate_signal(self, sequence: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        生成交易信号
        
        Args:
            sequence: 输入序列 (seq_len, features)
            
        Returns:
            signal: 'BUY', 'SELL', 'HOLD'
            confidence: 预测置信度
            probs: 各类别概率
        """
        # 转为tensor
        x = torch.FloatTensor(sequence).unsqueeze(0).to(DEVICE)  # (1, seq_len, features)
        
        # 预测
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        # 获取预测类别和置信度
        pred_class = probs.argmax()
        confidence = probs[pred_class]
        
        # 生成信号
        if confidence < self.min_confidence:
            signal = 'HOLD'
        else:
            signal = self.label_map[pred_class]
        
        return signal, confidence, probs


# ============================================================
# 模拟交易执行
# ============================================================

class PaperTrader:
    """
    模拟交易执行器
    
    功能:
    - 连接富途模拟账户
    - 执行买入/卖出订单
    - 仓位管理
    - 风控检查
    """
    
    def __init__(self, config: dict = None):
        self.config = config or TRADE_CONFIG
        
        self.quote_ctx = None
        self.trade_ctx = None
        
        # 账户信息
        self.account_id = None
        self.cash = 0
        self.positions = {}  # {code: {'qty': int, 'avg_price': float, 'market_value': float}}
        
        # 交易记录
        self.trade_history = []
        self.last_trade_time = {}  # {code: datetime} 冷却时间控制
        
        # 状态
        self.is_connected = False
    
    def connect(self) -> bool:
        """连接富途API"""
        try:
            # 行情连接
            self.quote_ctx = OpenQuoteContext(host=FUTU_HOST, port=FUTU_PORT)
            
            # 交易连接
            self.trade_ctx = OpenSecTradeContext(
                host=FUTU_HOST, 
                port=FUTU_PORT,
                filter_trdmarket=self.config['market']
            )
            
            # 解锁交易（模拟账户不需要密码）
            if self.config['env'] == TrdEnv.SIMULATE:
                ret, data = self.trade_ctx.unlock_trade(password="")
                if ret != RET_OK:
                    print(f"  [WARN] 解锁交易失败: {data}")
            
            # 获取账户列表
            ret, data = self.trade_ctx.get_acc_list()
            if ret == RET_OK:
                # 筛选模拟账户
                sim_accounts = data[data['trd_env'] == 'SIMULATE']
                if not sim_accounts.empty:
                    self.account_id = sim_accounts.iloc[0]['acc_id']
                    print(f"  模拟账户ID: {self.account_id}")
            
            self.is_connected = True
            print("  [OK] 富途API连接成功")
            
            # 更新账户信息
            self._update_account_info()
            
            return True
            
        except Exception as e:
            print(f"  [ERROR] 连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        if self.quote_ctx:
            self.quote_ctx.close()
        if self.trade_ctx:
            self.trade_ctx.close()
        self.is_connected = False
        print("  [OK] 连接已断开")
    
    def _update_account_info(self):
        """更新账户信息"""
        if not self.trade_ctx:
            return
        
        # 获取账户资金
        ret, data = self.trade_ctx.accinfo_query(trd_env=self.config['env'])
        if ret == RET_OK and not data.empty:
            self.cash = data.iloc[0]['cash']
            print(f"  可用资金: ${self.cash:,.2f}")
        
        # 获取持仓
        ret, data = self.trade_ctx.position_list_query(trd_env=self.config['env'])
        if ret == RET_OK and not data.empty:
            for _, row in data.iterrows():
                self.positions[row['code']] = {
                    'qty': row['qty'],
                    'avg_price': row['cost_price'],
                    'market_value': row['market_val']
                }
            print(f"  持仓数量: {len(self.positions)} 只")
    
    def get_quote(self, code: str) -> Optional[dict]:
        """获取实时报价"""
        if not self.quote_ctx:
            return None
        
        try:
            ret, data = self.quote_ctx.get_market_snapshot([code])
            if ret == RET_OK and not data.empty:
                row = data.iloc[0]
                return {
                    'code': code,
                    'last_price': row['last_price'],
                    'bid_price': row['bid_price'],
                    'ask_price': row['ask_price'],
                    'volume': row['volume'],
                    'turnover': row['turnover']
                }
        except Exception as e:
            print(f"  [ERROR] 获取报价失败: {e}")
        return None
    
    def get_orderbook(self, code: str) -> Optional[dict]:
        """获取订单簿"""
        if not self.quote_ctx:
            return None
        
        try:
            ret, data = self.quote_ctx.get_order_book(code)
            if ret == RET_OK:
                orderbook = {'code': code}
                
                # 解析买盘（富途API返回列表，每个元素是 [price, volume, order_count]）
                bid_data = data.get('Bid', [])
                for i, item in enumerate(bid_data[:10], 1):
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        orderbook[f'bid{i}_price'] = item[0]
                        orderbook[f'bid{i}_vol'] = item[1]
                    else:
                        orderbook[f'bid{i}_price'] = 0
                        orderbook[f'bid{i}_vol'] = 0
                
                # 解析卖盘
                ask_data = data.get('Ask', [])
                for i, item in enumerate(ask_data[:10], 1):
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        orderbook[f'ask{i}_price'] = item[0]
                        orderbook[f'ask{i}_vol'] = item[1]
                    else:
                        orderbook[f'ask{i}_price'] = 0
                        orderbook[f'ask{i}_vol'] = 0
                
                return orderbook
        except Exception as e:
            print(f"  [ERROR] 获取订单簿失败: {e}")
        return None
    
    def check_risk(self, code: str, signal: str, price: float) -> Tuple[bool, str]:
        """
        风控检查
        
        Args:
            code: 股票代码
            signal: 交易信号
            price: 当前价格
            
        Returns:
            (是否通过, 原因)
        """
        # 冷却时间检查
        if code in self.last_trade_time:
            elapsed = (datetime.now() - self.last_trade_time[code]).total_seconds()
            if elapsed < self.config['cooldown_seconds']:
                return False, f"冷却中 ({int(self.config['cooldown_seconds'] - elapsed)}s)"
        
        # 仓位检查
        if signal == 'BUY':
            # 检查是否已满仓
            total_value = self.cash + sum(p['market_value'] for p in self.positions.values())
            if code in self.positions:
                current_pct = self.positions[code]['market_value'] / total_value
                if current_pct >= self.config['max_position_pct']:
                    return False, f"已达最大仓位 ({current_pct*100:.1f}%)"
        
        elif signal == 'SELL':
            # 检查是否有持仓
            if code not in self.positions or self.positions[code]['qty'] <= 0:
                return False, "无持仓可卖"
        
        return True, "通过"
    
    def check_stop_loss_take_profit(self, code: str, price: float) -> Optional[str]:
        """
        检查止损止盈条件
        
        Args:
            code: 股票代码
            price: 当前价格
            
        Returns:
            触发的条件类型 ('STOP_LOSS', 'TAKE_PROFIT') 或 None
        """
        if code not in self.positions or self.positions[code]['qty'] <= 0:
            return None
        
        avg_price = self.positions[code]['avg_price']
        if avg_price <= 0:
            return None
            
        pnl_pct = (price - avg_price) / avg_price
        
        if pnl_pct <= -self.config['stop_loss_pct']:
            return 'STOP_LOSS'
        if pnl_pct >= self.config['take_profit_pct']:
            return 'TAKE_PROFIT'
        
        return None
    
    def calculate_qty(self, code: str, signal: str, price: float) -> int:
        """计算下单数量"""
        if signal == 'BUY':
            # 计算可买数量
            total_value = self.cash + sum(p['market_value'] for p in self.positions.values())
            max_value = total_value * self.config['max_position_pct']
            
            current_value = self.positions.get(code, {}).get('market_value', 0)
            available_value = max_value - current_value
            
            # 港股一手100股
            lot_size = 100
            qty = int(available_value / price / lot_size) * lot_size
            
            return max(0, qty)
        
        elif signal == 'SELL':
            # 卖出全部持仓
            return self.positions.get(code, {}).get('qty', 0)
        
        return 0
    
    def place_order(self, code: str, signal: str, qty: int, price: float) -> Optional[str]:
        """
        下单
        
        Args:
            code: 股票代码
            signal: 'BUY' 或 'SELL'
            qty: 数量
            price: 价格
            
        Returns:
            订单ID，或None（失败）
        """
        if not self.trade_ctx or qty <= 0:
            return None
        
        side = TrdSide.BUY if signal == 'BUY' else TrdSide.SELL
        
        ret, data = self.trade_ctx.place_order(
            price=price,
            qty=qty,
            code=code,
            trd_side=side,
            order_type=self.config['order_type'],
            trd_env=self.config['env']
        )
        
        if ret == RET_OK:
            order_id = data.iloc[0]['order_id']
            
            # 记录交易
            trade_record = {
                'time': datetime.now().isoformat(),
                'code': code,
                'signal': signal,
                'qty': qty,
                'price': price,
                'order_id': order_id
            }
            self.trade_history.append(trade_record)
            self.last_trade_time[code] = datetime.now()
            
            print(f"  [ORDER] {signal} {code} x {qty} @ ${price:.2f} (ID: {order_id})")
            
            return order_id
        else:
            print(f"  [ERROR] 下单失败: {data}")
            return None
    
    def execute_signal(self, code: str, signal: str, confidence: float, dry_run: bool = False) -> bool:
        """
        执行交易信号
        
        Args:
            code: 股票代码
            signal: 交易信号
            confidence: 置信度
            dry_run: 是否只模拟（不实际下单）
            
        Returns:
            是否执行成功
        """
        if signal == 'HOLD':
            return False
        
        # 获取当前价格
        quote = self.get_quote(code)
        if not quote:
            print(f"  [WARN] 无法获取报价")
            return False
        
        price = quote['ask_price'] if signal == 'BUY' else quote['bid_price']
        
        # 风控检查
        passed, reason = self.check_risk(code, signal, price)
        if not passed:
            print(f"  [RISK] {reason}")
            return False
        
        # 计算数量
        qty = self.calculate_qty(code, signal, price)
        if qty <= 0:
            print(f"  [WARN] 计算数量为0")
            return False
        
        # 执行下单
        if dry_run:
            print(f"  [DRY-RUN] {signal} {code} x {qty} @ ${price:.2f} (置信度: {confidence:.2%})")
            return True
        else:
            order_id = self.place_order(code, signal, qty, price)
            return order_id is not None


# ============================================================
# 主交易循环
# ============================================================

class TradingEngine:
    """
    交易引擎
    
    整合数据获取、特征计算、信号生成、交易执行
    """
    
    def __init__(
        self,
        code: str,
        model_path: Path,
        scaler_path: Path = None,
        dry_run: bool = False
    ):
        self.code = code
        self.dry_run = dry_run
        
        # 加载模型
        print("\n[1] 加载模型...")
        self.model = load_model(model_path)
        
        # 加载标准化器
        print("\n[2] 加载标准化器...")
        scaler_params = None
        if scaler_path and scaler_path.exists():
            scaler_params = load_scaler(scaler_path)
            print(f"  标准化器加载成功")
        else:
            print(f"  [WARN] 未找到标准化器，使用原始特征")
        
        # 初始化组件
        self.feature_calc = RealtimeFeatureCalculator(seq_len=100, scaler_params=scaler_params)
        self.signal_gen = SignalGenerator(self.model, min_confidence=TRADE_CONFIG['min_confidence'])
        self.trader = PaperTrader()
        
        # 状态
        self.is_running = False
        self.stats = {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'executed_trades': 0
        }
    
    def start(self):
        """启动交易引擎"""
        print("\n[3] 连接富途API...")
        if not self.trader.connect():
            return
        
        # 订阅行情
        print("\n[4] 订阅行情...")
        ret, data = self.trader.quote_ctx.subscribe([self.code], [SubType.ORDER_BOOK])
        if ret != RET_OK:
            print(f"  [ERROR] 订阅失败: {data}")
            return
        print(f"  订阅成功: {self.code}")
        
        self.is_running = True
        print("\n[5] 开始交易...")
        print("="*60)
        print(f"  股票: {self.code}")
        print(f"  模式: {'模拟运行' if self.dry_run else '实盘模拟'}")
        print(f"  最小置信度: {TRADE_CONFIG['min_confidence']:.0%}")
        print("="*60)
        print("\n  等待数据填充缓冲区...")
        
        try:
            self._trading_loop()
        except KeyboardInterrupt:
            print("\n\n  [INFO] 用户中断")
        finally:
            self.stop()
    
    def _trading_loop(self):
        """交易主循环"""
        update_count = 0
        
        while self.is_running:
            try:
                # 获取订单簿
                orderbook = self.trader.get_orderbook(self.code)
                if orderbook is None:
                    time.sleep(1)
                    continue
                
                # 更新特征
                self.feature_calc.update_orderbook(orderbook)
                update_count += 1
                
                # 获取当前价格用于止损止盈检查
                current_price = (orderbook.get('bid1_price', 0) + orderbook.get('ask1_price', 0)) / 2
                
                # 检查止损止盈（优先于模型信号）
                sl_tp_trigger = self.trader.check_stop_loss_take_profit(self.code, current_price)
                if sl_tp_trigger:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    trigger_type = "止损" if sl_tp_trigger == 'STOP_LOSS' else "止盈"
                    print(f"  {timestamp} | [!] 触发{trigger_type}，执行卖出")
                    if self.trader.execute_signal(self.code, 'SELL', 1.0, self.dry_run):
                        self.stats['executed_trades'] += 1
                    time.sleep(10)
                    continue
                
                # 检查是否准备好
                if not self.feature_calc.is_ready():
                    if update_count % 10 == 0:
                        print(f"  缓冲区: {len(self.feature_calc.feature_buffer)}/100")
                    time.sleep(1)
                    continue
                
                # 获取序列
                sequence = self.feature_calc.get_sequence()
                if sequence is None:
                    time.sleep(1)
                    continue
                
                # 生成信号
                signal, confidence, probs = self.signal_gen.generate_signal(sequence)
                self.stats['total_signals'] += 1
                
                # 统计
                if signal == 'BUY':
                    self.stats['buy_signals'] += 1
                elif signal == 'SELL':
                    self.stats['sell_signals'] += 1
                else:
                    self.stats['hold_signals'] += 1
                
                # 打印信号
                timestamp = datetime.now().strftime('%H:%M:%S')
                prob_str = f"[跌:{probs[0]:.1%} 稳:{probs[1]:.1%} 涨:{probs[2]:.1%}]"
                
                if signal != 'HOLD':
                    print(f"  {timestamp} | {signal:4s} | 置信度:{confidence:.1%} | {prob_str}")
                    
                    # 执行信号
                    if self.trader.execute_signal(self.code, signal, confidence, self.dry_run):
                        self.stats['executed_trades'] += 1
                else:
                    # 每10个HOLD信号打印一次
                    if self.stats['hold_signals'] % 10 == 0:
                        print(f"  {timestamp} | HOLD | 置信度:{confidence:.1%} | {prob_str}")
                
                # 等待下一个周期
                time.sleep(10)  # 10秒一次
                
            except Exception as e:
                print(f"  [ERROR] 交易循环异常: {e}")
                time.sleep(5)  # 出错后等待5秒再继续
    
    def stop(self):
        """停止交易引擎"""
        self.is_running = False
        
        print("\n" + "="*60)
        print("  交易统计")
        print("="*60)
        print(f"  总信号数: {self.stats['total_signals']}")
        print(f"  买入信号: {self.stats['buy_signals']}")
        print(f"  卖出信号: {self.stats['sell_signals']}")
        print(f"  持有信号: {self.stats['hold_signals']}")
        print(f"  执行交易: {self.stats['executed_trades']}")
        print("="*60)
        
        # 保存交易记录
        if self.trader.trade_history:
            history_path = Path('trades') / f'history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            history_path.parent.mkdir(exist_ok=True)
            with open(history_path, 'w') as f:
                json.dump(self.trader.trade_history, f, indent=2)
            print(f"  交易记录已保存: {history_path}")
        
        self.trader.disconnect()


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='模拟交易模块')
    parser.add_argument('--code', type=str, default='HK.00700', help='股票代码')
    parser.add_argument('--model', type=str, default='models/transformer/model.pt', help='模型路径')
    parser.add_argument('--scaler', type=str, default=None, help='标准化器路径')
    parser.add_argument('--dry-run', action='store_true', help='模拟运行（不实际下单）')
    parser.add_argument('--confidence', type=float, default=0.6, help='最小置信度')
    
    args = parser.parse_args()
    
    print("="*60)
    print("  OFI论文 - 模拟交易模块")
    print("="*60)
    print(f"  股票代码: {args.code}")
    print(f"  模型路径: {args.model}")
    print(f"  运行模式: {'模拟运行' if args.dry_run else '实盘模拟'}")
    
    # 更新配置
    TRADE_CONFIG['min_confidence'] = args.confidence
    
    # 确定标准化器路径
    model_path = Path(args.model)
    if args.scaler:
        scaler_path = Path(args.scaler)
    else:
        # 尝试从数据集目录查找
        scaler_path = DATA_PROCESSED / 'combined' / 'dataset_T100_k20' / 'scaler.pkl'
    
    # 启动交易引擎
    engine = TradingEngine(
        code=args.code,
        model_path=model_path,
        scaler_path=scaler_path,
        dry_run=args.dry_run
    )
    
    engine.start()


if __name__ == "__main__":
    main()
