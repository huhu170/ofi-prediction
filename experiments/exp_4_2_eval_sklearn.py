# -*- coding: utf-8 -*-
"""
快速评估sklearn模型
直接从数据库获取数据并评估已训练的sklearn模型
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from exp_config import *
import pandas as pd
import numpy as np
import pickle

# 导入sklearn模型包装器
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "kline_model_trainer", 
        PROJECT_ROOT / "scripts" / "13b_kline_model_trainer.py"
    )
    trainer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer_module)
    SklearnModelWrapper = trainer_module.SklearnModelWrapper
    
    # 注入到__main__命名空间
    import __main__
    __main__.SklearnModelWrapper = SklearnModelWrapper
    print("[OK] SklearnModelWrapper imported")
except Exception as e:
    print(f"[ERROR] Could not import SklearnModelWrapper: {e}")
    sys.exit(1)

# sklearn评估
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def load_sklearn_model(model_path: Path):
    """加载sklearn模型"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def generate_test_data_from_db(code: str, n_samples: int = 5000) -> tuple:
    """从数据库生成测试数据"""
    try:
        import psycopg2
        import os
        from dotenv import load_dotenv
        
        env_path = PROJECT_ROOT / ".apikey.env"
        load_dotenv(env_path, override=True)
        
        conn = psycopg2.connect(
            host="127.0.0.1",
            port=int(os.getenv("DB_PORT", "5433")),
            database=os.getenv("DB_NAME", "futu_ofi"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "ofi123456")
        )
        
        # 查询最近的K线数据（表名是kline，列名有_price后缀）
        query = f"""
        SELECT ts as time_key, open_price as open, high_price as high, 
               low_price as low, close_price as close, volume
        FROM kline
        WHERE code = '{code}' AND ktype = 'K_1M'
        ORDER BY ts DESC
        LIMIT {n_samples + 100}
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        if df.empty:
            return None, None
        
        df = df.sort_values('time_key').reset_index(drop=True)
        
        # 简单特征计算
        df['return_1'] = df['close'].pct_change()
        df['return_5'] = df['close'].pct_change(5)
        df['return_20'] = df['close'].pct_change(20)
        df['return_60'] = df['close'].pct_change(60)
        df['kline_position'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        df['range_pct'] = (df['high'] - df['low']) / df['open']
        df['relative_volume'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_change'] = df['volume'].pct_change()
        df['rsi'] = 50  # 简化
        df['bb_position'] = 0  # 简化
        df['atr_pct'] = df['range_pct'].rolling(14).mean()
        df['volatility_20'] = df['return_1'].rolling(20).std()
        df['ti'] = df['kline_position'] * df['volume']
        df['ti_5'] = df['ti'].rolling(5).sum()
        df['ti_60'] = df['ti'].rolling(60).sum()
        df['ti_zscore'] = (df['ti'] - df['ti'].rolling(10).mean()) / (df['ti'].rolling(10).std() + 1e-8)
        df['return_zscore'] = (df['return_1'] - df['return_1'].rolling(10).mean()) / (df['return_1'].rolling(10).std() + 1e-8)
        df['pv_corr'] = df['return_1'].rolling(20).corr(df['volume'])
        df['macd_dif'] = 0
        df['macd_dea'] = 0
        df['macd'] = 0
        df['market_regime'] = 1
        
        # 标签
        df['future_return'] = df['close'].shift(-5) / df['close'] - 1
        df['label'] = np.where(df['future_return'] > 0.002, 2, 
                              np.where(df['future_return'] < -0.002, 0, 1))
        
        df = df.dropna()
        
        # 特征列
        feature_cols = ['kline_position', 'range_pct', 'return_1', 'return_5', 'return_20', 
                       'return_60', 'return_zscore', 'atr_pct', 'volatility_20', 'ti', 
                       'ti_5', 'ti_60', 'ti_zscore', 'relative_volume', 'volume_change',
                       'pv_corr', 'rsi', 'bb_position', 'macd_dif', 'macd_dea', 'macd', 'market_regime']
        
        X = df[feature_cols].values
        y = df['label'].values
        
        # 处理inf和nan
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 创建滑动窗口
        seq_len = 60
        X_seq = []
        y_seq = []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i+seq_len])
            y_seq.append(y[i+seq_len-1])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # 取最后20%作为测试
        test_start = int(len(X_seq) * 0.8)
        X_test = X_seq[test_start:]
        y_test = y_seq[test_start:]
        
        return X_test, y_test
        
    except Exception as e:
        print(f"  [ERROR] Database connection failed: {e}")
        return None, None


def evaluate_sklearn_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """评估sklearn模型"""
    # 展平 (N, T, F) -> (N, T*F)
    if X_test.ndim == 3:
        X_flat = X_test.reshape(X_test.shape[0], -1)
    else:
        X_flat = X_test
    
    # 处理inf和nan值
    X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)
    
    # clip极端值到合理范围
    X_flat = np.clip(X_flat, -1e6, 1e6)
    
    y_pred = model.predict(X_flat)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
    }
    
    try:
        y_proba = model.predict_proba(X_flat)
        metrics['auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr')
    except:
        metrics['auc'] = 0.5
    
    return metrics


def main():
    print("=" * 70)
    print("  评估sklearn模型 (LogisticRegression / RandomForest / XGBoost)")
    print("=" * 70)
    
    sklearn_models = ['logistic_regression', 'random_forest', 'xgboost']
    model_names = {'logistic_regression': 'LogisticRegression', 
                   'random_forest': 'RandomForest', 
                   'xgboost': 'XGBoost'}
    
    results = []
    
    for code, name, sector in STOCK_LIST:
        code_str = code.replace('.', '_')
        print(f"\n{code} {name}...")
        
        # 生成测试数据
        X_test, y_test = generate_test_data_from_db(code)
        
        if X_test is None:
            print(f"  [SKIP] No data available")
            continue
        
        print(f"  Test samples: {len(X_test)}")
        
        for model_type in sklearn_models:
            model_path = MODELS_DIR / model_type / f"model_{code_str}_1M.pkl"
            
            if not model_path.exists():
                continue
            
            try:
                model = load_sklearn_model(model_path)
                metrics = evaluate_sklearn_model(model, X_test, y_test)
                
                results.append({
                    '股票代码': code,
                    '股票名称': name,
                    '行业': sector,
                    '模型': model_names[model_type],
                    'Accuracy': metrics['accuracy'],
                    'F1-macro': metrics['f1_macro'],
                    'F1-weighted': metrics['f1_weighted'],
                    'AUC': metrics['auc'],
                })
                
                print(f"  [OK] {model_type}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
                
            except Exception as e:
                print(f"  [ERROR] {model_type}: {e}")
    
    if results:
        df = pd.DataFrame(results)
        
        # 保存
        output_path = TABLES_DIR / 'table_4_2_sklearn_eval.csv'
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存: {output_path}")
        
        # 汇总
        print("\n" + "=" * 70)
        print("  sklearn模型性能汇总")
        print("=" * 70)
        summary = df.groupby('模型')[['Accuracy', 'F1-macro', 'AUC']].agg(['mean', 'std', 'count'])
        print(summary.round(4).to_string())
        
        return df
    else:
        print("\n[ERROR] No results generated. Check database connection.")
        return None


if __name__ == "__main__":
    main()
