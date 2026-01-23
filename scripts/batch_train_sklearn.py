"""批量训练 XGBoost 和 RandomForest 模型"""
import subprocess
import sys
from pathlib import Path

# 股票列表（排除已有的 HK.00700）
STOCKS = [
    "HK.00005", "HK.00388", "HK.00939", "HK.00941", 
    "HK.01211", "HK.01299", "HK.01810", "HK.03690", 
    "HK.09988", "HK.800000"
]

MODELS = ["xgboost", "random_forest"]

# 模型目录
MODEL_DIR = Path(__file__).parent.parent / "models"

def main():
    script_dir = Path(__file__).parent
    trainer_script = script_dir / "13b_kline_model_trainer.py"
    
    total = len(STOCKS) * len(MODELS)
    count = 0
    skipped = 0
    
    for stock in STOCKS:
        for model in MODELS:
            count += 1
            
            # 检查模型是否已存在
            code_str = stock.replace('.', '_')
            model_path = MODEL_DIR / model / f"model_{code_str}_1M.pkl"
            
            if model_path.exists():
                print(f"[{count}/{total}] SKIP {model} for {stock} (already exists)", flush=True)
                skipped += 1
                continue
            
            print(f"\n[{count}/{total}] Training {model} for {stock}...", flush=True)
            
            result = subprocess.run(
                [sys.executable, str(trainer_script), "--code", stock, "--model", model],
                cwd=script_dir.parent
            )
            
            if result.returncode == 0:
                print(f"  [OK] {model} for {stock} completed", flush=True)
            else:
                print(f"  [ERROR] {model} for {stock} failed", flush=True)
    
    print(f"\n[DONE] Batch training completed! (Skipped: {skipped})", flush=True)

if __name__ == "__main__":
    main()
