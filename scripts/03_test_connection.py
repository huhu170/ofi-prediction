"""
数据库连接测试脚本
用于验证 PostgreSQL + TimescaleDB 是否正确安装和配置
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
env_path = Path(__file__).parent.parent / ".apikey.env"
load_dotenv(env_path)

# 从环境变量读取数据库配置
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5433")),
    "database": os.getenv("DB_NAME", "futu_ofi"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "")
}


def test_psycopg2():
    """测试 psycopg2 连接"""
    print("\n" + "="*50)
    print("测试 psycopg2 连接...")
    print("="*50)
    
    try:
        import psycopg2
        
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 测试基本查询
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"✅ PostgreSQL 版本: {version[:50]}...")
        
        # 测试 TimescaleDB
        cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';")
        result = cursor.fetchone()
        if result:
            print(f"✅ TimescaleDB 版本: {result[0]}")
        else:
            print("❌ TimescaleDB 扩展未安装")
        
        # 测试表是否存在
        cursor.execute("""
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public' 
            AND tablename IN ('orderbook', 'ticker', 'quote', 'ofi_features');
        """)
        tables = cursor.fetchall()
        print(f"✅ 已创建的表: {[t[0] for t in tables]}")
        
        # 测试超表
        cursor.execute("""
            SELECT hypertable_name FROM timescaledb_information.hypertables;
        """)
        hypertables = cursor.fetchall()
        print(f"✅ TimescaleDB 超表: {[t[0] for t in hypertables]}")
        
        cursor.close()
        conn.close()
        print("\n✅ psycopg2 连接测试通过！")
        return True
        
    except ImportError:
        print("❌ psycopg2 未安装，请运行: pip install psycopg2-binary")
        return False
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False


def test_sqlalchemy():
    """测试 SQLAlchemy 连接"""
    print("\n" + "="*50)
    print("测试 SQLAlchemy 连接...")
    print("="*50)
    
    try:
        from sqlalchemy import create_engine, text
        
        connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        engine = create_engine(connection_string)
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ SQLAlchemy 连接成功！")
        
        return True
        
    except ImportError:
        print("❌ SQLAlchemy 未安装，请运行: pip install sqlalchemy")
        return False
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False


def test_pandas_read():
    """测试 Pandas 读取"""
    print("\n" + "="*50)
    print("测试 Pandas 读取...")
    print("="*50)
    
    try:
        import pandas as pd
        from sqlalchemy import create_engine
        
        connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        engine = create_engine(connection_string)
        
        # 读取表结构
        df = pd.read_sql("SELECT * FROM orderbook LIMIT 0", engine)
        print(f"✅ orderbook 表列数: {len(df.columns)}")
        print(f"   列名: {list(df.columns)[:5]}... (共{len(df.columns)}列)")
        
        return True
        
    except ImportError:
        print("❌ pandas 未安装，请运行: pip install pandas")
        return False
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return False


def main():
    print("="*50)
    print("  OFI论文 - 数据库连接测试")
    print("="*50)
    print(f"\n数据库配置:")
    print(f"  Host: {DB_CONFIG['host']}")
    print(f"  Port: {DB_CONFIG['port']}")
    print(f"  Database: {DB_CONFIG['database']}")
    print(f"  User: {DB_CONFIG['user']}")
    
    results = []
    results.append(("psycopg2", test_psycopg2()))
    results.append(("SQLAlchemy", test_sqlalchemy()))
    results.append(("Pandas", test_pandas_read()))
    
    print("\n" + "="*50)
    print("  测试结果汇总")
    print("="*50)
    
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*50)
    
    if all_passed:
        print("\n🎉 所有测试通过！数据库已准备就绪。")
        print("   下一步：运行数据采集脚本")
    else:
        print("\n⚠️ 部分测试失败，请检查配置。")
        print("   1. 确认 PostgreSQL 服务已启动")
        print("   2. 确认数据库密码正确")
        print("   3. 确认已执行建表SQL")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
