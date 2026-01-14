"""
富途API连接测试脚本
用于验证 OpenD 是否正常运行
"""

import sys
import io

# 设置UTF-8输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from futu import *

def test_connection():
    """测试与OpenD的连接"""
    print("="*50)
    print("  富途OpenD连接测试")
    print("="*50)
    
    # 连接本地OpenD（默认端口11111）
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    
    try:
        # 测试1：获取全局状态
        ret, data = quote_ctx.get_global_state()
        if ret == RET_OK:
            print(f"\n[OK] OpenD连接成功!")
            print(f"     市场状态: {data}")
        else:
            print(f"\n[FAIL] 连接失败: {data}")
            return False
        
        # 测试2：获取港股报价（免费）
        print("\n" + "-"*50)
        print("测试港股报价(免费)...")
        ret, data = quote_ctx.get_market_snapshot(['HK.00700'])  # 腾讯
        if ret == RET_OK:
            print(f"[OK] 港股报价获取成功!")
            print(f"     腾讯(00700) 最新价: {data['last_price'].values[0]}")
        else:
            print(f"[FAIL] 获取失败: {data}")
        
        # 测试3：获取订单簿（关键功能）
        print("\n" + "-"*50)
        print("测试订单簿数据...")
        ret, data = quote_ctx.get_order_book('HK.00700', num=5)
        if ret == RET_OK:
            print(f"[OK] 订单簿获取成功!")
            print(f"     买一: {data['Bid'][0] if len(data['Bid']) > 0 else 'N/A'}")
            print(f"     卖一: {data['Ask'][0] if len(data['Ask']) > 0 else 'N/A'}")
        else:
            print(f"[FAIL] 获取失败: {data}")
            print("     提示: 可能需要订阅行情权限")
        
        # 测试4：检查行情权限
        print("\n" + "-"*50)
        print("检查行情权限...")
        ret, data = quote_ctx.get_owner_plate(['HK.00700'])
        if ret == RET_OK:
            print(f"[OK] 权限检查通过")
        
        print("\n" + "="*50)
        print("  测试完成!")
        print("="*50)
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 发生错误: {e}")
        return False
        
    finally:
        quote_ctx.close()


if __name__ == "__main__":
    print("\n[注意] 请确保:")
    print("   1. OpenD 已启动")
    print("   2. 已登录富途账号")
    print("   3. 监听端口为 11111(默认)\n")
    
    test_connection()
