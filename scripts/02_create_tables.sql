-- ============================================================
-- OFI论文实验数据库 - 建表脚本
-- 数据库：futu_ofi
-- 创建时间：2026-01-13
-- ============================================================

-- 启用 TimescaleDB 扩展（如果尚未启用）
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================
-- 表1: 订单簿数据 (orderbook)
-- 来源: 富途API get_order_book / OrderBookHandlerBase
-- 用途: 计算OFI、订单簿深度特征
-- ============================================================

DROP TABLE IF EXISTS orderbook CASCADE;

CREATE TABLE orderbook (
    id BIGSERIAL,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- 采集时间戳
    code VARCHAR(20) NOT NULL,              -- 股票代码 如 'US.AAPL'
    name VARCHAR(50),                       -- 股票名称
    svr_recv_time_bid TIMESTAMPTZ,          -- 服务器收到买盘时间
    svr_recv_time_ask TIMESTAMPTZ,          -- 服务器收到卖盘时间
    
    -- 买盘前10档（price, volume, order_num）
    bid1_price DECIMAL(12,4), bid1_vol BIGINT, bid1_orders INT,
    bid2_price DECIMAL(12,4), bid2_vol BIGINT, bid2_orders INT,
    bid3_price DECIMAL(12,4), bid3_vol BIGINT, bid3_orders INT,
    bid4_price DECIMAL(12,4), bid4_vol BIGINT, bid4_orders INT,
    bid5_price DECIMAL(12,4), bid5_vol BIGINT, bid5_orders INT,
    bid6_price DECIMAL(12,4), bid6_vol BIGINT, bid6_orders INT,
    bid7_price DECIMAL(12,4), bid7_vol BIGINT, bid7_orders INT,
    bid8_price DECIMAL(12,4), bid8_vol BIGINT, bid8_orders INT,
    bid9_price DECIMAL(12,4), bid9_vol BIGINT, bid9_orders INT,
    bid10_price DECIMAL(12,4), bid10_vol BIGINT, bid10_orders INT,
    
    -- 卖盘前10档
    ask1_price DECIMAL(12,4), ask1_vol BIGINT, ask1_orders INT,
    ask2_price DECIMAL(12,4), ask2_vol BIGINT, ask2_orders INT,
    ask3_price DECIMAL(12,4), ask3_vol BIGINT, ask3_orders INT,
    ask4_price DECIMAL(12,4), ask4_vol BIGINT, ask4_orders INT,
    ask5_price DECIMAL(12,4), ask5_vol BIGINT, ask5_orders INT,
    ask6_price DECIMAL(12,4), ask6_vol BIGINT, ask6_orders INT,
    ask7_price DECIMAL(12,4), ask7_vol BIGINT, ask7_orders INT,
    ask8_price DECIMAL(12,4), ask8_vol BIGINT, ask8_orders INT,
    ask9_price DECIMAL(12,4), ask9_vol BIGINT, ask9_orders INT,
    ask10_price DECIMAL(12,4), ask10_vol BIGINT, ask10_orders INT,
    
    -- 原始数据备份（用于调试，生产环境可设为NULL节省空间）
    raw_data JSONB,
    
    PRIMARY KEY (ts, code)
);

-- 转为 TimescaleDB 超表（按时间自动分区）
SELECT create_hypertable('orderbook', 'ts', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_orderbook_code ON orderbook (code, ts DESC);

COMMENT ON TABLE orderbook IS '订单簿数据（10档深度）- 来自富途实时摆盘API';


-- ============================================================
-- 表2: 逐笔成交数据 (ticker)
-- 来源: 富途API get_rt_ticker / TickerHandlerBase
-- 用途: 计算成交方向、交易量、买卖压力
-- ============================================================

DROP TABLE IF EXISTS ticker CASCADE;

CREATE TABLE ticker (
    id BIGSERIAL,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- 采集时间戳
    code VARCHAR(20) NOT NULL,              -- 股票代码
    name VARCHAR(50),                       -- 股票名称
    trade_time TIMESTAMPTZ,                 -- 成交时间（API返回的time字段）
    sequence BIGINT,                        -- 逐笔序号（唯一标识）
    price DECIMAL(12,4),                    -- 成交价格
    volume BIGINT,                          -- 成交数量
    turnover DECIMAL(18,4),                 -- 成交金额
    direction VARCHAR(10),                  -- 逐笔方向: BUY/SELL/NEUTRAL
    ticker_type VARCHAR(20),                -- 逐笔类型: AUTO_MATCH/ODD_LOT等
    push_data_type VARCHAR(20),             -- 数据来源: CACHE/REALTIME
    
    PRIMARY KEY (ts, code, sequence)
);

SELECT create_hypertable('ticker', 'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_ticker_code ON ticker (code, ts DESC);
CREATE INDEX IF NOT EXISTS idx_ticker_direction ON ticker (code, direction, ts DESC);

COMMENT ON TABLE ticker IS '逐笔成交数据 - 来自富途实时逐笔API';


-- ============================================================
-- 表3: 实时报价数据 (quote)
-- 来源: 富途API get_stock_quote / StockQuoteHandlerBase
-- 用途: 获取价格快照、计算收益率
-- ============================================================

DROP TABLE IF EXISTS quote CASCADE;

CREATE TABLE quote (
    id BIGSERIAL,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- 采集时间戳
    code VARCHAR(20) NOT NULL,              -- 股票代码
    name VARCHAR(50),                       -- 股票名称
    data_date DATE,                         -- 数据日期
    data_time TIME,                         -- 数据时间
    last_price DECIMAL(12,4),               -- 最新价
    open_price DECIMAL(12,4),               -- 开盘价
    high_price DECIMAL(12,4),               -- 最高价
    low_price DECIMAL(12,4),                -- 最低价
    prev_close_price DECIMAL(12,4),         -- 昨收价
    volume BIGINT,                          -- 成交量
    turnover DECIMAL(18,4),                 -- 成交额
    turnover_rate DECIMAL(8,4),             -- 换手率
    amplitude DECIMAL(8,4),                 -- 振幅
    price_spread DECIMAL(12,4),             -- 价差
    
    -- 盘前数据
    pre_price DECIMAL(12,4),
    pre_high_price DECIMAL(12,4),
    pre_low_price DECIMAL(12,4),
    pre_volume BIGINT,
    pre_turnover DECIMAL(18,4),
    
    -- 盘后数据
    after_price DECIMAL(12,4),
    after_high_price DECIMAL(12,4),
    after_low_price DECIMAL(12,4),
    after_volume BIGINT,
    after_turnover DECIMAL(18,4),
    
    PRIMARY KEY (ts, code)
);

SELECT create_hypertable('quote', 'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_quote_code ON quote (code, ts DESC);

COMMENT ON TABLE quote IS '实时报价数据 - 来自富途实时报价API';


-- ============================================================
-- 表4: OFI特征数据 (ofi_features)
-- 来源: 由原始数据计算得出
-- 用途: 模型训练的特征表，导出为Parquet
-- ============================================================

DROP TABLE IF EXISTS ofi_features CASCADE;

CREATE TABLE ofi_features (
    ts TIMESTAMPTZ NOT NULL,                -- 时间戳（10秒聚合窗口）
    code VARCHAR(20) NOT NULL,              -- 股票代码
    
    -- ========== 价格特征 ==========
    mid_price DECIMAL(12,6),                -- 中间价 (bid1 + ask1) / 2
    spread DECIMAL(12,6),                   -- 买卖价差 ask1 - bid1
    spread_bps DECIMAL(12,4),               -- 价差基点 spread / mid_price * 10000
    return_pct DECIMAL(12,8),               -- 收益率 (当前mid - 上一个mid) / 上一个mid
    
    -- ========== OFI系列特征 ==========
    ofi_l1 DECIMAL(18,4),                   -- 单档OFI（第1档）
    ofi_l5 DECIMAL(18,4),                   -- 5档加权OFI（指数衰减）
    ofi_l10 DECIMAL(18,4),                  -- 10档加权OFI
    smart_ofi DECIMAL(18,4),                -- Smart-OFI（撤单率修正）
    
    -- ========== 订单簿形态特征 ==========
    bid_depth_5 BIGINT,                     -- 买盘5档总量
    ask_depth_5 BIGINT,                     -- 卖盘5档总量
    bid_depth_10 BIGINT,                    -- 买盘10档总量
    ask_depth_10 BIGINT,                    -- 卖盘10档总量
    depth_imbalance_5 DECIMAL(12,6),        -- 5档深度不平衡 (bid-ask)/(bid+ask)
    depth_imbalance_10 DECIMAL(12,6),       -- 10档深度不平衡
    
    -- ========== 成交特征 ==========
    buy_volume BIGINT,                      -- 窗口内买方成交量
    sell_volume BIGINT,                     -- 窗口内卖方成交量
    trade_count INT,                        -- 窗口内成交笔数
    trade_imbalance DECIMAL(12,6),          -- 成交不平衡 (buy-sell)/(buy+sell)
    avg_trade_size DECIMAL(12,4),           -- 平均成交规模
    
    -- ========== 动态特征 ==========
    ofi_ma_10 DECIMAL(18,4),                -- OFI 10期移动平均
    ofi_std_10 DECIMAL(18,4),               -- OFI 10期标准差
    ofi_zscore DECIMAL(12,4),               -- OFI Z-score
    return_ma_10 DECIMAL(12,8),             -- 收益率10期均值
    return_std_10 DECIMAL(12,8),            -- 收益率10期标准差
    
    -- ========== 标签（预测目标） ==========
    label_20 SMALLINT,                      -- 20步后方向 (-1=下跌, 0=平稳, 1=上涨)
    label_50 SMALLINT,                      -- 50步后方向
    label_100 SMALLINT,                     -- 100步后方向
    future_return_20 DECIMAL(12,8),         -- 20步后收益率（回归目标）
    future_return_50 DECIMAL(12,8),
    future_return_100 DECIMAL(12,8),
    
    PRIMARY KEY (ts, code)
);

SELECT create_hypertable('ofi_features', 'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_ofi_features_code ON ofi_features (code, ts DESC);
CREATE INDEX IF NOT EXISTS idx_ofi_features_label ON ofi_features (code, label_20, ts DESC);

COMMENT ON TABLE ofi_features IS 'OFI特征表 - 计算后的特征数据，用于模型训练';


-- ============================================================
-- 启用压缩（节省存储空间）
-- ============================================================

-- 订单簿表压缩
ALTER TABLE orderbook SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'code',
    timescaledb.compress_orderby = 'ts DESC'
);

-- 自动压缩7天前的数据
SELECT add_compression_policy('orderbook', INTERVAL '7 days', if_not_exists => TRUE);

-- 逐笔表压缩
ALTER TABLE ticker SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'code',
    timescaledb.compress_orderby = 'ts DESC'
);
SELECT add_compression_policy('ticker', INTERVAL '7 days', if_not_exists => TRUE);

-- 报价表压缩
ALTER TABLE quote SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'code',
    timescaledb.compress_orderby = 'ts DESC'
);
SELECT add_compression_policy('quote', INTERVAL '7 days', if_not_exists => TRUE);


-- ============================================================
-- 验证建表结果
-- ============================================================

-- 查看所有表
SELECT tablename FROM pg_tables WHERE schemaname = 'public';

-- 查看 TimescaleDB 超表
SELECT hypertable_name, num_dimensions 
FROM timescaledb_information.hypertables;

-- 查看压缩策略
SELECT * FROM timescaledb_information.compression_settings;


-- ============================================================
-- 完成提示
-- ============================================================
DO $$
BEGIN
    RAISE NOTICE '========================================';
    RAISE NOTICE '✅ 数据库表创建完成！';
    RAISE NOTICE '  - orderbook: 订单簿数据（10档深度）';
    RAISE NOTICE '  - ticker: 逐笔成交数据';
    RAISE NOTICE '  - quote: 实时报价数据';
    RAISE NOTICE '  - ofi_features: OFI特征数据';
    RAISE NOTICE '========================================';
END $$;
