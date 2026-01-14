# 数据库安装指南

> 📅 创建时间：2026-01-13
> 🎯 目标：搭建 PostgreSQL + TimescaleDB 环境用于存储高频交易数据

---

## 一、安装 PostgreSQL

### Windows 安装

1. **下载安装包**
   - 访问：https://www.postgresql.org/download/windows/
   - 下载 PostgreSQL 16.x 版本（推荐最新稳定版）
   - 或使用 EDB 安装器：https://www.enterprisedb.com/downloads/postgres-postgresql-downloads

2. **安装步骤**
   - 运行安装程序
   - 选择安装目录（默认即可）
   - 设置数据目录（默认即可）
   - **设置超级用户密码**（记住这个密码！）
   - 端口保持默认 `5432`
   - 选择语言（默认即可）
   - 完成安装

3. **验证安装**
   ```powershell
   # 打开 PowerShell，输入：
   psql --version
   # 应显示类似：psql (PostgreSQL) 16.x
   ```

---

## 二、安装 TimescaleDB

### Windows 安装

1. **下载 TimescaleDB**
   - 访问：https://docs.timescale.com/self-hosted/latest/install/installation-windows/
   - 或直接下载：https://github.com/timescale/timescaledb/releases
   - 选择与你的 PostgreSQL 版本匹配的 TimescaleDB

2. **安装步骤**
   - 运行 TimescaleDB 安装程序
   - 选择你的 PostgreSQL 安装目录
   - 完成安装

3. **启用扩展**
   ```powershell
   # 连接到 PostgreSQL
   psql -U postgres
   
   # 输入密码后，执行：
   CREATE EXTENSION IF NOT EXISTS timescaledb;
   ```

---

## 三、创建数据库

### 方法一：使用 psql 命令行

```powershell
# 1. 连接到 PostgreSQL
psql -U postgres

# 2. 创建数据库
CREATE DATABASE futu_hft;

# 3. 连接到新数据库
\c futu_hft

# 4. 启用 TimescaleDB 扩展
CREATE EXTENSION IF NOT EXISTS timescaledb;

# 5. 执行建表脚本
\i 'd:/论文项目/database/init_schema.sql'
```

### 方法二：使用 pgAdmin 图形界面

1. 打开 pgAdmin（安装 PostgreSQL 时自带）
2. 右键 "Databases" → "Create" → "Database"
3. 名称填写：`futu_hft`
4. 点击 "Save"
5. 右键新数据库 → "Query Tool"
6. 打开 `init_schema.sql` 文件，执行

---

## 四、验证安装

```sql
-- 连接到 futu_hft 数据库后执行：

-- 检查 TimescaleDB 版本
SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';

-- 查看所有表
\dt

-- 查看超表信息
SELECT * FROM timescaledb_information.hypertables;
```

预期输出应显示4张表：`orderbook`, `ticker`, `quote`, `ofi_features`

---

## 五、连接信息

安装完成后，记录以下连接信息（用于Python脚本）：

```python
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'futu_hft',
    'user': 'postgres',
    'password': '你设置的密码'
}
```

---

## 六、常见问题

### Q1: psql 命令找不到？
将 PostgreSQL 的 bin 目录添加到系统 PATH：
```
C:\Program Files\PostgreSQL\16\bin
```

### Q2: TimescaleDB 扩展创建失败？
确保 TimescaleDB 版本与 PostgreSQL 版本匹配。

### Q3: 中文路径问题？
如果执行 SQL 脚本时报错，尝试将脚本复制到英文路径下执行。

---

## 七、下一步

安装完成后：
1. 执行 `init_schema.sql` 创建表结构
2. 运行数据采集脚本 `collector.py`
3. 开始采集数据！
